"""Build bounded QASPER train/validation records; never opens test files."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "v1/experiment.json"
OUT = HERE / "data/v1"
CURATED = ROOT / "curated"
TOKENIZER = ROOT / ".cache/qwen3-0.6b-4bit"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def source_paragraph(doc, section_index, paragraph_index):
    section = next((s for s in doc["sections"] if s["section_index"] == section_index), None)
    if section is None or not 0 <= paragraph_index < len(section["paragraphs"]):
        return None
    return section["paragraphs"][paragraph_index]


def answerable_evidence(q, doc):
    evidence = q["evidence_passages"]
    if not evidence or any(not p["source_match"] for p in evidence):
        return None, None, "missing_or_unmatched_evidence"
    unique = {}
    for p in evidence:
        key = (p["section_index"], p["paragraph_index"])
        if not p["text"] or source_paragraph(doc, *key) != p["text"]:
            return None, None, "source_paragraph_mismatch"
        unique.setdefault(key, {"text": p["text"], "ids": []})["ids"].append(p["evidence_id"])
    if len(unique) != 1:
        return None, None, "multiple_distinct_gold_paragraphs"
    (section, paragraph), record = next(iter(unique.items()))
    candidate_answers = sorted(set(q["expected_answers"]), key=lambda a: (len(a), a))
    target = next((a for a in candidate_answers if 1 <= len(a) <= 120 and norm(a)
                   and norm(a) in norm(record["text"])), None)
    if target is None:
        return None, None, "reference_not_in_gold_paragraph"
    passage = {"section_index": section, "paragraph_index": paragraph,
               "evidence_ids": sorted(record["ids"]), "text": record["text"],
               "source_type": "qasper_gold_source_paragraph", "source_match": True}
    return target, passage, None


def unanswerable_excerpt(doc):
    for section in doc["sections"]:
        for i, paragraph in enumerate(section["paragraphs"]):
            if paragraph and 80 <= len(paragraph) <= 650:
                return {"section_index": section["section_index"], "paragraph_index": i,
                        "evidence_ids": [], "text": paragraph,
                        "source_type": "document_excerpt_not_gold", "source_match": True}
    return None


def make_row(q, doc, tokenizer, config):
    if q["answerability"] == "answerable":
        target, passage, error = answerable_evidence(q, doc)
        if error:
            return None, error
    elif q["answerability"] == "unanswerable":
        target = "UNANSWERABLE"
        passage = unanswerable_excerpt(doc)
        if passage is None:
            return None, "no_bounded_document_excerpt"
    else:
        return None, "invalid_answerability"
    p = passage
    passage_id = f"section:{p['section_index']}:paragraph:{p['paragraph_index']}"
    prompt = [{"role": "system", "content": config["system_instruction"]},
              {"role": "user", "content":
               f"Document: {q['document_id']} — {doc['title']}\n"
               f"Passage [{passage_id}]: {p['text']}\n\nQuestion: {q['question']}"}]
    supervised_target = (f"{target} [{passage_id}]" if q["answerability"] == "answerable"
                         else "UNANSWERABLE")
    messages = prompt + [{"role": "assistant", "content": supervised_target}]
    prefix = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                           enable_thinking=False)["input_ids"]
    full = tokenizer.apply_chat_template(messages, enable_thinking=False)["input_ids"]
    if full[:len(prefix)] != prefix or len(full) <= len(prefix) + 1:
        raise RuntimeError(f"Qwen template/mask mismatch: {q['question_id']}")
    max_seq = config["maximum_sequence_tokens"]
    if len(full) > max_seq or len(prefix) + config["maximum_generation_tokens"] > max_seq:
        return None, "full_context_exceeds_token_budget"
    return {"schema_version": 1, "split": q["split"], "document_id": q["document_id"],
            "question_id": q["question_id"], "question": q["question"],
            "answerability": q["answerability"], "expected_answers": q["expected_answers"],
            "target_answer": supervised_target, "reference_answer": target,
            "reference_citation": passage_id if q["answerability"] == "answerable" else None,
            "messages": messages,
            "selected_passages": [{**p, "passage_id": passage_id,
                                   "sha256": sha(p["text"].encode())}],
            "prompt_tokens": len(prefix), "full_tokens": len(full),
            "completion_tokens": len(full) - len(prefix),
            "provenance": {"dataset": q["source_dataset"],
                           "source_revision": q["source_revision"],
                           "parquet_revision": q["parquet_revision"],
                           "annotation_ids": [a["annotation_id"] for a in q["annotations"]],
                           "source_split": q["split"], "page_numbers_available": False}}, None


def build_split(split, tokenizer, config, blocked):
    docs = {d["document_id"]: d for d in rows(CURATED / f"{split}_documents.jsonl")}
    counts = Counter()
    skipped = Counter()
    selected = []
    decisions = []
    used_docs = set(blocked)
    target = config["targets"][split]
    for q in rows(CURATED / f"{split}_questions.jsonl"):
        label = q["answerability"]
        if all(counts[a] >= n for a, n in target.items()):
            break
        if counts[label] >= target[label] or q["document_id"] in used_docs:
            continue
        row, error = make_row(q, docs[q["document_id"]], tokenizer, config)
        if error:
            skipped[error] += 1
            decisions.append({"question_id": q["question_id"], "document_id": q["document_id"],
                              "split": split, "decision": "skip", "reason": error})
            continue
        selected.append(row)
        used_docs.add(q["document_id"])
        counts[label] += 1
        decisions.append({"question_id": q["question_id"], "document_id": q["document_id"],
                          "split": split, "decision": "include", "reason": "eligible"})
    if dict(counts) != target:
        raise RuntimeError(f"Cannot reach {split} targets: {dict(counts)}; skipped={dict(skipped)}")
    return selected, decisions, dict(skipped)


def write_jsonl(path, data):
    with path.open("w", encoding="utf-8") as stream:
        for row in data:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    baseline_docs = {q["document_id"] for q in rows(ROOT / "validation_gold_eval.jsonl")}
    train, train_decisions, train_skips = build_split("train", tokenizer, config, set())
    valid, valid_decisions, valid_skips = build_split("validation", tokenizer, config, baseline_docs)
    if {r["document_id"] for r in train} & {r["document_id"] for r in valid}:
        raise RuntimeError("Cross-split document overlap")
    OUT.mkdir(parents=True, exist_ok=True)
    review = []
    for label in ("answerable", "unanswerable"):
        for row in [x for x in valid if x["answerability"] == label][:5]:
            review.append({"question_id": row["question_id"],
                           "document_id": row["document_id"], "answerability": label,
                           "question": row["question"],
                           "reference_answer": row["reference_answer"],
                           "passages": row["selected_passages"],
                           "manual_evidence_support": None,
                           "review_status": "pending"})
    files = {"train.jsonl": train, "validation_gold.jsonl": valid,
             "selection_audit.jsonl": train_decisions + valid_decisions,
             "evidence_review_sample.jsonl": review}
    for name, data in files.items():
        write_jsonl(OUT / name, data)
    tok_config = json.loads((TOKENIZER / "tokenizer_config.json").read_text())
    manifest = {"schema_version": 1, "config_sha256": sha(CONFIG.read_bytes()),
                "curated_manifest_sha256": sha((ROOT / "curated_manifest.json").read_bytes()),
                "tokenizer_config_sha256": sha((TOKENIZER / "tokenizer_config.json").read_bytes()),
                "chat_template_sha256": sha(tok_config["chat_template"].encode()),
                "source_dataset": config["source_dataset"],
                "source_revision": config["source_revision"],
                "parquet_revision": config["parquet_revision"],
                "dataset_card_license": "CC BY 4.0",
                "attribution": "Dasigi et al. (2021), QASPER: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
                "rights_limitation": "QASPER dataset license does not independently establish redistribution rights for every underlying S2ORC paper",
                "counts": {"train": dict(Counter(x["answerability"] for x in train)),
                           "validation": dict(Counter(x["answerability"] for x in valid))},
                "skipped_examined_candidates": {"train": train_skips, "validation": valid_skips},
                "file_sha256": {name: sha((OUT / name).read_bytes()) for name in files},
                "test_files_read": False, "context_truncation": "none; over-budget candidates skipped"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()["counts"], indent=2))
