"""Make a small train/valid QASPER sample for completion-masked QLoRA smoke tests."""
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "curated"
OUT = ROOT / "feasibility_data"
MODEL = ROOT / ".cache" / "qwen3-0.6b-4bit"
MAX_SEQUENCE = 384
TARGETS = {"train": {"answerable": 24, "unanswerable": 8},
           "validation": {"answerable": 4, "unanswerable": 4}}
INSTRUCTIONS = ("Answer the question using only the provided research-paper passages. "
                "If the passages do not establish an answer, output exactly UNANSWERABLE. "
                "Otherwise output only a concise answer. Do not invent citations or page numbers.")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def answer_and_passage(q):
    """Require an exact source paragraph and an answer string inside it."""
    for p in q["evidence_passages"]:
        if not p["source_match"] or not p["text"] or len(p["text"]) > 850:
            continue
        content = norm(p["text"])
        for answer in sorted(q["expected_answers"], key=len):
            if 1 <= len(answer) <= 120 and norm(answer) and norm(answer) in content:
                return answer, p
    return None, None


def unanswerable_passage(doc):
    for section in doc["sections"]:
        for i, paragraph in enumerate(section["paragraphs"]):
            if paragraph and 80 <= len(paragraph) <= 850:
                return {"evidence_id": f"excerpt:{section['section_index']}:{i}",
                        "text": paragraph, "section_index": section["section_index"],
                        "paragraph_index": i, "source_type": "document_excerpt_not_gold"}
    return None


def make_messages(q, doc, passage):
    context = f"[section {passage['section_index']}, evidence {passage['evidence_id']}] {passage['text']}"
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"Document: {q['document_id']} — {doc['title']}\nPassages:\n{context}\n\nQuestion: {q['question']}"}]


def select(split, tokenizer, blocked_docs):
    docs = {d["document_id"]: d for d in rows(DATA / f"{split}_documents.jsonl")}
    chosen_docs = set(blocked_docs)
    selected = []
    counts = {"answerable": 0, "unanswerable": 0}
    for q in rows(DATA / f"{split}_questions.jsonl"):
        label = q["answerability"]
        if counts[label] >= TARGETS[split][label] or q["document_id"] in chosen_docs:
            continue
        doc = docs[q["document_id"]]
        if label == "answerable":
            answer, passage = answer_and_passage(q)
            if passage is None:
                continue
        else:
            answer = "UNANSWERABLE"
            passage = unanswerable_passage(doc)
            if passage is None:
                continue
        messages = make_messages(q, doc, passage)
        complete = messages + [{"role": "assistant", "content": answer}]
        full_ids = tokenizer.apply_chat_template(complete, enable_thinking=False)["input_ids"]
        prefix_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   enable_thinking=False)["input_ids"]
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise RuntimeError("Qwen chat-template prefix mismatch")
        if len(full_ids) > MAX_SEQUENCE or len(full_ids) <= len(prefix_ids) + 1:
            continue
        selected.append({"messages": complete, "split": split, "document_id": q["document_id"],
                         "question_id": q["question_id"], "answerability": label,
                         "evidence_id": passage["evidence_id"], "source_revision": q["source_revision"],
                         "full_tokens": len(full_ids), "prompt_tokens": len(prefix_ids),
                         "completion_tokens": len(full_ids) - len(prefix_ids)})
        chosen_docs.add(q["document_id"])
        counts[label] += 1
    if counts != TARGETS[split]:
        raise RuntimeError(f"insufficient clean {split} examples: {counts}")
    return selected


def build():
    OUT.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    baseline_docs = {q["document_id"] for q in rows(ROOT / "validation_gold_eval.jsonl")}
    train = select("train", tokenizer, set())
    valid = select("validation", tokenizer, baseline_docs)
    for name, data in (("train", train), ("valid", valid)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in data:
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"source_curated_manifest_sha256": hashlib.sha256((ROOT / "curated_manifest.json").read_bytes()).hexdigest(),
                "model_revision": "73e3e38d981303bc594367cd910ea6eb48349da8",
                "selection": "source order, distinct documents; answerable answer substring of matched source paragraph; unanswerable consensus; validation excludes baseline documents",
                "max_sequence_tokens": MAX_SEQUENCE, "counts": {"train": TARGETS["train"], "valid": TARGETS["validation"]},
                "file_sha256": {name: hashlib.sha256((OUT / f"{name}.jsonl").read_bytes()).hexdigest() for name in ("train", "valid")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()["counts"], indent=2))
