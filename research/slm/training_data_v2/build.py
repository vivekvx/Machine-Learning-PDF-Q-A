"""Train-only evidence audit, review queue, candidate ledger, and frozen IDs.

No model inference, training, or held-out test-question files are read.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
sys.path.insert(0, str(SLM / "portable"))
from prepare import make_row

OUT = HERE / "v1"
TRAIN = SLM / "portable/data/v1/train.jsonl"
VALID = SLM / "portable/data/v1/validation_gold.jsonl"
QUESTIONS = SLM / "curated/train_questions.jsonl"
DOCUMENTS = SLM / "curated/train_documents.jsonl"
DECISIONS = SLM / "curated/document_decisions.jsonl"
TOKENIZER = SLM / ".cache/qwen3-0.6b-4bit"
CONFIG = OUT / "config.json"
REVIEW = SLM / "EVIDENCE_REVIEW_SAMPLE.csv"
STOP = {"what", "which", "where", "when", "who", "whom", "whose", "why", "how", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "of", "for", "to", "in", "on", "by", "with", "and", "or", "their", "this", "that", "they", "it", "be", "from", "as"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return " ".join(re.findall(r"\w+", str(text).casefold()))


def terms(text):
    return {word for word in re.findall(r"\w+", str(text).casefold())
            if word not in STOP and len(word) > 1}


def answer_polarity(answer):
    value = norm(answer)
    return value if value in {"yes", "no"} else None


def signals(row, question, doc):
    flags = []
    passages = row["selected_passages"]
    if len(passages) != 1 or not passages[0].get("text"):
        flags.append("missing_selected_passage")
        passage = None
        text = ""
    else:
        passage = passages[0]
        text = passage["text"]
        section = next((s for s in doc["sections"] if s["section_index"] == passage["section_index"]), None)
        if section is None or not 0 <= passage["paragraph_index"] < len(section["paragraphs"]) or section["paragraphs"][passage["paragraph_index"]] != text:
            flags.append("source_paragraph_mismatch")
        if hashlib.sha256(text.encode()).hexdigest() != passage["sha256"]:
            flags.append("passage_hash_mismatch")
    annotations = question["annotations"]
    annotation_ids = [a["annotation_id"] for a in annotations]
    if annotation_ids != row["provenance"]["annotation_ids"]:
        flags.append("annotation_id_mismatch")
    if row["answerability"] != question["answerability"] or row["split"] != question["split"]:
        flags.append("label_or_split_mismatch")
    qterms = terms(row["question"])
    pterms = terms(text)
    overlap = len(qterms & pterms) / len(qterms) if qterms else None
    if row["answerability"] == "answerable":
        original_evidence = question["evidence_passages"]
        if not original_evidence or not passage or not passage.get("evidence_ids"):
            flags.append("missing_original_evidence")
        elif set(passage["evidence_ids"]) != {e["evidence_id"] for e in original_evidence}:
            flags.append("evidence_id_mismatch")
        target = norm(row["reference_answer"])
        if not target or target not in norm(text):
            flags.append("reference_not_in_selected_passage")
        absent_variants = [answer for answer in question["expected_answers"] if norm(answer) not in norm(text)]
        if absent_variants:
            flags.append("reference_variant_not_in_passage")
        variants = [terms(answer) for answer in question["expected_answers"]]
        if len(variants) > 1 and any(len(x & y) / len(x | y) < 0.5 for i, x in enumerate(variants)
                                      for y in variants[i + 1:] if x | y):
            flags.append("answer_variant_low_overlap")
        yes_no_values = {a["yes_no"] for a in annotations if a["yes_no"] is not None}
        if len(yes_no_values) > 1 or (answer_polarity(row["reference_answer"]) is not None and
                                      yes_no_values and answer_polarity(row["reference_answer"]) !=
                                      ("yes" if True in yes_no_values else "no")):
            flags.append("possible_annotation_contradiction")
        if answer_polarity(row["reference_answer"]) and {"yes", "no"}.issubset(pterms):
            flags.append("possible_passage_polarity_conflict")
        if len(terms(row["reference_answer"])) <= 1:
            flags.append("short_reference_weak_match")
        if overlap is not None and overlap < 0.2:
            flags.append("low_question_passage_overlap")
    else:
        absent_variants = []
        if question["evidence_passages"] or (passage and passage.get("evidence_ids")):
            flags.append("unexpected_unanswerable_gold_evidence")
        flags.append("unanswerable_excerpt_not_proof")
        if overlap is not None and overlap >= 0.5:
            flags.append("unanswerable_high_question_overlap")
    categories = {
        "missing": any(flag in flags for flag in ("missing_selected_passage", "missing_original_evidence", "source_paragraph_mismatch", "passage_hash_mismatch", "evidence_id_mismatch")),
        "possible_contradiction": any(flag in flags for flag in ("possible_annotation_contradiction", "possible_passage_polarity_conflict")),
        "ambiguous": any(flag in flags for flag in ("answer_variant_low_overlap", "reference_variant_not_in_passage", "short_reference_weak_match")),
        "potentially_insufficient": any(flag in flags for flag in ("reference_not_in_selected_passage", "low_question_passage_overlap", "unanswerable_excerpt_not_proof")),
    }
    return {"flags": sorted(set(flags)), "categories": categories,
            "question_passage_term_overlap": overlap,
            "reference_variants_absent_from_passage": absent_variants,
            "automatic_support_status": "unverified_requires_human_review"}


def audit_row(row, question, doc):
    check = signals(row, question, doc)
    passage = row["selected_passages"][0]
    section = next(s for s in doc["sections"] if s["section_index"] == passage["section_index"])
    return {"question_id": row["question_id"], "document_id": row["document_id"],
            "split": "train", "title": doc["title"], "abstract": doc["abstract"],
            "section_name": section["section_name"], "question": row["question"],
            "answerability": row["answerability"], "reference_answer": row["reference_answer"],
            "expected_answers": row["expected_answers"], "target_answer": row["target_answer"],
            "selected_passage": passage, "original_annotations": question["annotations"],
            "original_evidence_passages": question["evidence_passages"],
            "provenance": row["provenance"], "prompt_tokens": row["prompt_tokens"],
            "full_tokens": row["full_tokens"], **check}


def stable_order(records, seed):
    return sorted(records, key=lambda x: (hashlib.sha256(f"{seed}|{x['question_id']}".encode()).hexdigest(),
                                          x["question_id"]))


def select_review(audited, config):
    seed = config["manual_review_sample"]["seed"]
    selections = []
    strata_counts = {}
    for label, goal in (("answerable", 40), ("unanswerable", 24)):
        subset = [r for r in audited if r["answerability"] == label]
        if label == "answerable":
            high = [r for r in subset if any(r["categories"].values())]
        else:
            high = [r for r in subset if "unanswerable_high_question_overlap" in r["flags"]]
        high_ids = {r["question_id"] for r in high}
        low = [r for r in subset if r["question_id"] not in high_ids]
        chosen_high = stable_order(high, seed)[:goal // 2]
        chosen_low = stable_order(low, seed)[:goal // 2]
        chosen_ids = {r["question_id"] for r in chosen_high + chosen_low}
        remainder = stable_order([r for r in subset if r["question_id"] not in chosen_ids], seed)
        chosen = chosen_high + chosen_low + remainder[:goal - len(chosen_ids)]
        if len(chosen) != goal:
            raise RuntimeError(f"Insufficient {label} review records")
        strata_counts[label] = {"high_risk": sum(r["question_id"] in high_ids for r in chosen),
                                "lower_risk": sum(r["question_id"] not in high_ids for r in chosen)}
        selections += stable_order(chosen, seed)
    return selections, strata_counts


def write_jsonl(path, records):
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_review_csv(path, records):
    fields = ["question_id", "document_id", "split", "answerability", "title", "abstract",
              "section_name", "question", "reference_answer", "expected_answers_json",
              "selected_passage_id", "selected_passage_text", "selected_passage_sha256",
              "original_annotations_json", "original_evidence_passages_json", "automatic_flags_json",
              "reviewer_id", "human_support_judgment", "human_evidence_condition",
              "human_answerability_judgment", "human_contradiction_judgment",
              "human_medical_domain_judgment", "human_rights_judgment", "review_notes", "reviewed_at"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            passage = row["selected_passage"]
            writer.writerow({"question_id": row["question_id"], "document_id": row["document_id"],
                             "split": "train", "answerability": row["answerability"],
                             "title": row["title"], "abstract": row["abstract"],
                             "section_name": row["section_name"], "question": row["question"],
                             "reference_answer": row["reference_answer"],
                             "expected_answers_json": json.dumps(row["expected_answers"], ensure_ascii=False),
                             "selected_passage_id": passage["passage_id"],
                             "selected_passage_text": passage["text"],
                             "selected_passage_sha256": passage["sha256"],
                             "original_annotations_json": json.dumps(row["original_annotations"], ensure_ascii=False),
                             "original_evidence_passages_json": json.dumps(row["original_evidence_passages"], ensure_ascii=False),
                             "automatic_flags_json": json.dumps(row["flags"]),
                             "reviewer_id": "", "human_support_judgment": "", "human_evidence_condition": "",
                             "human_answerability_judgment": "", "human_contradiction_judgment": "",
                             "human_medical_domain_judgment": "", "human_rights_judgment": "",
                             "review_notes": "", "reviewed_at": ""})


def document_text_hash(doc):
    text = "\n".join([doc["title"], doc["abstract"] or ""] +
                     [p for section in doc["sections"] for p in section["paragraphs"]])
    return hashlib.sha256(norm(text).encode()).hexdigest()


def duplicate_audit(docs, considered_ids):
    exact = defaultdict(list)
    for doc_id in considered_ids:
        exact[document_text_hash(docs[doc_id])].append(doc_id)
    exact_pairs = [{"document_ids": ids, "rule": "normalized_full_document_sha256"}
                   for ids in exact.values() if len(ids) > 1]
    titles = [(doc_id, norm(docs[doc_id]["title"])) for doc_id in sorted(considered_ids)]
    near_pairs = []
    for i, (left_id, left) in enumerate(titles):
        for right_id, right in titles[i + 1:]:
            if abs(len(left) - len(right)) > max(len(left), len(right)) * 0.12:
                continue
            similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
            if similarity >= 0.92:
                near_pairs.append({"left_document_id": left_id, "right_document_id": right_id,
                                   "normalized_title_similarity": round(similarity, 6),
                                   "rule": "title_similarity_at_least_0.92_review_signal"})
    return exact_pairs, near_pairs


def variant_plan(row):
    answerable = row["answerability"] == "answerable"
    if answerable:
        schedules = {"standard_lora": ["sufficient"] * 4,
                     "raft_distractor_lora": ["sufficient_gold_only", "sufficient_with_one_distractor",
                                              "sufficient_with_two_distractors", "sufficient_position_shift"],
                     "evidence_sufficiency_lora": ["sufficient", "partial", "irrelevant", "absent"]}
    else:
        schedules = {"standard_lora": ["document_unanswerable_excerpt"] * 4,
                     "raft_distractor_lora": ["irrelevant_excerpt", "irrelevant_one_distractor",
                                              "irrelevant_two_distractors", "absent"],
                     "evidence_sufficiency_lora": ["irrelevant", "irrelevant", "absent", "absent"]}
    return {"question_id": row["question_id"], "document_id": row["document_id"],
            "source_split": "train", "original_answerability": row["answerability"],
            "source_passage_id": row["selected_passages"][0]["passage_id"],
            "source_passage_sha256": row["selected_passages"][0]["sha256"],
            "annotation_ids": row["provenance"]["annotation_ids"],
            "arm_exposure_conditions": schedules,
            "materialization_status": "blocked_pending_independent_evidence_and_rights_review",
            "negative_passage_ids": None, "partial_redaction": None,
            "targets": {"sufficient": row["target_answer"] if answerable else None,
                        "insufficient_context": "UNANSWERABLE"}}


def build(output_dir=OUT, review_path=REVIEW):
    config = json.loads(CONFIG.read_text())
    for path, key in ((TRAIN, "train_source_sha256"), (VALID, "validation_source_sha256"),
                      (QUESTIONS, "curated_train_questions_sha256"),
                      (DOCUMENTS, "curated_train_documents_sha256")):
        if digest(path) != config[key]:
            raise RuntimeError(f"Pinned source changed: {path}")
    outputs = [output_dir / name for name in ("selected_audit.jsonl", "candidate_expansion.jsonl",
               "duplicate_review_signals.jsonl", "license_ledger.jsonl", "variant_plan.jsonl",
               "train_ids.jsonl", "validation_ids.jsonl", "manifest.json")] + [review_path]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Versioned output already exists; do not overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(rows(TRAIN))
    valid = list(rows(VALID))
    questions = list(rows(QUESTIONS))
    by_qid = {q["question_id"]: q for q in questions}
    docs = {d["document_id"]: d for d in rows(DOCUMENTS)}
    decisions = {d["document_id"]: d for d in rows(DECISIONS) if d["split"] == "train"}
    if len(selected) != config["train_question_count"] or len(valid) != config["validation_question_count"]:
        raise RuntimeError("Frozen selected cohort count drift")
    if len({r["document_id"] for r in selected}) != len(selected) or len({r["document_id"] for r in valid}) != len(valid):
        raise RuntimeError("Document reuse within a frozen cohort")
    if any(r["split"] != "train" or decisions[r["document_id"]]["status"] != "include" for r in selected):
        raise RuntimeError("Selected row outside included train documents")
    audited = [audit_row(row, by_qid[row["question_id"]], docs[row["document_id"]]) for row in selected]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    experiment = json.loads((SLM / "portable/v1/experiment.json").read_text())
    selected_doc_ids = {r["document_id"] for r in selected}
    eligible_by_doc = defaultdict(list)
    rejected = Counter()
    rejected_by_doc = defaultdict(Counter)
    for question in questions:
        row, reason = make_row(question, docs[question["document_id"]], tokenizer, experiment)
        if reason:
            rejected[reason] += 1
            rejected_by_doc[question["document_id"]][reason] += 1
        else:
            eligible_by_doc[row["document_id"]].append(row)
    expansion = []
    for doc_id, eligible in eligible_by_doc.items():
        if doc_id in selected_doc_ids:
            continue
        row = eligible[0]
        q = by_qid[row["question_id"]]
        detail = audit_row(row, q, docs[doc_id])
        expansion.append({"question_id": row["question_id"], "document_id": doc_id,
                          "title": docs[doc_id]["title"], "answerability": row["answerability"],
                          "prompt_tokens": row["prompt_tokens"], "full_tokens": row["full_tokens"],
                          "passage_id": row["selected_passages"][0]["passage_id"],
                          "passage_sha256": row["selected_passages"][0]["sha256"],
                          "annotation_ids": row["provenance"]["annotation_ids"],
                          "automated_flags": detail["flags"],
                          "other_eligible_questions_in_document": len(eligible) - 1,
                          "rejected_questions_in_document": dict(rejected_by_doc[doc_id]),
                          "paper_training_rights": "unverified",
                          "addition_status": "candidate_only_not_added_pending_review"})
    if len(expansion) != 122:
        raise RuntimeError(f"Expected 122 extra eligible documents; found {len(expansion)}")
    considered = selected_doc_ids | {r["document_id"] for r in expansion}
    exact, near = duplicate_audit(docs, considered)
    duplicate_signals = exact + near
    sample, strata = select_review(audited, config)
    ledger = []
    candidate_doc_ids = {r["document_id"] for r in expansion}
    for doc_id, decision in decisions.items():
        role = ("selected_352" if doc_id in selected_doc_ids else
                "additional_122_candidate" if doc_id in candidate_doc_ids else
                "excluded_medical_by_study_policy" if decision["status"] == "exclude" else
                "quarantined_by_study_policy" if decision["status"] == "quarantine" else
                "included_not_selected")
        ledger.append({"document_id": doc_id, "split": "train", "title": decision["title"],
                       "document_filter_status": decision["status"], "document_filter_reasons": decision["reasons"],
                       "study_role": role, "dataset_license": "CC BY 4.0 (pinned QASPER card)",
                       "underlying_paper_training_rights": "unverified",
                       "individual_paper_license_url": None,
                       "rights_review_status": "pending_independent_review",
                       "study_policy_restriction": decision["status"] != "include"})
    write_jsonl(output_dir / "selected_audit.jsonl", audited)
    write_jsonl(output_dir / "candidate_expansion.jsonl", expansion)
    write_jsonl(output_dir / "duplicate_review_signals.jsonl", duplicate_signals)
    write_jsonl(output_dir / "license_ledger.jsonl", ledger)
    write_jsonl(output_dir / "variant_plan.jsonl", [variant_plan(r) for r in selected])
    write_jsonl(output_dir / "train_ids.jsonl", [{"question_id": r["question_id"], "document_id": r["document_id"],
                                          "answerability": r["answerability"], "source_split": "train"} for r in selected])
    write_jsonl(output_dir / "validation_ids.jsonl", [{"question_id": r["question_id"], "document_id": r["document_id"],
                                               "answerability": r["answerability"], "source_split": "validation"} for r in valid])
    write_review_csv(review_path, sample)
    files = {path.name: digest(path) for path in outputs if path.name != "manifest.json"}
    flag_counts = Counter(flag for r in audited for flag in r["flags"])
    category_counts = Counter(key for r in audited for key, value in r["categories"].items() if value)
    manifest = {"schema_version": 1, "status": "automated_audit_only_no_human_judgments",
                "config_sha256": digest(CONFIG), "source_sha256": {path.name: digest(path) for path in
                                    (TRAIN, VALID, QUESTIONS, DOCUMENTS, DECISIONS)},
                "selected_count": len(selected), "selected_answerability": dict(Counter(r["answerability"] for r in selected)),
                "review_sample_count": len(sample), "review_strata": strata,
                "automatic_flag_counts_selected": dict(flag_counts),
                "automatic_category_counts_selected": dict(category_counts),
                "additional_candidate_documents": len(expansion),
                "additional_candidate_answerability": dict(Counter(r["answerability"] for r in expansion)),
                "all_train_question_rejections": dict(rejected),
                "all_train_question_eligible": sum(len(v) for v in eligible_by_doc.values()),
                "eligible_unique_train_documents": len(eligible_by_doc),
                "exact_duplicate_document_groups_considered": len(exact),
                "near_duplicate_title_pairs_considered": len(near),
                "license_ledger_status_counts": dict(Counter(r["study_role"] for r in ledger)),
                "file_sha256": files,
                "tool_sha256": {
                    "training_data_v2/build.py": digest(HERE / "build.py"),
                    "portable/prepare.py": digest(SLM / "portable/prepare.py"),
                    "training_data_v2/abstention_metrics.py": digest(HERE / "abstention_metrics.py"),
                },
                "test_question_or_answer_files_read": False,
                "training_executed": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train-only audit in an empty output location")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--review-csv", type=Path, default=REVIEW)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.review_csv), indent=2))
