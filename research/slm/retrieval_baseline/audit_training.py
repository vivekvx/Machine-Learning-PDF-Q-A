"""Audit all eligible curated TRAIN questions without opening validation/test answers."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SLM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLM / "portable"))
from prepare import make_row, rows
from transformers import AutoTokenizer


def audit():
    config = json.loads((SLM / "portable/v1/experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(SLM / ".cache/qwen3-0.6b-4bit", local_files_only=True)
    docs = {d["document_id"]: d for d in rows(SLM / "curated/train_documents.jsonl")}
    selected = list(rows(SLM / "portable/data/v1/train.jsonl"))
    selected_ids = {r["question_id"] for r in selected}
    selected_docs = {r["document_id"] for r in selected}
    eligible_by_doc = defaultdict(list)
    rejected = Counter()
    eligible_questions = Counter()
    for question in rows(SLM / "curated/train_questions.jsonl"):
        doc = docs.get(question["document_id"])
        if doc is None:
            rejected["document_not_in_included_train_cohort"] += 1
            continue
        row, reason = make_row(question, doc, tokenizer, config)
        if reason:
            rejected[reason] += 1
        else:
            eligible_questions[row["answerability"]] += 1
            eligible_by_doc[row["document_id"]].append(row)
    expansion_docs = {d: value for d, value in eligible_by_doc.items() if d not in selected_docs}
    expansion_labels = Counter()
    for candidates in expansion_docs.values():
        expansion_labels[candidates[0]["answerability"]] += 1
    result = {
        "selected_examples": len(selected), "selected_unique_documents": len(selected_docs),
        "selected_answerability": dict(Counter(r["answerability"] for r in selected)),
        "selected_answerable_source_matched": sum(r["answerability"] == "answerable" and
                                                   r["selected_passages"][0]["source_match"] for r in selected),
        "selected_unanswerable_excerpt_not_gold": sum(r["answerability"] == "unanswerable" and
                                                        r["selected_passages"][0]["source_type"] == "document_excerpt_not_gold"
                                                        for r in selected),
        "eligible_questions_all_included_train_docs": dict(eligible_questions),
        "eligible_unique_documents_all": len(eligible_by_doc),
        "additional_eligible_documents_outside_selected_352": len(expansion_docs),
        "additional_one_per_document_label_first_source_order": dict(expansion_labels),
        "additional_question_ids_first_source_order": [candidates[0]["question_id"]
                                                       for candidates in expansion_docs.values()],
        "rejected_all_examined_train_questions": dict(rejected),
        "selected_ids_all_eligible": selected_ids.issubset({r["question_id"] for values in eligible_by_doc.values()
                                                               for r in values}),
        "test_files_read": False,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite versioned audit")
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "additional_question_ids_first_source_order"}, indent=2))
