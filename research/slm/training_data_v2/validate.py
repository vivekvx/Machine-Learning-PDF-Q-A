"""Read-only validation of frozen train-only audit artifacts."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
OUT = HERE / "v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate():
    config = json.loads((OUT / "config.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert digest(OUT / "config.json") == manifest["config_sha256"]
    sources = {
        "train.jsonl": SLM / "portable/data/v1/train.jsonl",
        "validation_gold.jsonl": SLM / "portable/data/v1/validation_gold.jsonl",
        "train_questions.jsonl": SLM / "curated/train_questions.jsonl",
        "train_documents.jsonl": SLM / "curated/train_documents.jsonl",
        "document_decisions.jsonl": SLM / "curated/document_decisions.jsonl",
    }
    for name, path in sources.items():
        assert digest(path) == manifest["source_sha256"][name]
    assert digest(sources["train.jsonl"]) == config["train_source_sha256"]
    assert digest(sources["validation_gold.jsonl"]) == config["validation_source_sha256"]
    for name, expected in manifest["file_sha256"].items():
        path = SLM / name if name == "EVIDENCE_REVIEW_SAMPLE.csv" else OUT / name
        assert digest(path) == expected, name
    for name, expected in manifest["tool_sha256"].items():
        assert digest(SLM / name) == expected, name
    train = rows(sources["train.jsonl"])
    valid = rows(sources["validation_gold.jsonl"])
    questions = {q["question_id"]: q for q in rows(sources["train_questions.jsonl"])}
    docs = {d["document_id"]: d for d in rows(sources["train_documents.jsonl"])}
    audit = rows(OUT / "selected_audit.jsonl")
    train_ids = rows(OUT / "train_ids.jsonl")
    valid_ids = rows(OUT / "validation_ids.jsonl")
    plans = rows(OUT / "variant_plan.jsonl")
    assert len(train) == len(audit) == len(train_ids) == len(plans) == 352
    assert config["training"]["optimizer_steps"] == len(train) * config["training"]["exposures_per_question_per_arm"]
    assert len(valid) == len(valid_ids) == 80
    assert [r["question_id"] for r in train] == [r["question_id"] for r in audit] == [r["question_id"] for r in train_ids] == [r["question_id"] for r in plans]
    assert [r["question_id"] for r in valid] == [r["question_id"] for r in valid_ids]
    assert len({r["document_id"] for r in train}) == 352
    assert len({r["document_id"] for r in valid}) == 80
    assert not {r["document_id"] for r in train} & {r["document_id"] for r in valid}
    for original, record, plan in zip(train, audit, plans):
        raw = questions[original["question_id"]]
        assert record["original_annotations"] == raw["annotations"]
        assert record["original_evidence_passages"] == raw["evidence_passages"]
        assert record["selected_passage"] == original["selected_passages"][0]
        assert record["provenance"] == original["provenance"]
        assert record["automatic_support_status"] == "unverified_requires_human_review"
        assert record["split"] == "train"
        p = record["selected_passage"]
        section = next(s for s in docs[record["document_id"]]["sections"] if s["section_index"] == p["section_index"])
        assert section["paragraphs"][p["paragraph_index"]] == p["text"]
        assert hashlib.sha256(p["text"].encode()).hexdigest() == p["sha256"]
        assert plan["materialization_status"].startswith("blocked_")
        assert set(plan["arm_exposure_conditions"]) == set(config["training"]["arms"])
        assert all(len(x) == 4 for x in plan["arm_exposure_conditions"].values())
        assert plan["negative_passage_ids"] is None and plan["partial_redaction"] is None
    candidates = rows(OUT / "candidate_expansion.jsonl")
    assert len(candidates) == 122
    assert len({r["document_id"] for r in candidates}) == 122
    assert not {r["document_id"] for r in candidates} & {r["document_id"] for r in train}
    assert all(r["paper_training_rights"] == "unverified" and r["addition_status"].startswith("candidate_only")
               for r in candidates)
    assert max(r["full_tokens"] for r in candidates) <= 512
    ledger = rows(OUT / "license_ledger.jsonl")
    assert len(ledger) == 888
    assert all(r["split"] == "train" and r["underlying_paper_training_rights"] == "unverified" for r in ledger)
    assert Counter(r["study_role"] for r in ledger) == manifest["license_ledger_status_counts"]
    with (SLM / "EVIDENCE_REVIEW_SAMPLE.csv").open(encoding="utf-8", newline="") as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 64
    assert Counter(r["answerability"] for r in review) == {"answerable": 40, "unanswerable": 24}
    assert len({r["question_id"] for r in review}) == 64
    human_fields = ("reviewer_id", "human_support_judgment", "human_evidence_condition",
                    "human_answerability_judgment", "human_contradiction_judgment",
                    "human_medical_domain_judgment", "human_rights_judgment", "review_notes", "reviewed_at")
    assert all(not row[field] for row in review for field in human_fields)
    audit_by_id = {r["question_id"]: r for r in audit}
    for row in review:
        original = audit_by_id[row["question_id"]]
        assert row["selected_passage_text"] == original["selected_passage"]["text"]
        assert json.loads(row["original_annotations_json"]) == original["original_annotations"]
        assert json.loads(row["original_evidence_passages_json"]) == original["original_evidence_passages"]
    assert manifest["status"] == "automated_audit_only_no_human_judgments"
    assert manifest["test_question_or_answer_files_read"] is False
    assert manifest["training_executed"] is False
    return {"status": "validated", "selected": len(train), "review_rows": len(review),
            "candidate_documents": len(candidates), "ledger_documents": len(ledger),
            "human_reviews_completed": 0, "test_answers_read": False}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
