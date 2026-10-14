"""Deterministic candidate cohort from independent reviews and rights decisions.

Produces a conservative *candidate* only; it cannot clear unreviewed variants or
turn this dataset into a training-ready release.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from acl_metadata import SELECTED, load_selected
from review_core import SAMPLE, DECISIONS as REVIEW_DECISIONS, latest_reviews, load_sample, review_events, sha256
from rights_core import LEDGER, DECISIONS as RIGHTS_DECISIONS, latest_rights, read_jsonl

HERE = Path(__file__).resolve().parent
FROZEN_MANIFEST = HERE.parent / "training_data_v2/v1/manifest.json"
FROZEN_VALIDATION_IDS = HERE.parent / "training_data_v2/v1/validation_ids.jsonl"


def construct(*, selected_path=SELECTED, sample_path=SAMPLE, review_path=REVIEW_DECISIONS,
              ledger_path=LEDGER, rights_path=RIGHTS_DECISIONS):
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if sha256(selected_path) != manifest["file_sha256"]["selected_audit.jsonl"] or sha256(sample_path) != manifest["file_sha256"]["EVIDENCE_REVIEW_SAMPLE.csv"]:
        raise ValueError("Frozen training audit or review sample changed")
    selected = load_selected(selected_path)
    sample_ids = {r["question_id"] for r in load_sample(sample_path)}
    validation_ids = {json.loads(line)["document_id"] for line in FROZEN_VALIDATION_IDS.read_text().splitlines() if line}
    if {r["document_id"] for r in selected} & validation_ids:
        raise ValueError("Train/validation document overlap")
    ledger = read_jsonl(ledger_path)
    provenance_manifest = json.loads((HERE / "v1/provenance_manifest.json").read_text())
    if sha256(ledger_path) != provenance_manifest["provenance_ledger_sha256"]:
        raise ValueError("Pinned ACL provenance ledger changed")
    by_doc = {r["document_id"]: r for r in ledger}
    if len(ledger) != 352 or len(by_doc) != 352 or set(by_doc) != {r["document_id"] for r in selected}:
        raise ValueError("Provenance ledger must align with all 352 frozen train documents")
    reviews = latest_reviews(review_events(review_path), sample_ids, sha256(sample_path))
    by_question = defaultdict(list)
    for (qid, _), event in reviews.items():
        by_question[qid].append(event)
    rights = latest_rights(read_jsonl(rights_path), sha256(ledger_path), set(by_doc))
    by_rights = defaultdict(list)
    for (doc_id, _), event in rights.items():
        by_rights[doc_id].append(event)
    candidates = []
    decisions = []
    for row in selected:
        qid, doc_id = row["question_id"], row["document_id"]
        reasons = []
        if qid not in sample_ids:
            reasons.append("not_in_64_question_review_sample")
        events = by_question[qid]
        if len(events) < 2:
            reasons.append("fewer_than_two_independent_evidence_reviews")
        else:
            verdicts = {(e["judgment"], e["abstention_appropriate"], e["document_answerability_consistent"], e["full_document_reviewed"], e["medical_domain"]) for e in events}
            if len(verdicts) != 1:
                reasons.append("evidence_reviewer_disagreement")
            if any(e["medical_domain"] != "No" for e in events):
                reasons.append("medical_domain_not_cleared")
            if any(e["document_answerability_consistent"] != "Yes" for e in events):
                reasons.append("document_answerability_not_confirmed")
            if row["answerability"] == "answerable":
                if any(e["judgment"] != "Supported" or e["abstention_appropriate"] != "No" for e in events):
                    reasons.append("answerable_reference_or_abstention_not_approved")
            elif any(e["judgment"] != "Unsupported" or e["abstention_appropriate"] != "Yes" or e["full_document_reviewed"] != "Yes" for e in events):
                reasons.append("unanswerable_full_document_review_not_approved")
        if "possible_annotation_contradiction" in row["flags"]:
            reasons.append("original_annotation_contradiction_requires_adjudication")
        source = by_doc[doc_id]
        if source["acl_publication_license_class"] == "restrictive_acl_publication":
            reasons.append("restrictive_acl_publication_license")
        elif source["acl_publication_license_class"] != "verified_permissive_acl_publication":
            reasons.append("acl_publication_license_unresolved")
        rights_events = by_rights[doc_id]
        if not rights_events or not any(e["decision"] == "permissive_verified" and e["source_version_matches"] == "Yes" and e["third_party_material_cleared"] == "Yes" and e["training_and_redistribution_allowed"] == "Yes" for e in rights_events):
            reasons.append("individual_source_rights_not_cleared")
        if any(e["decision"] in ("restricted", "unresolved") for e in rights_events):
            reasons.append("rights_reviewer_disagreement_or_restriction")
        decision = {"question_id": qid, "document_id": doc_id, "answerability": row["answerability"],
                    "candidate_included": not reasons, "reasons": reasons,
                    "evidence_reviewer_ids": sorted(e["reviewer_id"] for e in events),
                    "rights_reviewer_ids": sorted(e["reviewer_id"] for e in rights_events),
                    "source_passage_id": row["selected_passage"]["passage_id"],
                    "source_passage_sha256": row["selected_passage"]["sha256"]}
        decisions.append(decision)
        if not reasons:
            candidates.append({"question_id": qid, "document_id": doc_id,
                               "answerability": row["answerability"],
                               "source_passage_id": row["selected_passage"]["passage_id"],
                               "source_passage_sha256": row["selected_passage"]["sha256"],
                               "acl_id": source["acl_id"], "acl_source_url": source["source_url"],
                               "status": "reviewed_candidate_not_variant_training_ready"})
    return candidates, decisions


def write_snapshot(output_dir, *, selected_path=SELECTED, sample_path=SAMPLE,
                   review_path=REVIEW_DECISIONS, ledger_path=LEDGER, rights_path=RIGHTS_DECISIONS):
    output_dir = Path(output_dir)
    names = ("candidate_cohort.jsonl", "cohort_decisions.jsonl", "manifest.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Cohort snapshot already exists; create a new version")
    candidates, decisions = construct(selected_path=selected_path, sample_path=sample_path,
                                      review_path=review_path, ledger_path=ledger_path, rights_path=rights_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidate_cohort.jsonl", candidates), ("cohort_decisions.jsonl", decisions)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    sources = {"selected_audit.jsonl": sha256(selected_path), "EVIDENCE_REVIEW_SAMPLE.csv": sha256(sample_path),
               "provenance_ledger.jsonl": sha256(ledger_path)}
    for name, path in (("reviews.jsonl", review_path), ("rights_decisions.jsonl", rights_path)):
        sources[name] = sha256(path) if Path(path).exists() else hashlib.sha256(b"").hexdigest()
    manifest = {"schema_version": 1, "status": "candidate_only_not_training_ready", "source_sha256": sources,
                "candidate_count": len(candidates), "excluded_or_unresolved_count": len(decisions)-len(candidates),
                "reason_counts": dict(Counter(reason for row in decisions for reason in row["reasons"])),
                "tool_sha256": {"cohort.py": sha256(HERE / "cohort.py"),
                                "review_core.py": sha256(HERE / "review_core.py"),
                                "rights_core.py": sha256(HERE / "rights_core.py")},
                "file_sha256": {name: sha256(output_dir/name) for name in names[:-1]},
                "test_answers_read": False, "training_executed": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
