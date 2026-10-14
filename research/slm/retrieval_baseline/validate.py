"""Validate frozen retrieval/prediction artifacts and write a checksum manifest."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
sys.path.insert(0, str(HERE))
from response_format import parse_response
from run_retrieval import retrieval_metrics

OUT = Path(os.environ.get("SLM_RETRIEVAL_BASELINE_OUT", str(HERE / "v1")))
GOLD = SLM / "portable/data/v1/validation_gold.jsonl"
DOCS = SLM / "curated/validation_documents.jsonl"
CONDITIONS = ("gold", "tfidf_faiss", "minilm_faiss")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate():
    fixed = rows(GOLD)
    assert len(fixed) == 80
    ids = [r["question_id"] for r in fixed]
    assert len(set(ids)) == 80
    assert len({r["document_id"] for r in fixed}) == 80
    assert {r["split"] for r in fixed} == {"validation"}
    docs = {d["document_id"]: d for d in rows(DOCS)}
    paths = [OUT / "config.json", OUT / "training_audit.json"]
    retrieval_paragraph_counts = {}
    for condition in CONDITIONS:
        source_path = GOLD if condition == "gold" else OUT / f"validation_{condition}.jsonl"
        source = rows(source_path)
        assert [r["question_id"] for r in source] == ids
        for original, row in zip(fixed, source):
            assert row["question"] == original["question"]
            assert row["document_id"] == original["document_id"]
            assert row["answerability"] == original["answerability"]
            assert row["prompt_tokens"] + 64 <= 512
            for passage in row["selected_passages"]:
                section = next(s for s in docs[row["document_id"]]["sections"]
                               if s["section_index"] == passage["section_index"])
                assert section["paragraphs"][passage["paragraph_index"]] == passage["text"]
                assert hashlib.sha256(passage["text"].encode()).hexdigest() == passage["sha256"]
            if condition != "gold":
                assert len(row["retrieval_candidates"]) == 3
                candidate_ids = [p["passage_id"] for p in row["retrieval_candidates"]]
                assert len(set(candidate_ids)) == 3
                source_ids = {f"section:{s['section_index']}:paragraph:{i}"
                              for s in docs[row["document_id"]]["sections"]
                              for i, text in enumerate(s["paragraphs"]) if text and text.strip()}
                assert set(candidate_ids).issubset(source_ids)
                assert row["source_paragraph_count"] == len(source_ids)
                scores = [p["score"] for p in row["retrieval_candidates"]]
                assert scores == sorted(scores, reverse=True)
                assert {p["passage_id"] for p in row["selected_passages"]} == {
                    p["passage_id"] for p in row["retrieval_candidates"] if p["included"]}
                if row["answerability"] == "answerable":
                    assert row["gold_passage_retrieved"] == (row["reference_citation"] in
                                                                 {p["passage_id"] for p in row["selected_passages"]})
        if condition != "gold":
            retrieval_paragraph_counts[condition] = [r["source_paragraph_count"] for r in source]
            manifest_path = OUT / f"validation_{condition}.manifest.json"
            meta = json.loads(manifest_path.read_text())
            assert digest(source_path) == meta["output_sha256"]
            assert digest(GOLD) == meta["source_sha256"]
            assert retrieval_metrics(source)["packed_evidence_count"] == meta["metrics"]["packed_evidence_count"]
            paths.extend((source_path, manifest_path))
        prediction_path = OUT / f"predictions_{condition}.json"
        audit_path = OUT / f"predictions_{condition}.manual_audit.jsonl"
        artifact = json.loads(prediction_path.read_text())
        assert artifact["status"] == "complete"
        assert artifact["preflight"]["ready"]
        assert artifact["preflight"]["validation_input_sha256"] == digest(source_path)
        assert artifact["inference"]["adapter"] is None
        predictions = artifact["predictions"]
        assert len(predictions) == 80
        assert [r["question_id"] for r in predictions] == ids
        for row, pred in zip(source, predictions):
            assert pred["document_id"] == row["document_id"]
            assert pred["answerability"] == row["answerability"]
            assert pred["prompt_tokens"] == row["prompt_tokens"]
            assert set(pred["selected_passage_ids"]) == {p["passage_id"] for p in row["selected_passages"]}
            assert pred["input_passage_sha256"] == {p["passage_id"]: p["sha256"] for p in row["selected_passages"]}
            assert pred["input_sha256"] == digest(source_path)
            assert pred["model_weights_sha256"] == artifact["preflight"]["model_weights_sha256"]
            parsed = parse_response(pred["raw_response"], pred["finish_reason"])
            assert pred["normalized_response"] == parsed["normalized_response"]
            assert pred["format_status"] == parsed["format_status"]
            assert pred["strict_abstention"] == parsed["exact_unanswerable"]
            assert pred["generation_truncated"] == parsed["generation_truncated"]
            assert pred["error"] is None
        assert len(rows(audit_path)) == 10
        paths.extend((prediction_path, audit_path))
    audit = json.loads((OUT / "training_audit.json").read_text())
    assert retrieval_paragraph_counts["tfidf_faiss"] == retrieval_paragraph_counts["minilm_faiss"]
    assert audit["selected_examples"] == 352 and audit["selected_unique_documents"] == 352
    assert audit["selected_ids_all_eligible"] and not audit["test_files_read"]
    smoke = OUT / "smoke_gold.json"
    assert json.loads(smoke.read_text())["status"] == "smoke_complete"
    paths.extend((smoke, OUT / "smoke_gold.manual_audit.jsonl"))
    spot = rows(OUT / "support_spotcheck.jsonl")
    assert len(spot) == 18
    for condition in CONDITIONS:
        chosen = [r for r in spot if r["condition"] == condition]
        assert len(chosen) == 6
        first_three_each = [r["question_id"] for label in ("answerable", "unanswerable")
                            for r in [x for x in fixed if x["answerability"] == label][:3]]
        assert [r["question_id"] for r in chosen] == first_three_each
    spot_meta = json.loads((OUT / "support_spotcheck.manifest.json").read_text())
    assert spot_meta["output_sha256"] == digest(OUT / "support_spotcheck.jsonl")
    provenance = rows(OUT / "prediction_provenance.jsonl")
    provenance_meta = json.loads((OUT / "prediction_provenance.manifest.json").read_text())
    assert len(provenance) == provenance_meta["rows"] == 240
    assert provenance_meta["output_sha256"] == digest(OUT / "prediction_provenance.jsonl")
    for index, condition in enumerate(CONDITIONS):
        block = provenance[index * 80:(index + 1) * 80]
        assert [r["question_id"] for r in block] == ids
        assert {r["condition"] for r in block} == {condition}
        for gold_row, record in zip(fixed, block):
            assert record["qasper_annotation_ids"] == gold_row["provenance"]["annotation_ids"]
            assert record["gold_qasper_evidence_ids"] == sorted({eid for p in gold_row["selected_passages"]
                                                                   for eid in p["evidence_ids"]})
            assert record["source_split"] == "validation"
    return {"schema_version": 1, "status": "validated", "fixed_validation_questions": 80,
            "answerable": 40, "unanswerable": 40, "test_answers_read": False,
            "files_sha256": {str(path.relative_to(SLM)): digest(path) for path in paths}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUT / "artifact_manifest.json")
    args = parser.parse_args()
    result = validate()
    if args.manifest.exists():
        existing = json.loads(args.manifest.read_text())
        if existing != result:
            raise SystemExit("Artifact manifest exists and differs; do not overwrite immutable outputs")
    else:
        args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
