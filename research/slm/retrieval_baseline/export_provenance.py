"""Join frozen predictions to original QASPER evidence and annotation identifiers."""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
OUT = Path(os.environ.get("SLM_RETRIEVAL_BASELINE_OUT", str(HERE / "v1")))
GOLD = SLM / "portable/data/v1/validation_gold.jsonl"
CONDITIONS = ("gold", "tfidf_faiss", "minilm_faiss")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def export():
    output = OUT / "prediction_provenance.jsonl"
    manifest = OUT / "prediction_provenance.manifest.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite versioned provenance output")
    fixed = rows(GOLD)
    result = []
    hashes = {}
    for condition in CONDITIONS:
        source_path = GOLD if condition == "gold" else OUT / f"validation_{condition}.jsonl"
        predictions_path = OUT / f"predictions_{condition}.json"
        source = rows(source_path)
        predictions = json.loads(predictions_path.read_text())
        if predictions["status"] != "complete" or len(source) != 80 or len(predictions["predictions"]) != 80:
            raise RuntimeError(f"Incomplete {condition} condition")
        hashes[condition] = {"input_sha256": digest(source_path),
                             "predictions_sha256": digest(predictions_path)}
        for gold, row, pred in zip(fixed, source, predictions["predictions"]):
            if gold["question_id"] != row["question_id"] or row["question_id"] != pred["question_id"]:
                raise RuntimeError("Question alignment drift")
            result.append({"condition": condition, "question_id": row["question_id"],
                           "document_id": row["document_id"], "source_split": "validation",
                           "qasper_annotation_ids": gold["provenance"]["annotation_ids"],
                           "gold_qasper_evidence_ids": sorted({eid for p in gold["selected_passages"]
                                                                 for eid in p["evidence_ids"]}),
                           "selected_passage_ids": pred["selected_passage_ids"],
                           "selected_passage_sha256": pred["input_passage_sha256"],
                           "selected_qasper_evidence_ids": sorted({eid for p in row["selected_passages"]
                                                                     for eid in p.get("evidence_ids", [])}),
                           "model_weights_sha256": pred["model_weights_sha256"],
                           "experiment_config_sha256": pred["experiment_config_sha256"],
                           "prediction_error": pred["error"]})
    with output.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    meta = {"schema_version": 1, "rows": len(result), "conditions": hashes,
            "gold_source_sha256": digest(GOLD), "output_sha256": digest(output),
            "test_answers_read": False,
            "note": "Retrieved passages carry section/paragraph IDs; their QASPER annotation evidence IDs are not assigned by retrieval, even when the paragraph matches gold. Gold evidence IDs are carried separately."}
    manifest.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(export(), indent=2))
