"""Offline, immutable retrieval comparison on the fixed 80 validation rows."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
sys.path.insert(0, str(SLM / "portable"))
from retrieve import (MINILM_REV, minilm_model, minilm_rank, pack, paragraphs,
                      tfidf_rank)
from transformers import AutoTokenizer

OUT = Path(os.environ.get("SLM_RETRIEVAL_BASELINE_OUT", str(HERE / "v1")))
SOURCE = SLM / "portable/data/v1/validation_gold.jsonl"
DOCS = SLM / "curated/validation_documents.jsonl"
EXPERIMENT = SLM / "portable/v1/experiment.json"
RETRIEVAL = SLM / "portable/v1/retrieval.json"
TOKENIZER = SLM / ".cache/qwen3-0.6b-4bit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def retrieval_metrics(rows):
    answerable = [r for r in rows if r["answerability"] == "answerable"]
    ranks = []
    for row in answerable:
        gold = row["reference_citation"]
        found = next((i + 1 for i, p in enumerate(row["retrieval_candidates"])
                      if p["passage_id"] == gold), None)
        ranks.append(found)
    latencies = sorted(r["retrieval_latency_seconds"] for r in rows)
    return {
        "questions": len(rows), "answerable_gold_questions": len(answerable),
        "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks),
        "mrr_at_3": sum(1 / rank for rank in ranks if rank is not None and rank <= 3) / len(ranks),
        "packed_evidence_coverage": sum(r["gold_passage_retrieved"] is True for r in answerable) / len(answerable),
        "packed_evidence_count": sum(r["gold_passage_retrieved"] is True for r in answerable),
        "candidates_skipped_for_budget": sum(not p["included"] for r in rows for p in r["retrieval_candidates"]),
        "questions_with_no_packed_passage": sum(not r["selected_passages"] for r in rows),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "mean_rank_latency_seconds": statistics.mean(r["ranking_latency_seconds"] for r in rows),
    }


def run(condition):
    if condition not in ("tfidf_faiss", "minilm_faiss"):
        raise ValueError(condition)
    output = OUT / f"validation_{condition}.jsonl"
    manifest = OUT / f"validation_{condition}.manifest.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("Versioned retrieval output exists; choose a new version instead of overwriting")
    config = json.loads((OUT / "config.json").read_text())
    if digest(SOURCE) != config["validation_source_sha256"]:
        raise RuntimeError("Fixed validation source checksum changed")
    retrieval = json.loads(RETRIEVAL.read_text())
    experiment = json.loads(EXPERIMENT.read_text())
    assert retrieval["conditions"][condition]["top_k"] == config["top_k"]
    assert experiment["maximum_sequence_tokens"] == config["maximum_sequence_tokens"]
    assert experiment["maximum_generation_tokens"] == config["maximum_generation_tokens"]
    source = list(read_rows(SOURCE))
    if len(source) != 80 or sum(r["answerability"] == "answerable" for r in source) != 40:
        raise RuntimeError("Expected fixed 40/40 validation sample")
    docs = {d["document_id"]: d for d in read_rows(DOCS)}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    start_load = time.perf_counter()
    model = minilm_model() if condition == "minilm_faiss" else None
    load_seconds = time.perf_counter() - start_load
    if model is not None:
        import numpy as np
        probe = model.encode(["evidence retrieval probe"], convert_to_numpy=True)
        if probe.shape != (1, 384) or not np.isfinite(probe).all() or not np.linalg.norm(probe[0]) > 0:
            raise RuntimeError("MiniLM embedding inference failed")
    rows_out = []
    for row in source:
        doc = docs[row["document_id"]]
        items = paragraphs(doc)
        if not items:
            raise RuntimeError(f"No source paragraphs: {row['document_id']}")
        started = time.perf_counter()
        ranking = (minilm_rank(row["question"], items, config["top_k"], model)
                   if model is not None else tfidf_rank(row["question"], items, config["top_k"]))
        ranked_at = time.perf_counter()
        result = pack({**row, "title": doc["title"]}, items, ranking, tokenizer, experiment)
        result["condition"] = condition
        result["source_paragraph_count"] = len(items)
        result["ranking_latency_seconds"] = ranked_at - started
        result["retrieval_latency_seconds"] = time.perf_counter() - started
        rows_out.append(result)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "schema_version": 1, "condition": condition, "rows": len(rows_out),
        "config_sha256": digest(OUT / "config.json"),
        "retrieval_config_sha256": digest(RETRIEVAL),
        "experiment_config_sha256": digest(EXPERIMENT),
        "source_sha256": digest(SOURCE), "documents_sha256": digest(DOCS),
        "tokenizer_config_sha256": digest(TOKENIZER / "tokenizer_config.json"),
        "minilm_revision": MINILM_REV if model is not None else None,
        "minilm_weights_sha256": None,
        "python": platform.python_version(), "platform": platform.platform(),
        "software_versions": {name: importlib.metadata.version(name) for name in
                              ("torch", "sentence-transformers", "transformers", "faiss-cpu", "scikit-learn", "numpy")},
        "model_load_seconds": load_seconds, "metrics": retrieval_metrics(rows_out),
        "output_sha256": digest(output), "test_answers_read": False,
    }
    if model is not None:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download("sentence-transformers/all-MiniLM-L6-v2", revision=MINILM_REV,
                                          local_files_only=True))
        metadata["minilm_weights_sha256"] = digest(snapshot / "model.safetensors")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("tfidf_faiss", "minilm_faiss"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.condition), indent=2))
