"""Safety-gated, model-only QA evaluation; sealed test requires a passed gate."""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.model import CandidateReader, predict  # noqa: E402
from working_qa.prepare import V1, sha, tokenize, verify  # noqa: E402
from working_qa.train import config, provenance, rows, tokenizer  # noqa: E402


def token_f1(prediction, expected):
    a, b = re.findall(r"\w+", prediction.lower()), re.findall(r"\w+", expected.lower())
    if not a or not b:
        return float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    return 2 * overlap / (len(a) + len(b))


def metrics(records):
    positives = [r for r in records if r["answerable"]]
    negatives = [r for r in records if not r["answerable"]]
    supported = [r for r in records if r["prediction"] != "UNANSWERABLE"]
    by_tier = {}
    for tier in sorted({r["tier"] for r in records}):
        subset = [r for r in records if r["tier"] == tier]
        answerable = [r for r in subset if r["answerable"]]
        by_tier[tier] = {"count": len(subset),
                         "answerable_exact_match": sum(r["exact"] for r in answerable) / len(answerable),
                         "unanswerable_accuracy": sum(r["exact"] for r in subset if not r["answerable"]) /
                                                   (len(subset) - len(answerable))}
    return {"count": len(records), "answerable_count": len(positives),
            "unanswerable_count": len(negatives),
            "answerable_exact_match": sum(r["exact"] for r in positives) / len(positives),
            "answerable_token_f1": sum(r["token_f1"] for r in positives) / len(positives),
            "unanswerable_accuracy": sum(r["exact"] for r in negatives) / len(negatives),
            "overall_exact_match": sum(r["exact"] for r in records) / len(records),
            "span_support_rate": sum(r["span_supported"] for r in supported) / len(supported) if supported else None,
            "non_abstaining_count": len(supported),
            "confusion": {"answerable_correct_span": sum(r["exact"] for r in positives),
                          "answerable_abstained": sum(r["prediction"] == "UNANSWERABLE" for r in positives),
                          "answerable_wrong_span": sum(not r["exact"] and r["prediction"] != "UNANSWERABLE" for r in positives),
                          "unanswerable_correct_abstention": sum(r["exact"] for r in negatives),
                          "unanswerable_false_span": sum(not r["exact"] for r in negatives)},
            "tiers": by_tier,
            "mean_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
            "median_latency_ms": sorted(r["latency_ms"] for r in records)[len(records) // 2]}


def evaluate(split):
    verify()
    c = config()
    run = json.loads((V1 / "runs" / "main" / "run.json").read_text())
    if run["status"] != "completed":
        raise RuntimeError("Main training run did not complete")
    best_step = run["worker_result"]["best_step"]
    checkpoint = V1 / "runs" / "main" / f"checkpoint-step-{best_step:04d}"
    output = V1 / "analysis" / ("validation_best" if split == "validation" else "test_final")
    if output.exists():
        raise FileExistsError("Evaluation already exists; results are immutable")
    if split == "test":
        selection = V1 / "analysis" / "validation_best" / "frozen_selection.json"
        if not selection.exists():
            raise PermissionError("Sealed test requires a passed, frozen validation gate")
        frozen = json.loads(selection.read_text())
        if frozen["checkpoint_sha256"] != sha(checkpoint / "model.safetensors") or not frozen["passed"]:
            raise PermissionError("Frozen checkpoint/gate mismatch")
    before = preflight("smoke")
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    if not before["ready"]:
        (output / "status.json").write_text(json.dumps({"status": "preflight_refused", "reasons": before["reasons"]}, indent=2) + "\n")
        return {"status": "preflight_refused", "reasons": before["reasons"]}
    tok = tokenizer()
    model = CandidateReader(c["model"], tok.vocab_size)
    optimizer = optim.AdamW(learning_rate=c["training"]["learning_rate"],
                            weight_decay=c["training"]["weight_decay"])
    load_checkpoint(checkpoint, model, optimizer, expected_provenance=provenance())
    model.eval()
    mx.metal.reset_peak_memory()
    if split == "validation":
        examples = rows("validation")
    else:
        examples = [json.loads(line) for line in (V1 / "test.sealed.jsonl").read_text().splitlines()]
    results = []
    start = time.monotonic()
    with (output / "predictions.jsonl").open("x") as stream:
        for index, row in enumerate(examples):
            if index % 16 == 0:
                observed = snapshot(os.getpid())
                reasons = guard_reasons(observed, before["resources"], before["safety"])
                if reasons:
                    (output / "status.json").write_text(json.dumps({"status": "stopped_by_guard", "at_index": index,
                                                                      "reasons": reasons}, indent=2) + "\n")
                    return {"status": "stopped_by_guard", "at_index": index, "reasons": reasons}
            enc = tokenize(tok, row)
            tick = time.monotonic()
            found = predict(model, enc, row["candidates"])
            latency_ms = 1000 * (time.monotonic() - tick)
            evidence = found["evidence"]
            span_supported = bool(evidence and row["passage"][evidence["start"]:evidence["end"]] == found["prediction"])
            record = {"id": row["id"], "split": split, "tier": row["tier"],
                      "answerable": row["answerable"], "expected_answer": row["expected_answer"],
                      "prediction": found["prediction"], "candidate_index": found["candidate_index"],
                      "confidence_uncalibrated": found["confidence_uncalibrated"],
                      "evidence": evidence, "span_supported": span_supported,
                      "exact": found["prediction"] == row["expected_answer"],
                      "token_f1": token_f1(found["prediction"], row["expected_answer"]),
                      "latency_ms": latency_ms}
            stream.write(json.dumps(record) + "\n")
            results.append(record)
    summary = metrics(results)
    gate = c["validation_gate"]
    harder = [summary["tiers"][tier]["answerable_exact_match"] for tier in ("multi_entity", "paraphrase_conflict")]
    passed = (split == "validation" and summary["count"] >= gate["minimum_examples"] and
              summary["answerable_exact_match"] >= gate["answerable_exact_match"] and
              summary["unanswerable_accuracy"] >= gate["unanswerable_accuracy"] and
              min(harder) >= gate["minimum_harder_subset_exact_match"] and
              summary["span_support_rate"] == gate["required_span_support_rate"])
    summary.update({"status": "completed", "validation_gate_passed": passed,
                    "split": split, "checkpoint_step": best_step,
                    "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
                    "model_source_sha256": sha(HERE / "model.py"),
                    "config_sha256": sha(V1 / "config.json"),
                    "dataset_manifest_sha256": sha(V1 / "manifest.json"),
                    "tokenizer_sha256": sha(V1 / "tokenizer.json"),
                    "prediction_sha256": sha(output / "predictions.jsonl"),
                    "peak_mlx_bytes": mx.metal.get_peak_memory(),
                    "wall_seconds": time.monotonic() - start})
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if passed:
        frozen = {"passed": True, "checkpoint_step": best_step,
                  "checkpoint_sha256": summary["checkpoint_sha256"],
                  "validation_prediction_sha256": summary["prediction_sha256"],
                  "inference_protocol": "v1 bracketed candidate values; max softmax score including NULL; exact source span copy"}
        (output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.split), indent=2))
