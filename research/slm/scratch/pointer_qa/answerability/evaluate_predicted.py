"""Validation-only checker evaluation on spans proposed by the frozen pointer."""
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402
from pointer_qa.prepare import sha  # noqa: E402
from pointer_qa.v2.model import predict  # noqa: E402
from pointer_qa.answerability.model import SufficiencyHead, frozen_features  # noqa: E402
from pointer_qa.answerability.train import config, load_plan, metrics, provenance, rows, selected_pointer  # noqa: E402


def evaluate():
    p, c = load_plan(), config()
    run = json.loads((HERE / "v1" / "run" / "run.json").read_text())
    if run["status"] != "completed":
        raise RuntimeError("Checker did not complete its guarded run")
    result = run["worker_result"]
    selected = result["best_step"]
    output = HERE / "v1" / "predicted_validation"
    if output.exists():
        raise FileExistsError("Predicted-span validation already frozen")
    pre = preflight("smoke")
    output.mkdir()
    (output / "preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    if not pre["ready"]:
        stopped = {"status": "preflight_refused", "reasons": pre["reasons"]}
        (output / "status.json").write_text(json.dumps(stopped, indent=2) + "\n")
        return stopped
    pointer, tok = selected_pointer()
    head = SufficiencyHead(c["training"]["hidden_dim"])
    optimizer = optim.AdamW(learning_rate=c["training"]["learning_rate"],
                            weight_decay=c["training"]["weight_decay"])
    checkpoint = HERE / "v1" / "run" / f"checkpoint-step-{selected:04d}"
    load_checkpoint(checkpoint, head, optimizer,
                    expected_provenance=provenance() | {"scaler_sha256": sha(HERE / "v1" / "run" / "scaler.json")})
    head.eval()
    scale = json.loads((HERE / "v1" / "run" / "scaler.json").read_text())
    mean = mx.array(scale["mean"], dtype=mx.float32)
    std = mx.array(scale["std"], dtype=mx.float32)
    threshold = result["selected_validation"]["threshold"]
    unique = {}
    for row in rows("validation"):
        if row["scenario_id"] in unique:
            old = unique[row["scenario_id"]]
            if (old["passage"] != row["passage"] or old["question"] != row["question"] or
                old["supported_candidate_indices"] != row["supported_candidate_indices"]):
                raise ValueError("Conflicting scenario records")
        else:
            unique[row["scenario_id"]] = row
    records = []
    started = time.monotonic()
    mx.metal.reset_peak_memory()
    with (output / "predictions.jsonl").open("x") as stream:
        for i, row in enumerate(unique.values()):
            if i % 16 == 0:
                reasons = guard_reasons(snapshot(os.getpid()), pre["resources"], pre["safety"])
                if reasons:
                    stopped = {"status": "stopped_by_guard", "at_index": i, "reasons": reasons}
                    (output / "status.json").write_text(json.dumps(stopped, indent=2) + "\n")
                    return stopped
            tick = time.monotonic()
            pairs, candidates = pair_inputs(tok, row["passage"], row["question"])
            found = predict(pointer, pairs, candidates)
            index = found["candidate_index"]
            feature, lengths = frozen_features(pointer, tok, row["passage"], row["question"],
                                               index, c["training"]["maximum_full_input_tokens"])
            z = head((mx.array(feature, dtype=mx.float32) - mean) / std)
            mx.eval(z)
            probability = float(mx.sigmoid(z).item())
            record = {"scenario_id": row["scenario_id"], "source_question_id": row["source_question_id"],
                      "document_id": row["document_id"], "category": row["category"],
                      "question": row["question"], "predicted_candidate_index": index,
                      "predicted_span": found["evidence"], "predicted_answer": found["prediction"],
                      "supported_candidate_indices": row["supported_candidate_indices"],
                      "label_supported": index in row["supported_candidate_indices"],
                      "probability": probability, "accepted": probability >= threshold,
                      "correct_classification": (probability >= threshold) == (index in row["supported_candidate_indices"]),
                      "latency_ms": (time.monotonic() - tick) * 1000,
                      "full_input_tokens": lengths["full_input_tokens"]}
            stream.write(json.dumps(record) + "\n")
            records.append(record)
    calset = set(p["calibration_document_ids"])
    assessset = set(p["assessment_document_ids"])
    cal = [r for r in records if r["document_id"] in calset]
    assessment = [r for r in records if r["document_id"] in assessset]
    summary = {"status": "completed", "selection": "frozen_pointer_argmax", "threshold": threshold,
               "calibration": metrics(cal, threshold), "assessment": metrics(assessment, threshold),
               "all_validation": metrics(records, threshold),
               "mean_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
               "peak_mlx_bytes": mx.metal.get_peak_memory(), "wall_seconds": time.monotonic() - started,
               "pointer_checkpoint_sha256": result["pointer_checkpoint_sha256"],
               "checker_checkpoint_sha256": sha(checkpoint / "model.safetensors"),
               "predictions_sha256": sha(output / "predictions.jsonl"), "test_file_opened": False}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
