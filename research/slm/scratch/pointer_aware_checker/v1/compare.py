"""Frozen old checker versus new checker on identical fresh gate candidates."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint  # noqa: E402
from pointer_qa.prepare import sha  # noqa: E402
from evidence_checker.v2.model import SupportCalibrator  # noqa: E402
from evidence_checker.v2.train import provenance as old_provenance  # noqa: E402
from pointer_aware_checker.v1.train import scenario_metrics, class_metrics  # noqa: E402


def compare():
    destination = HERE / "run" / "comparison.json"
    if destination.exists():
        raise FileExistsError("Comparison already frozen")
    run = json.loads((HERE / "run" / "run.json").read_text())
    if run["status"] != "completed":
        raise ValueError("New guarded run incomplete")
    worker = run["worker_result"]
    if sha(HERE / "run" / "gate_predictions.jsonl") != worker["gate_predictions_sha256"]:
        raise ValueError("New gate predictions changed")
    new_rows = [json.loads(x) for x in (HERE / "run" / "gate_predictions.jsonl").read_text().splitlines()]
    old_selection = json.loads((SCRATCH / "evidence_checker" / "v2" / "selection.json").read_text())
    old_root = SCRATCH / "evidence_checker" / "v2" / "run"
    old_checkpoint = old_root / f"checkpoint-step-{old_selection['checkpoint_step']:04d}"
    if sha(old_checkpoint / "model.safetensors") != old_selection["checkpoint_sha256"]:
        raise ValueError("Old checker changed")
    scaler_path = old_root / "scaler.json"
    if sha(scaler_path) != old_selection["scaler_sha256"]:
        raise ValueError("Old scaler changed")
    scaler = json.loads(scaler_path.read_text())
    old = SupportCalibrator()
    optimizer = optim.AdamW(learning_rate=.01, weight_decay=0.)
    load_checkpoint(old_checkpoint, old, optimizer,
                    expected_provenance=old_provenance() | {"scaler_sha256": old_selection["scaler_sha256"]})
    old.eval()
    cached = [json.loads(x) for x in (HERE / "run" / "final_gate_features.jsonl").read_text().splitlines()]
    if [r["id"] for r in cached] != [r["id"] for r in new_rows]:
        raise ValueError("Candidate order mismatch")
    old_rows = []
    for source, new in zip(cached, new_rows):
        feature = mx.array([(source["features"][i]-scaler["mean"][i])/scaler["std"][i]
                            for i in range(8)], dtype=mx.float32)
        z = old(feature)
        mx.eval(z)
        old_rows.append({key: new[key] for key in new if key != "probability"} |
                        {"probability": float(mx.sigmoid(z).item())})
    with (HERE / "run" / "old_checker_same_gate_predictions.jsonl").open("x") as out:
        for r in old_rows:
            out.write(json.dumps(r, sort_keys=True) + "\n")
    old_metric = scenario_metrics(old_rows, old_selection["threshold"])
    new_metric = scenario_metrics(new_rows, worker["threshold"])
    if new_metric != worker["final_gate"]:
        raise ValueError("New gate metric mismatch")
    groups = defaultdict(list)
    for row in new_rows:
        if row["pointer_selected"]:
            groups[row["condition"]].append(row)
    result = {"same_candidate_ids": True, "scenario_count": sum(len(v) for v in groups.values()),
        "new_threshold": worker["threshold"], "old_frozen_threshold": old_selection["threshold"],
        "pointer_only_original_exact_count": sum(x["label_supported"] for x in groups["original"]),
        "pointer_only_original_count": len(groups["original"]),
        "old_checker": old_metric, "new_checker": new_metric,
        "old_checker_all_candidates": class_metrics(old_rows, old_selection["threshold"]),
        "new_checker_all_candidates": class_metrics(new_rows, worker["threshold"]),
        "new_prediction_sha256": sha(HERE / "run" / "gate_predictions.jsonl"),
        "old_prediction_sha256": sha(HERE / "run" / "old_checker_same_gate_predictions.jsonl"),
        "old_checkpoint_sha256": old_selection["checkpoint_sha256"],
        "new_checkpoint_sha256": worker["checkpoint_sha256"], "sealed_test_opened": False}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
