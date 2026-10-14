"""Read-only v3 pilot analysis; no test data path."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checkpoint import digest
from v3_data import V3, load_verified, load_plan
from v3_worker import restore_exact

SCRATCH = Path(__file__).resolve().parent.parent
DEFAULT_RUN = SCRATCH / "runs/v4-cpython-v3-pilot"


def analyze(run_dir=DEFAULT_RUN):
    run_dir = Path(run_dir).resolve()
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    plan = load_plan(config, train_windows, val_windows)
    run = json.loads((run_dir / "run.json").read_text())
    if run["status"] != "completed" or run["completed_steps"] != 100:
        raise ValueError("Pilot not completed at preregistered budget")
    if run["v3_manifest_sha256"] != digest(V3 / "manifest.json") or run["pilot_plan_sha256"] != digest(V3 / "pilot_plan.v1.json"):
        raise ValueError("Pilot provenance changed")
    curve = [json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()]
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    if [row["step"] for row in curve] != config["evaluation_steps"] or len(steps) != 100:
        raise ValueError("Curve or step log incomplete")
    if [(s["document_id"], s["window_offset"]) for s in steps] != [(w["document_id"], w["offset"]) for w in plan["training_sample_order"]]:
        raise ValueError("Observed sample order differs from frozen plan")
    worker = run["worker_result"]
    best = min(curve[1:], key=lambda row: row["validation"]["document_balanced_loss"])
    if worker["best"]["step"] != best["step"]:
        raise ValueError("Checkpoint selection differs from preregistration")
    model_config = {**json.loads((SCRATCH / "config.v1.json").read_text())["model"], **config["architecture"]}
    restored = {}
    for step in sorted({best["step"], 100}):
        path = run_dir / f"checkpoints/step-{step:04d}"
        _, result = restore_exact(path, model_config, tokenizer, config, worker["provenance"],
                                  plan["fixed_train"], plan["fixed_validation"], next(row for row in curve if row["step"] == step))
        restored[str(step)] = result
    scalar_keys = ("gradient_l2_norm", "gradient_max_absolute", "parameter_l2_norm", "parameter_max_absolute",
                   "minibatch_loss_before_update", "duration_seconds", "mlx_peak_bytes")
    summaries = {}
    for key in scalar_keys:
        values = [row[key] for row in steps]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite step metric: {key}")
        summaries[key] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
    fixed_train_ids = {(w["document_id"], w["offset"]) for w in plan["fixed_train"]}
    sample_ids = {(w["document_id"], w["offset"]) for w in plan["training_sample_order"]}
    curve_rows = [{"step": row["step"], "fixed_train_loss": row["train"]["loss"],
                   "fixed_train_perplexity": row["train"]["perplexity"],
                   "validation_window_loss": row["validation"]["loss"],
                   "validation_document_balanced_loss": row["validation"]["document_balanced_loss"],
                   "validation_document_balanced_perplexity": row["validation"]["document_balanced_perplexity"]}
                  for row in curve]
    rises = {"fixed_train": [curve_rows[i]["step"] for i in range(1, len(curve_rows)) if curve_rows[i]["fixed_train_loss"] > curve_rows[i - 1]["fixed_train_loss"]],
             "document_balanced_validation": [curve_rows[i]["step"] for i in range(1, len(curve_rows)) if curve_rows[i]["validation_document_balanced_loss"] > curve_rows[i - 1]["validation_document_balanced_loss"]]}
    return {"status": "verified", "manifest_sha256": digest(V3 / "manifest.json"),
            "pilot_config_sha256": digest(V3 / "pilot_config.v1.json"),
            "pilot_plan_sha256": digest(V3 / "pilot_plan.v1.json"),
            "curve": curve_rows, "scheduled_loss_rises": rises, "best_validation_step": best["step"],
            "sampled_unique_windows": len(sample_ids), "fixed_train_windows_seen_during_training": len(fixed_train_ids & sample_ids),
            "sampled_training_documents": len({w["document_id"] for w in plan["training_sample_order"]}),
            "sampled_training_groups": len({w["group_id"] for w in plan["training_sample_order"]}),
            "training_target_tokens_processed": sum(s["tokens"] for s in steps),
            "training_corpus_token_fraction_exposed": worker["training_corpus_token_fraction_exposed"],
            "step_metric_summaries": summaries, "restored_checkpoints": restored,
            "test_split_evaluated": False,
            "interpretation_limit": "Only 100 updates and 5.14% of unique training tokens were exposed; falling next-token loss does not imply coherent generation or technical QA ability."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RUN / "analysis.json")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_RUN / "curve.csv")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        raise FileExistsError("Analysis artifacts cannot be overwritten")
    result = analyze(args.run_dir)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["curve"][0]))
        writer.writeheader()
        writer.writerows(result["curve"])
    print(json.dumps({"status": result["status"], "best_step": result["best_validation_step"],
                      "scheduled_loss_rises": result["scheduled_loss_rises"],
                      "restored_steps": list(result["restored_checkpoints"])}))
