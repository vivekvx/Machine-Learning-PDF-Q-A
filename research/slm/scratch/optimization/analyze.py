"""Summarize the paired fixed-window curves and verify saved tensor restoration."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
REAL = SCRATCH / "real_text"
CONTROL = SCRATCH / "runs/v2-expanded-cpython"
TRIAL = SCRATCH / "runs/v3-optimization-low-lr"
sys.path[:0] = [str(SCRATCH), str(REAL), str(HERE)]
from checkpoint import digest, load_checkpoint  # noqa: E402
from model import ScratchTransformer  # noqa: E402
from expanded_worker import evaluate, load_verified, probe_values  # noqa: E402
from audit import verify_ledger  # noqa: E402


def exact_tensor_restoration(path, model_config, tokenizer, config, provenance, val_windows, curve_row):
    model = ScratchTransformer(model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["trial_learning_rate"], weight_decay=config["weight_decay"])
    metadata = load_checkpoint(path, model, optimizer, expected_provenance=provenance)
    model_arrays = mx.load(str(path / "model.safetensors"))
    optimizer_arrays = mx.load(str(path / "adamw.safetensors"))
    restored_model = dict(tree_flatten(model.parameters()))
    restored_optimizer = dict(tree_flatten(optimizer.state))
    if set(restored_model) != set(model_arrays) or set(restored_optimizer) != set(optimizer_arrays):
        raise ValueError("Restored tensor keys differ from saved checkpoint")
    for key, original in model_arrays.items():
        if not bool(mx.all(restored_model[key] == original).item()):
            raise ValueError(f"Model tensor differs after restoration: {key}")
    for key, original in optimizer_arrays.items():
        if not bool(mx.all(restored_optimizer[key] == original).item()):
            raise ValueError(f"Optimizer tensor differs after restoration: {key}")
    if int(restored_optimizer["step"].item()) != metadata["step"]:
        raise ValueError("Optimizer step differs after restoration")
    model.eval()
    actual_train = evaluate(model, val_windows["train"])["loss"]
    actual_val = evaluate(model, val_windows["validation"])["loss"]
    if abs(actual_train - curve_row["train"]["loss"]) > 1e-5 or abs(actual_val - curve_row["validation"]["loss"]) > 1e-5:
        raise ValueError("Restored fixed-window losses differ")
    probe = json.loads((path / "probe_logits.json").read_text())["values"]
    probe_error = max(abs(a - b) for a, b in zip(probe, probe_values(model, val_windows["validation"][0])))
    if probe_error > 1e-6:
        raise ValueError("Restored probe differs")
    return {"step": metadata["step"], "exact_model_tensors": len(model_arrays),
            "exact_optimizer_tensors": len(optimizer_arrays), "optimizer_step": int(restored_optimizer["step"].item()),
            "fixed_train_loss": actual_train, "fixed_validation_loss": actual_val,
            "max_absolute_probe_logit_difference": probe_error,
            "model_sha256": digest(path / "model.safetensors"), "optimizer_sha256": digest(path / "adamw.safetensors")}


def analyze(trial_dir=TRIAL):
    trial_dir = Path(trial_dir).resolve()
    verify_ledger()
    config = json.loads((HERE / "v1/config.json").read_text())
    run = json.loads((trial_dir / "run.json").read_text())
    if run["status"] != "completed" or run["completed_steps"] != 80:
        raise ValueError("Trial did not complete 80 steps")
    control = {r["step"]: r for r in map(json.loads, (CONTROL / "curve.jsonl").read_text().splitlines())}
    trial = {r["step"]: r for r in map(json.loads, (trial_dir / "curve.jsonl").read_text().splitlines())}
    if sorted(control) != config["evaluation_steps"] or sorted(trial) != config["evaluation_steps"]:
        raise ValueError("Evaluation steps differ")
    rows = []
    for step in sorted(control):
        rows.append({"step": step, "control_train_loss": control[step]["train"]["loss"],
                     "control_validation_loss": control[step]["validation"]["loss"],
                     "trial_train_loss": trial[step]["train"]["loss"],
                     "trial_validation_loss": trial[step]["validation"]["loss"],
                     "control_validation_perplexity": control[step]["validation"]["perplexity"],
                     "trial_validation_perplexity": trial[step]["validation"]["perplexity"]})
    _, _, tokenizer, train, val, train_windows, val_windows = load_verified()
    model_config = json.loads((SCRATCH / "config.v1.json").read_text())["model"]
    worker_result = run["worker_result"]
    provenance = worker_result["provenance"]
    best_step = worker_result["best"]["step"]
    checks = {str(step): exact_tensor_restoration(trial_dir / f"checkpoints/step-{step:04d}", model_config,
                                                   tokenizer, config, provenance,
                                                   {"train": train_windows, "validation": val_windows}, trial[step])
              for step in sorted({best_step, 80})}
    steps = [json.loads(x) for x in (trial_dir / "steps.jsonl").read_text().splitlines()]
    if len(steps) != 80:
        raise ValueError("Step log incomplete")
    summaries = {}
    for key in ("gradient_l2_norm", "gradient_max_absolute", "parameter_l2_norm",
                "minibatch_loss_before_update", "duration_seconds", "mlx_peak_bytes"):
        values = [s[key] for s in steps]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"Non-finite step measurement: {key}")
        summaries[key] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
    return {"status": "verified", "config_sha256": digest(HERE / "v1/config.json"),
            "control_ledger_sha256": digest(CONTROL / "artifact_hashes.json"),
            "control_best_step": min(control.values(), key=lambda row: row["validation"]["loss"])["step"],
            "trial_best_step": best_step, "fixed_curve": rows, "trial_step_summaries": summaries,
            "trial_checkpoint_restoration": checks, "test_split_evaluated": False,
            "comparison_limit": "One paired run per learning rate; three validation documents; historical control has no per-step gradient norm log."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=TRIAL)
    parser.add_argument("--output-json", type=Path, default=HERE / "v1/paired_analysis.json")
    parser.add_argument("--output-csv", type=Path, default=HERE / "v1/paired_curve.csv")
    args = parser.parse_args()
    result = analyze(args.run_dir)
    output = args.output_json
    curve_csv = args.output_csv
    if output.exists() or curve_csv.exists():
        raise FileExistsError("Analysis outputs cannot be overwritten")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with curve_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["fixed_curve"][0]))
        writer.writeheader()
        writer.writerows(result["fixed_curve"])
    print(json.dumps({"status": result["status"], "control_best_step": result["control_best_step"],
                      "trial_best_step": result["trial_best_step"], "restored_steps": list(result["trial_checkpoint_restoration"])}))
