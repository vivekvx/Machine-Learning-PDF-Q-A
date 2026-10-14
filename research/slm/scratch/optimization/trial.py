"""Opt-in, guarded v2 paired trial. The only optimization change is AdamW LR."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
REAL = SCRATCH / "real_text"
TRIAL = HERE / "v1/config.json"
AUDIT = HERE / "v1/frozen_checkpoint_reevaluation.json"
sys.path[:0] = [str(SCRATCH), str(REAL), str(HERE)]
from checkpoint import digest  # noqa: E402
from model import ScratchTransformer, next_token_loss, parameter_count  # noqa: E402
from expanded_worker import evaluate, generate, load_verified, probe_values, checkpoint_at, restore_and_verify  # noqa: E402
from audit import tensor_stats, window_sha, verify_ledger  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402


def verified_setup():
    spec = json.loads(TRIAL.read_text())
    old, manifest, tok, train, val, train_windows, val_windows = load_verified()
    audit = json.loads(AUDIT.read_text())
    verify_ledger()
    if audit["status"] != "verified" or audit["train_window_sha256"] != window_sha(train_windows) or audit["validation_window_sha256"] != window_sha(val_windows):
        raise ValueError("Frozen-window audit invalid")
    expected = {"control_learning_rate": old["learning_rate"], "weight_decay": old["weight_decay"],
                "optimizer_steps": old["max_optimizer_steps"], "batch_size": old["batch_size"],
                "sequence_length": old["sequence_length"], "seed": old["seed"],
                "wall_time_cap_seconds": old["wall_time_cap_seconds"]}
    if any(spec[k] != v for k, v in expected.items()) or spec["trial_learning_rate"] != 0.0001:
        raise ValueError("Preregistered trial differs from v2 controls")
    if spec["optimizer_steps"] > 80 or spec["batch_size"] != 1 or spec["wall_time_cap_seconds"] > 180:
        raise ValueError("Trial safety cap exceeded")
    if [json.loads(x)["train_document_id"] for x in (SCRATCH / "runs/v2-expanded-cpython/steps.jsonl").read_text().splitlines()] != [train_windows[(step - 1) % len(train_windows)]["document_id"] for step in range(1, 81)]:
        raise ValueError("Control sample order differs from preregistration")
    return spec, old, manifest, tok, train, val, train_windows, val_windows


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "optimization-v1":
        raise RuntimeError("Live memory watchdog required")
    spec, old, manifest, tokenizer, train, val, train_windows, val_windows = verified_setup()
    model_config = json.loads((SCRATCH / "config.v1.json").read_text())["model"]
    mx.random.seed(spec["seed"])
    model = ScratchTransformer(model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=spec["trial_learning_rate"], weight_decay=spec["weight_decay"])
    mx.eval(model.parameters())
    if parameter_count(model) != 1082496:
        raise ValueError("Parameter count changed")
    model.eval()
    initial = {"step": 0, "train": evaluate(model, train_windows), "validation": evaluate(model, val_windows)}
    control_initial = json.loads((SCRATCH / "runs/v2-expanded-cpython/curve.jsonl").read_text().splitlines()[0])
    if any(abs(initial[k]["loss"] - control_initial[k]["loss"]) > 1e-6 for k in ("train", "validation")):
        raise ValueError("Initial model, data, or evaluation changed from control")
    model.train()
    initial_probe = probe_values(model, val_windows[0])
    mx.metal.reset_peak_memory()
    objective = nn.value_and_grad(model, next_token_loss)
    provenance = {"trial_config_sha256": digest(TRIAL), "audit_sha256": digest(AUDIT),
                  "corpus_manifest_sha256": digest(REAL / "v2/manifest.json"),
                  "train_sha256": digest(REAL / "v2/train.jsonl"), "validation_sha256": digest(REAL / "v2/validation.jsonl"),
                  "tokenizer_sha256": digest(REAL / "v2/tokenizer.json"),
                  "base_model_config_sha256": digest(SCRATCH / "config.v1.json"),
                  "model_source_sha256": digest(SCRATCH / "model.py"), "worker_source_sha256": digest(__file__)}
    curve = [initial]
    checkpoints = {}
    started = time.monotonic()
    completed = 0
    with (run_dir / "steps.jsonl").open("x") as steps_file, (run_dir / "curve.jsonl").open("x") as curve_file:
        curve_file.write(json.dumps(initial, sort_keys=True) + "\n")
        curve_file.flush()
        for step in range(1, spec["optimizer_steps"] + 1):
            if time.monotonic() - started >= spec["wall_time_cap_seconds"]:
                raise TimeoutError("Worker wall-time cap reached")
            w = train_windows[(step - 1) % len(train_windows)]
            ids = w["ids"]
            x = mx.array([ids[:-1]], dtype=mx.int32)
            y = mx.array([ids[1:]], dtype=mx.int32)
            tick = time.monotonic()
            loss, grads = objective(model, x, y)
            mx.eval(loss, grads)
            grad_stats = tensor_stats(grads)
            if not bool(mx.isfinite(loss).item()) or not grad_stats["all_finite"] or grad_stats["l2_norm"] <= 0:
                raise FloatingPointError(f"Invalid loss or gradient at step {step}")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            parameter_stats = tensor_stats(model.parameters())
            if not parameter_stats["all_finite"] or not tensor_stats(optimizer.state)["all_finite"]:
                raise FloatingPointError(f"Invalid parameter or optimizer state at step {step}")
            duration = time.monotonic() - tick
            completed = step
            record = {"step": step, "train_document_id": w["document_id"], "window_offset": w["offset"],
                      "tokens": spec["sequence_length"], "minibatch_loss_before_update": float(loss.item()),
                      "gradient_l2_norm": grad_stats["l2_norm"], "gradient_max_absolute": grad_stats["max_absolute"],
                      "parameter_l2_norm": parameter_stats["l2_norm"], "parameter_max_absolute": parameter_stats["max_absolute"],
                      "duration_seconds": duration, "tokens_per_second": spec["sequence_length"] / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(), "mlx_cache_bytes": mx.metal.get_cache_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            steps_file.write(json.dumps(record, sort_keys=True) + "\n")
            steps_file.flush()
            if step in spec["evaluation_steps"]:
                model.eval()
                assessment = {"step": step, "train": evaluate(model, train_windows), "validation": evaluate(model, val_windows)}
                model.train()
                if not math.isfinite(assessment["train"]["loss"]) or not math.isfinite(assessment["validation"]["loss"]):
                    raise FloatingPointError("Non-finite fixed-window evaluation")
                if assessment["train"]["loss"] > 2 * initial["train"]["loss"]:
                    raise RuntimeError("Training divergence abort")
                curve.append(assessment)
                curve_file.write(json.dumps(assessment, sort_keys=True) + "\n")
                curve_file.flush()
                os.fsync(curve_file.fileno())
                print(json.dumps({"step": step, "train_loss": assessment["train"]["loss"],
                                  "validation_loss": assessment["validation"]["loss"]}), flush=True)
                if step in spec["checkpoint_steps"]:
                    checkpoints[step], _ = checkpoint_at(run_dir, step, model, optimizer, val_windows[0], provenance, step % len(train_windows))
    if completed != 80 or len(curve) != 9 or len(checkpoints) != 8:
        raise RuntimeError("Paired trial did not complete preregistered budget")
    best = min(curve[1:], key=lambda row: row["validation"]["loss"])
    best_model, best_restore = restore_and_verify(checkpoints[best["step"]], model_config, tokenizer,
                                                 {"learning_rate": spec["trial_learning_rate"], "weight_decay": spec["weight_decay"]},
                                                 provenance, val_windows[0])
    final_model, final_restore = restore_and_verify(checkpoints[80], model_config, tokenizer,
                                                   {"learning_rate": spec["trial_learning_rate"], "weight_decay": spec["weight_decay"]},
                                                   provenance, val_windows[0])
    memory_final_diff = max(abs(a - b) for a, b in zip(probe_values(model, val_windows[0]), probe_values(final_model, val_windows[0])))
    if memory_final_diff > 1e-6:
        raise RuntimeError("Final checkpoint failed exact probe restoration")
    generations = {"policy": "Frozen v2 first-100-character prompt; greedy 32-token continuation",
                   "best": [generate(best_model, tokenizer, row, old["generation_tokens"]) for row in (train[0], val[0], val[2])],
                   "final": [generate(final_model, tokenizer, row, old["generation_tokens"]) for row in (train[0], val[0], val[2])]}
    (run_dir / "generations.json").write_text(json.dumps(generations, indent=2) + "\n")
    result = {"status": "completed", "completed_steps": completed, "parameter_count": parameter_count(model),
              "initial_probe_first_eight": initial_probe[:8], "initial": initial, "best": best, "final": curve[-1],
              "best_restore": best_restore, "final_restore": final_restore,
              "final_in_memory_probe_max_abs_difference": memory_final_diff,
              "fresh_random_initialization": True, "prior_checkpoint_loaded": False, "test_file_opened": False,
              "worker_wall_seconds": time.monotonic() - started, "mlx_peak_bytes": mx.metal.get_peak_memory(),
              "mlx_active_bytes": mx.metal.get_active_memory(), "provenance": provenance}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Run artifacts cannot be overwritten")
    spec, *_ = verified_setup()
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running", "preflight": before,
              "snapshots": [], "trial_config_sha256": digest(TRIAL)}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "optimization-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", str(run_dir)],
                                   cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, before["resources"], before["safety"])
            if time.monotonic() - started > spec["wall_time_cap_seconds"]:
                reasons.append("parent wall-time cap exceeded")
            if reasons:
                result["stop_reasons"] = reasons
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.5)
        result["exit_code"] = process.wait()
    result["wall_seconds"] = time.monotonic() - started
    result["after"] = snapshot()
    step_file = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in step_file.read_text().splitlines()] if step_file.exists() else []
    result["completed_steps"] = len(steps)
    result["tokens_per_second"] = sum(s["tokens"] for s in steps) / sum(s["duration_seconds"] for s in steps) if steps else None
    result["mean_step_seconds"] = sum(s["duration_seconds"] for s in steps) / len(steps) if steps else None
    result["peak_sampled_worker_rss_bytes"] = max((s["process_rss_bytes"] or 0 for s in result["snapshots"]), default=None)
    result["minimum_system_available_bytes"] = min((s["available_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((s["memory_pressure_free_percent"] for s in result["snapshots"] if s["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((s["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for s in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((s["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((s["disk_free_bytes"] for s in result["snapshots"]), default=None)
    output = run_dir / "worker_result.json"
    if output.exists():
        result["worker_result"] = json.loads(output.read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and result.get("worker_result") and len(steps) == 80
                        else "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
    elif args.execute:
        result = execute(args.execute)
        print(json.dumps({k: v for k, v in result.items() if k != "snapshots"}, indent=2))
    else:
        print(json.dumps({"mode": "preflight_only", **preflight("smoke")}, indent=2))
