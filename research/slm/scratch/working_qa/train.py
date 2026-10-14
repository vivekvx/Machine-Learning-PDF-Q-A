"""Frozen-plan, guarded MLX training for the scratch candidate reader."""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.model import CandidateReader, loss, parameter_count, predict  # noqa: E402
from working_qa.prepare import V1, V3ByteBPE, sha, tokenize, verify  # noqa: E402


def config():
    return json.loads((V1 / "config.json").read_text())


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Test partition is sealed during training and model selection")
    return [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]


def tokenizer():
    info = json.loads((V1 / "tokenizer.json").read_text())
    return V3ByteBPE(info["merges"], info["target_size"])


def fixed_ids(data, count):
    if count % 8:
        raise ValueError("Fixed count must divide four tiers and two labels")
    cells = defaultdict(list)
    for row in data:
        cells[(row["tier"], row["answerable"])].append(row["id"])
    return [id for tier in config()["tiers"] for label in (True, False)
            for id in cells[(tier, label)][:count // 8]]


def plan():
    verify()
    destination = V1 / "plan.json"
    if destination.exists():
        raise FileExistsError("Training plan already frozen")
    c = config()
    train, validation = rows("train"), rows("validation")
    order = [row["id"] for row in train]
    rng = random.Random(c["seed"])
    rng.shuffle(order)
    # All unique examples are seen before any repetition. The remaining 600
    # steps replay a fresh deterministic shuffled prefix.
    repeat = order.copy()
    rng.shuffle(repeat)
    order += repeat[:c["training"]["main_optimizer_steps_cap"] - len(order)]
    tiny = fixed_ids(train, 8)
    output = {"schema_version": 1, "config_sha256": sha(V1 / "config.json"),
              "manifest_sha256": sha(V1 / "manifest.json"),
              "tokenizer_sha256": sha(V1 / "tokenizer.json"),
              "model_sha256": sha(HERE / "model.py"),
              "prepare_sha256": sha(HERE / "prepare.py"),
              "tiny_ids": tiny, "main_order": order,
              "fixed_train_ids": fixed_ids(train, c["training"]["fixed_train_examples"]),
              "fixed_validation_ids": fixed_ids(validation, c["training"]["fixed_validation_examples"]),
              "expected_parameter_count": parameter_count(CandidateReader(c["model"], tokenizer().vocab_size)),
              "test_sealed": True, "checkpoint_selection": c["checkpoint_selection"]}
    destination.write_text(json.dumps(output, indent=2) + "\n")
    return {k: v for k, v in output.items() if not k.endswith("ids") and k != "main_order"}


def load_plan():
    p = json.loads((V1 / "plan.json").read_text())
    expected = {"config_sha256": sha(V1 / "config.json"),
                "manifest_sha256": sha(V1 / "manifest.json"),
                "tokenizer_sha256": sha(V1 / "tokenizer.json"),
                "model_sha256": sha(HERE / "model.py"),
                "prepare_sha256": sha(HERE / "prepare.py")}
    if any(p[key] != value for key, value in expected.items()):
        raise ValueError("Frozen plan input hash mismatch")
    return p


def fresh_model(c, p):
    mx.random.seed(c["seed"])
    model = CandidateReader(c["model"], tokenizer().vocab_size)
    mx.eval(model.parameters())
    if parameter_count(model) != p["expected_parameter_count"]:
        raise ValueError("Model parameter count changed")
    return model


def target(row):
    return row["gold_candidate_index"] + 1 if row["answerable"] else 0


def assess(model, entries):
    model.eval()
    losses, metrics = [], defaultdict(lambda: [0, 0])
    for row, enc in entries:
        ids = mx.array([enc["ids"]], dtype=mx.int32)
        value = loss(model, ids, enc["question_end"], enc["candidates"], target(row))
        mx.eval(value)
        losses.append(float(value.item()))
        result = predict(model, enc, row["candidates"])
        correct = result["prediction"] == row["expected_answer"]
        for key in ("all", "answerable" if row["answerable"] else "unanswerable", row["tier"]):
            metrics[key][0] += int(correct)
            metrics[key][1] += 1
    return {"nll": statistics.mean(losses), "exact": metrics["all"][0] / metrics["all"][1],
            "balanced_accuracy": 0.5 * (metrics["answerable"][0] / metrics["answerable"][1] +
                                        metrics["unanswerable"][0] / metrics["unanswerable"][1]),
            "per_group": {key: {"correct": good, "count": total, "accuracy": good / total}
                          for key, (good, total) in metrics.items()}}


def provenance():
    return {"config_sha256": sha(V1 / "config.json"), "manifest_sha256": sha(V1 / "manifest.json"),
            "plan_sha256": sha(V1 / "plan.json"), "train_sha256": sha(V1 / "train.jsonl"),
            "validation_sha256": sha(V1 / "validation.jsonl"),
            "tokenizer_sha256": sha(V1 / "tokenizer.json"), "model_sha256": sha(HERE / "model.py")}


def worker(mode, run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "working-qa-v1":
        raise RuntimeError("Direct worker launch is forbidden; live parent watchdog required")
    verify()
    c, p = config(), load_plan()
    settings = c["training"]
    if (settings["total_optimizer_step_cap"] > 3400 or
        settings["total_wall_time_cap_seconds"] > 420 or
        settings["batch_size"] != 1 or settings["main_optimizer_steps_cap"] > 3000):
        raise ValueError("Preregistered compute ceiling exceeded")
    if mode not in ("tiny", "main"):
        raise ValueError("Unknown mode")
    if mode == "main":
        tiny_result = json.loads((V1 / "runs" / "tiny" / "worker_result.json").read_text())
        if tiny_result["status"] != "tiny_passed" or tiny_result["exact"] != 1.0:
            raise RuntimeError("Tiny full-task memorization did not pass")
    tok = tokenizer()
    train, validation = rows("train"), rows("validation")
    by_id = {row["id"]: row for row in train + validation}
    ids_needed = set((p["tiny_ids"] if mode == "tiny" else p["main_order"]) +
                     (p["tiny_ids"] if mode == "tiny" else p["fixed_train_ids"] + p["fixed_validation_ids"]))
    encoded = {id: tokenize(tok, by_id[id]) for id in ids_needed}
    order = p["tiny_ids"] * math.ceil(settings["tiny_optimizer_steps_cap"] / len(p["tiny_ids"])) if mode == "tiny" else p["main_order"]
    cap = settings["tiny_optimizer_steps_cap"] if mode == "tiny" else settings["main_optimizer_steps_cap"]
    order = order[:cap]
    fixed_train_ids = p["tiny_ids"] if mode == "tiny" else p["fixed_train_ids"]
    fixed_val_ids = [] if mode == "tiny" else p["fixed_validation_ids"]
    fixed_train = [(by_id[id], encoded[id]) for id in fixed_train_ids]
    fixed_val = [(by_id[id], encoded[id]) for id in fixed_val_ids]
    model = fresh_model(c, p)
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(model, loss)
    mx.metal.reset_peak_memory()
    started = time.monotonic()
    initial = {"step": 0, "train": assess(model, fixed_train)}
    if fixed_val:
        initial["validation"] = assess(model, fixed_val)
    best_key, best_step, stagnant = (-1.0, -math.inf, 0), None, 0
    completed = 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps(initial) + "\n")
        curves.flush()
        for step, id in enumerate(order, 1):
            if time.monotonic() - started >= (settings["tiny_wall_time_cap_seconds"] if mode == "tiny" else settings["main_wall_time_cap_seconds"]):
                raise TimeoutError("Worker wall-time cap")
            row, enc = by_id[id], encoded[id]
            model.train()
            tick = time.monotonic()
            value, gradients = objective(model, mx.array([enc["ids"]], dtype=mx.int32),
                                         enc["question_end"], enc["candidates"], target(row))
            mx.eval(value, gradients)
            gradients, norm = optim.clip_grad_norm(gradients, settings["gradient_clip_l2"])
            mx.eval(gradients, norm)
            loss_value, norm_value = float(value.item()), float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value <= 0:
                raise FloatingPointError("Nonfinite or zero loss/gradient")
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            duration = time.monotonic() - tick
            steps.write(json.dumps({"step": step, "id": id, "answerable": row["answerable"],
                                    "minibatch_nll": loss_value, "gradient_l2": norm_value,
                                    "tokens": len(enc["ids"]), "seconds": duration,
                                    "tokens_per_second": len(enc["ids"]) / duration,
                                    "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            interval = settings["tiny_evaluation_interval_steps"] if mode == "tiny" else settings["main_evaluation_interval_steps"]
            if step % interval:
                continue
            assessment = {"step": step, "train": assess(model, fixed_train)}
            if fixed_val:
                assessment["validation"] = assess(model, fixed_val)
            curves.write(json.dumps(assessment) + "\n")
            curves.flush()
            print(json.dumps({"step": step, "train_exact": assessment["train"]["exact"],
                              "validation_exact": assessment.get("validation", {}).get("exact")}), flush=True)
            if mode == "tiny":
                if assessment["train"]["exact"] == 1.0:
                    save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                    step=step, cursor=step, provenance=provenance())
                    break
            else:
                save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                step=step, cursor=step, provenance=provenance())
                val = assessment["validation"]
                key = (val["balanced_accuracy"], -val["nll"], -step)
                if key > best_key:
                    best_key, best_step, stagnant = key, step, 0
                else:
                    stagnant += 1
                if (step >= settings["minimum_steps_before_early_stop"] and
                    stagnant >= settings["early_stop_patience_evaluations"]):
                    break
    final_curve = [json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()]
    if mode == "tiny":
        passed = final_curve[-1]["train"]["exact"] == 1.0
        best_step = completed if passed else None
        status = "tiny_passed" if passed else "tiny_failed"
    else:
        passed, status = True, "main_completed"
    restoration = None
    if best_step is not None:
        restored = fresh_model(c, p)
        restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
        metadata = load_checkpoint(run_dir / f"checkpoint-step-{best_step:04d}", restored,
                                   restored_optimizer, expected_provenance=provenance())
        before = mx.load(str(run_dir / f"checkpoint-step-{best_step:04d}" / "model.safetensors"))
        after = dict(tree_flatten(restored.parameters()))
        exact = set(before) == set(after) and all(bool(mx.all(before[key] == after[key]).item()) for key in before)
        fixed = fixed_train if mode == "tiny" else fixed_val
        restored_eval = assess(restored, fixed)
        original_eval = next(item["train" if mode == "tiny" else "validation"] for item in final_curve if item["step"] == best_step)
        if not exact or abs(restored_eval["nll"] - original_eval["nll"]) > 1e-5:
            raise ValueError("Checkpoint restoration differs")
        restoration = {"exact_tensors": len(before), "nll_difference": restored_eval["nll"] - original_eval["nll"],
                       "optimizer_step": metadata["step"]}
    result = {"status": status, "steps": completed, "best_step": best_step,
              "exact": final_curve[-1]["train"]["exact"] if mode == "tiny" else None,
              "initial": initial, "final": final_curve[-1],
              "best": next((item for item in final_curve if item["step"] == best_step), None),
              "restoration": restoration, "parameter_count": parameter_count(model),
              "peak_mlx_bytes": mx.metal.get_peak_memory(),
              "worker_wall_seconds": time.monotonic() - started,
              "test_file_opened": False, "provenance": provenance()}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(mode):
    c, p = config(), load_plan()
    run_dir = V1 / "runs" / mode
    if run_dir.exists():
        raise FileExistsError("Run already exists and cannot be overwritten")
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "plan_sha256": sha(V1 / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "working-qa-v1"
    wall = c["training"]["tiny_wall_time_cap_seconds" if mode == "tiny" else "main_wall_time_cap_seconds"]
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", mode],
                                cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            observed = snapshot(proc.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic() - started > wall:
                reasons.append("Parent wall-time cap exceeded")
            if reasons:
                result["stop_reasons"] = reasons
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(0.5)
        result["exit_code"] = proc.wait()
    result["wall_seconds"] = time.monotonic() - started
    result["after"] = snapshot()
    lines = (run_dir / "steps.jsonl").read_text().splitlines() if (run_dir / "steps.jsonl").exists() else []
    steps = [json.loads(line) for line in lines]
    result["completed_steps"] = len(steps)
    result["tokens_processed"] = sum(item["tokens"] for item in steps)
    result["tokens_per_second"] = (result["tokens_processed"] / sum(item["seconds"] for item in steps)) if steps else None
    result["mean_step_seconds"] = statistics.mean(item["seconds"] for item in steps) if steps else None
    result["peak_worker_rss_bytes"] = max((s["process_rss_bytes"] or 0 for s in result["snapshots"]), default=None)
    result["minimum_available_bytes"] = min((s["available_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((s["memory_pressure_free_percent"] for s in result["snapshots"] if s["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((s["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for s in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((s["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((s["disk_free_bytes"] for s in result["snapshots"]), default=None)
    if (run_dir / "worker_result.json").exists():
        result["worker_result"] = json.loads((run_dir / "worker_result.json").read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and "worker_result" in result else
                        "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "snapshots"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", choices=("tiny", "main"))
    parser.add_argument("--worker", choices=("tiny", "main"))
    args = parser.parse_args()
    if sum(bool(x) for x in (args.plan, args.execute, args.worker)) != 1:
        parser.error("Choose exactly one action")
    print(json.dumps(plan() if args.plan else execute(args.execute) if args.execute else worker(args.worker, V1 / "runs" / args.worker), indent=2))
