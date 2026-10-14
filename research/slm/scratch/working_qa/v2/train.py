"""One bounded comparison with candidate-specific neural evidence scoring."""
import argparse
import json
import math
import os
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
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.prepare import V1, sha, verify  # noqa: E402
from working_qa.train import rows, tokenizer  # noqa: E402
from working_qa.v2.model import PairReader, loss, parameter_count, predict  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402


def config():
    return json.loads((HERE / "config.json").read_text())


def plan():
    verify()
    if (HERE / "plan.json").exists():
        raise FileExistsError("v2 plan already frozen")
    c = config()
    v1 = json.loads((V1 / "plan.json").read_text())
    v1_steps = json.loads((V1 / "runs" / "main" / "run.json").read_text())["completed_steps"]
    tiny_steps = json.loads((V1 / "runs" / "tiny" / "run.json").read_text())["completed_steps"]
    if v1_steps + tiny_steps + c["training"]["tiny_steps_cap"] + c["training"]["main_steps_cap"] > c["training"]["combined_v1_v2_step_cap"]:
        raise ValueError("Combined preregistered step cap exceeded")
    p = {"schema_version": 2, "config_sha256": sha(HERE / "config.json"),
         "dataset_manifest_sha256": sha(V1 / "manifest.json"),
         "v1_plan_sha256": sha(V1 / "plan.json"),
         "model_sha256": sha(HERE / "model.py"),
         "prepare_sha256": sha(HERE / "prepare.py"),
         "tokenizer_sha256": sha(V1 / "tokenizer.json"),
         "tiny_ids": v1["tiny_ids"],
         "main_order": v1["main_order"][:c["training"]["main_steps_cap"]],
         "fixed_train_ids": v1["fixed_train_ids"],
         "fixed_validation_ids": v1["fixed_validation_ids"],
         "expected_parameter_count": parameter_count(PairReader(c["model"], tokenizer().vocab_size)),
         "test_sealed": True, "v1_completed_steps": v1_steps + tiny_steps}
    (HERE / "plan.json").write_text(json.dumps(p, indent=2) + "\n")
    return {key: value for key, value in p.items() if key not in ("tiny_ids", "main_order", "fixed_train_ids", "fixed_validation_ids")}


def load_plan():
    p = json.loads((HERE / "plan.json").read_text())
    if (p["config_sha256"] != sha(HERE / "config.json") or
        p["dataset_manifest_sha256"] != sha(V1 / "manifest.json") or
        p["v1_plan_sha256"] != sha(V1 / "plan.json") or
        p["model_sha256"] != sha(HERE / "model.py") or
        p["prepare_sha256"] != sha(HERE / "prepare.py") or
        p["tokenizer_sha256"] != sha(V1 / "tokenizer.json")):
        raise ValueError("v2 frozen plan hash mismatch")
    return p


def fresh(c, p):
    mx.random.seed(c["seed"])
    model = PairReader(c["model"], tokenizer().vocab_size)
    mx.eval(model.parameters())
    if parameter_count(model) != p["expected_parameter_count"]:
        raise ValueError("v2 parameter count changed")
    return model


def pair_data(tok, row):
    pairs, candidates = pair_inputs(tok, row["passage"], row["question"])
    if [(c["start"], c["end"]) for c in candidates] != [(c["start"], c["end"]) for c in row["candidates"]]:
        raise ValueError("Pair parser differs from frozen annotations")
    return pairs, candidates


def assess(model, entries):
    model.eval()
    losses, groups = [], defaultdict(lambda: [0, 0])
    for row, pairs, candidates in entries:
        value = loss(model, pairs, row["gold_candidate_index"])
        mx.eval(value)
        losses.append(float(value.item()))
        result = predict(model, pairs, candidates)
        good = result["prediction"] == row["expected_answer"]
        for group in ("all", "answerable" if row["answerable"] else "unanswerable", row["tier"]):
            groups[group][0] += int(good)
            groups[group][1] += 1
    return {"nll": statistics.mean(losses), "exact": groups["all"][0] / groups["all"][1],
            "balanced_accuracy": 0.5 * (groups["answerable"][0] / groups["answerable"][1] +
                                        groups["unanswerable"][0] / groups["unanswerable"][1]),
            "per_group": {key: {"correct": right, "count": count, "accuracy": right / count}
                          for key, (right, count) in groups.items()}}


def provenance():
    return {"config_sha256": sha(HERE / "config.json"),
            "plan_sha256": sha(HERE / "plan.json"),
            "dataset_manifest_sha256": sha(V1 / "manifest.json"),
            "train_sha256": sha(V1 / "train.jsonl"),
            "validation_sha256": sha(V1 / "validation.jsonl"),
            "tokenizer_sha256": sha(V1 / "tokenizer.json"),
            "model_sha256": sha(HERE / "model.py"),
            "prepare_sha256": sha(HERE / "prepare.py")}


def worker(mode, run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "working-qa-v2":
        raise RuntimeError("Live watchdog required")
    verify()
    c, p = config(), load_plan()
    settings = c["training"]
    if settings["main_steps_cap"] > 1450 or settings["tiny_steps_cap"] > 120 or settings["batch_size"] != 1:
        raise ValueError("Frozen v2 cap exceeded")
    if mode == "main":
        prior = json.loads((HERE / "runs" / "tiny" / "worker_result.json").read_text())
        if prior["status"] != "tiny_passed":
            raise RuntimeError("Tiny pairwise memorization gate failed")
    train, validation = rows("train"), rows("validation")
    by_id = {row["id"]: row for row in train + validation}
    needed = set(p["tiny_ids"] if mode == "tiny" else p["main_order"] + p["fixed_train_ids"] + p["fixed_validation_ids"])
    tok = tokenizer()
    data = {id: pair_data(tok, by_id[id]) for id in needed}
    order = ((p["tiny_ids"] * math.ceil(settings["tiny_steps_cap"] / 8))[:settings["tiny_steps_cap"]]
             if mode == "tiny" else p["main_order"])
    train_ids = p["tiny_ids"] if mode == "tiny" else p["fixed_train_ids"]
    val_ids = [] if mode == "tiny" else p["fixed_validation_ids"]
    entries = lambda ids: [(by_id[id], *data[id]) for id in ids]
    fixed_train, fixed_val = entries(train_ids), entries(val_ids)
    model = fresh(c, p)
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(model, loss)
    mx.metal.reset_peak_memory()
    start = time.monotonic()
    first = {"step": 0, "train": assess(model, fixed_train)}
    if fixed_val:
        first["validation"] = assess(model, fixed_val)
    best_key, best_step, stagnant, completed = (-1.0, -math.inf, 0), None, 0, 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps(first) + "\n")
        curves.flush()
        for step, id in enumerate(order, 1):
            if time.monotonic() - start >= (settings["tiny_wall_seconds"] if mode == "tiny" else settings["main_wall_seconds"]):
                raise TimeoutError("Worker wall cap")
            row, (pairs, _) = by_id[id], data[id]
            model.train()
            tick = time.monotonic()
            value, gradients = objective(model, pairs, row["gold_candidate_index"])
            mx.eval(value, gradients)
            gradients, norm = optim.clip_grad_norm(gradients, settings["gradient_clip_l2"])
            mx.eval(gradients, norm)
            loss_value, norm_value = float(value.item()), float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value <= 0:
                raise FloatingPointError("Invalid loss or gradients")
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            duration = time.monotonic() - tick
            token_count = sum(len(pair["ids"]) for pair in pairs)
            steps.write(json.dumps({"step": step, "id": id, "answerable": row["answerable"],
                                    "minibatch_nll": loss_value, "gradient_l2": norm_value,
                                    "tokens": token_count, "seconds": duration,
                                    "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            interval = settings["tiny_eval_interval"] if mode == "tiny" else settings["main_eval_interval"]
            if step % interval:
                continue
            current = {"step": step, "train": assess(model, fixed_train)}
            if fixed_val:
                current["validation"] = assess(model, fixed_val)
            curves.write(json.dumps(current) + "\n")
            curves.flush()
            print(json.dumps({"step": step, "train_exact": current["train"]["exact"],
                              "validation_exact": current.get("validation", {}).get("exact")}), flush=True)
            if mode == "tiny":
                if current["train"]["exact"] == 1.0:
                    save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                    step=step, cursor=step, provenance=provenance())
                    best_step = step
                    break
            else:
                save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                step=step, cursor=step, provenance=provenance())
                val = current["validation"]
                key = (val["balanced_accuracy"], -val["nll"], -step)
                if key > best_key:
                    best_key, best_step, stagnant = key, step, 0
                else:
                    stagnant += 1
                if step >= settings["minimum_steps_before_early_stop"] and stagnant >= settings["early_stop_patience_evaluations"]:
                    break
    curve = [json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()]
    status = ("tiny_passed" if best_step is not None else "tiny_failed") if mode == "tiny" else "main_completed"
    restoration = None
    if best_step is not None:
        restored = fresh(c, p)
        restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
        saved = run_dir / f"checkpoint-step-{best_step:04d}"
        meta = load_checkpoint(saved, restored, restored_optimizer, expected_provenance=provenance())
        prior = mx.load(str(saved / "model.safetensors"))
        actual = dict(tree_flatten(restored.parameters()))
        exact = set(prior) == set(actual) and all(bool(mx.all(prior[key] == actual[key]).item()) for key in prior)
        value = assess(restored, fixed_train if mode == "tiny" else fixed_val)["nll"]
        expected = next(x for x in curve if x["step"] == best_step)["train" if mode == "tiny" else "validation"]["nll"]
        if not exact or abs(value - expected) > 1e-5:
            raise ValueError("Restored model differs")
        restoration = {"exact_tensors": len(prior), "nll_difference": value - expected,
                       "optimizer_step": meta["step"]}
    result = {"status": status, "steps": completed, "best_step": best_step,
              "initial": first, "final": curve[-1],
              "best": next((x for x in curve if x["step"] == best_step), None),
              "restoration": restoration, "parameter_count": parameter_count(model),
              "peak_mlx_bytes": mx.metal.get_peak_memory(), "worker_wall_seconds": time.monotonic() - start,
              "test_file_opened": False, "provenance": provenance()}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(mode):
    c, p = config(), load_plan()
    run_dir = HERE / "runs" / mode
    if run_dir.exists():
        raise FileExistsError("Run already exists")
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "plan_sha256": sha(HERE / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "working-qa-v2"
    limit = c["training"]["tiny_wall_seconds" if mode == "tiny" else "main_wall_seconds"]
    start = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", mode],
                                   cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, before["resources"], before["safety"])
            if time.monotonic() - start > limit:
                reasons.append("Parent wall cap exceeded")
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
    result["wall_seconds"] = time.monotonic() - start
    result["after"] = snapshot()
    path = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    result["completed_steps"] = len(steps)
    result["tokens_processed"] = sum(x["tokens"] for x in steps)
    result["tokens_per_second"] = result["tokens_processed"] / sum(x["seconds"] for x in steps) if steps else None
    result["mean_step_seconds"] = statistics.mean(x["seconds"] for x in steps) if steps else None
    result["peak_worker_rss_bytes"] = max((x["process_rss_bytes"] or 0 for x in result["snapshots"]), default=None)
    result["minimum_available_bytes"] = min((x["available_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((x["memory_pressure_free_percent"] for x in result["snapshots"] if x["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((x["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((x["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((x["disk_free_bytes"] for x in result["snapshots"]), default=None)
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
    print(json.dumps(plan() if args.plan else execute(args.execute) if args.execute else worker(args.worker, HERE / "runs" / args.worker), indent=2))
