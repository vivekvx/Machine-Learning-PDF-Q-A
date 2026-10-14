"""Guarded, bounded fresh-weight extractive QA training and checkpoint selection."""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
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
from extractive_qa.prepare import V1, V3ByteBPE, sha, tokenized, verify  # noqa: E402
from extractive_qa.model import ExtractiveTransformer, loss, parameter_count  # noqa: E402


def config():
    return json.loads((V1 / "config.json").read_text())


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Sealed test cannot be opened during training")
    return [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]


def tokenizer():
    info = json.loads((V1 / "tokenizer.json").read_text())
    return V3ByteBPE(info["merges"], info["target_size"])


def plan():
    verify()
    if (V1 / "plan.json").exists():
        raise FileExistsError("Plan already exists")
    c = config()
    training, validation = rows("train"), rows("validation")
    rng = random.Random(c["seed"])
    positives, negatives = ([r["id"] for r in training if r["answerable"]],
                            [r["id"] for r in training if not r["answerable"]])
    rng.shuffle(positives)
    rng.shuffle(negatives)
    budget = c["training"]["maximum_optimizer_steps"]
    if budget % 2 or min(len(positives), len(negatives)) < budget // 2:
        raise ValueError("Cannot form balanced non-repeated sample order")
    order = [id for pair in zip(positives[:budget // 2], negatives[:budget // 2]) for id in pair]
    def fixed(subset, count):
        per_cell = count // 8
        if per_cell * 8 != count:
            raise ValueError("Fixed evaluation count must divide eight field/label cells")
        return [r["id"] for field in c["fields"] for answerable in (True, False)
                for r in [x for x in subset if x["asked_field"] == field and x["answerable"] == answerable][:per_cell]]
    output = {"schema_version": 1, "config_sha256": sha(V1 / "config.json"),
              "manifest_sha256": sha(V1 / "manifest.json"), "model_source_sha256": sha(HERE / "model.py"),
              "sample_order": order, "fixed_train_ids": fixed(training, c["training"]["fixed_train_examples"]),
              "fixed_validation_ids": fixed(validation, c["training"]["fixed_validation_examples"]),
              "selection": c["checkpoint_selection"], "test_sealed": True,
              "parameter_count_expected": parameter_count(ExtractiveTransformer(c["model"], tokenizer().vocab_size))}
    (V1 / "plan.json").write_text(json.dumps(output, indent=2) + "\n")
    return {k: v for k, v in output.items() if k not in ("sample_order", "fixed_train_ids", "fixed_validation_ids")}


def load_plan():
    p = json.loads((V1 / "plan.json").read_text())
    if p["config_sha256"] != sha(V1 / "config.json") or p["manifest_sha256"] != sha(V1 / "manifest.json") or p["model_source_sha256"] != sha(HERE / "model.py"):
        raise ValueError("Frozen plan inputs changed")
    return p


def new_model(c):
    mx.random.seed(c["seed"])
    model = ExtractiveTransformer(c["model"], tokenizer().vocab_size)
    mx.eval(model.parameters())
    if parameter_count(model) != load_plan()["parameter_count_expected"]:
        raise ValueError("Parameter count changed")
    return model


def row_loss(model, encoded, row, weights):
    ids = mx.array([encoded["ids"]], dtype=mx.int32)
    return loss(model, ids, encoded["passage_start"], encoded["passage_end"],
                row["answerable"], encoded["answer_start_token"] or 0,
                encoded["answer_end_token"] or 0, weights["answerability_loss_weight"],
                weights["span_loss_weight"])


def fixed_loss(model, entries, weights):
    model.eval()
    by_category = {"answerable": [], "unanswerable": []}
    for row, encoded in entries:
        value = row_loss(model, encoded, row, weights)
        mx.eval(value)
        by_category["answerable" if row["answerable"] else "unanswerable"].append(float(value.item()))
    values = by_category["answerable"] + by_category["unanswerable"]
    result = {"joint_loss": statistics.mean(values), "examples": len(values),
              "answerable_joint_loss": statistics.mean(by_category["answerable"]),
              "unanswerable_joint_loss": statistics.mean(by_category["unanswerable"])}
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError("Nonfinite fixed-set loss")
    return result


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "extractive-qa-v1":
        raise RuntimeError("Live watchdog required")
    verify()
    c = config()
    settings = c["training"]
    if settings["maximum_optimizer_steps"] > 1600 or settings["wall_time_cap_seconds"] > 180 or settings["batch_size"] != 1:
        raise ValueError("Hard training cap exceeded")
    p = load_plan()
    tok = tokenizer()
    training, validation = rows("train"), rows("validation")
    by_id = {r["id"]: r for r in training + validation}
    encoded = {id: tokenized(tok, row) for id, row in by_id.items()}
    fixed_train = [(by_id[id], encoded[id]) for id in p["fixed_train_ids"]]
    fixed_val = [(by_id[id], encoded[id]) for id in p["fixed_validation_ids"]]
    model = new_model(c)
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    mx.metal.reset_peak_memory()
    started = time.monotonic()
    curve = [{"step": 0, "train": fixed_loss(model, fixed_train, settings),
              "validation": fixed_loss(model, fixed_val, settings)}]
    objective = nn.value_and_grad(model, loss)
    provenance = {"config_sha256": sha(V1 / "config.json"), "manifest_sha256": sha(V1 / "manifest.json"),
                  "plan_sha256": sha(V1 / "plan.json"), "train_sha256": sha(V1 / "train.jsonl"),
                  "validation_sha256": sha(V1 / "validation.jsonl"),
                  "tokenizer_sha256": sha(V1 / "tokenizer.json"), "model_sha256": sha(HERE / "model.py")}
    best_step, best_loss, stagnant = 0, curve[0]["validation"]["joint_loss"], 0
    completed = 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps(curve[0]) + "\n")
        for step, id in enumerate(p["sample_order"], 1):
            if time.monotonic() - started >= settings["wall_time_cap_seconds"]:
                raise TimeoutError("Worker wall-time cap")
            row, enc = by_id[id], encoded[id]
            model.train()
            tick = time.monotonic()
            value, grads = objective(model, mx.array([enc["ids"]], dtype=mx.int32), enc["passage_start"],
                                     enc["passage_end"], row["answerable"], enc["answer_start_token"] or 0,
                                     enc["answer_end_token"] or 0, settings["answerability_loss_weight"],
                                     settings["span_loss_weight"])
            mx.eval(value, grads)
            grad_sq = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(grads))
            if not math.isfinite(float(value.item())) or not math.isfinite(grad_sq) or grad_sq <= 0:
                raise FloatingPointError("Invalid loss or gradients")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            param_sq = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(model.parameters()))
            if not math.isfinite(param_sq):
                raise FloatingPointError("Invalid model parameters")
            duration = time.monotonic() - tick
            steps.write(json.dumps({"step": step, "id": id, "answerable": row["answerable"],
                                    "minibatch_joint_loss": float(value.item()), "gradient_l2_norm": math.sqrt(grad_sq),
                                    "parameter_l2_norm": math.sqrt(param_sq), "tokens": len(enc["ids"]),
                                    "duration_seconds": duration, "tokens_per_second": len(enc["ids"]) / duration,
                                    "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            if step % settings["evaluation_interval_steps"] == 0:
                assessment = {"step": step, "train": fixed_loss(model, fixed_train, settings),
                              "validation": fixed_loss(model, fixed_val, settings)}
                curve.append(assessment)
                curves.write(json.dumps(assessment) + "\n")
                curves.flush()
                os.fsync(curves.fileno())
                print(json.dumps({"step": step, "train_joint": assessment["train"]["joint_loss"],
                                  "validation_joint": assessment["validation"]["joint_loss"]}), flush=True)
                if assessment["train"]["joint_loss"] > 2 * curve[0]["train"]["joint_loss"] or assessment["validation"]["joint_loss"] > curve[0]["validation"]["joint_loss"] + 0.5:
                    raise RuntimeError("Fixed-set divergence abort")
                if step % settings["checkpoint_interval_steps"] == 0:
                    save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                    step=step, cursor=step, provenance=provenance)
                if assessment["validation"]["joint_loss"] < best_loss - 1e-9:
                    best_loss, best_step, stagnant = assessment["validation"]["joint_loss"], step, 0
                else:
                    stagnant += 1
                if step >= settings["minimum_steps_before_early_stop"] and stagnant >= settings["early_stop_patience_evaluations"]:
                    break
    if best_step == 0:
        raise RuntimeError("No checkpoint improved on initial validation loss")
    selected = run_dir / f"checkpoint-step-{best_step:04d}"
    restored = new_model(c)
    opt_restored = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    metadata = load_checkpoint(selected, restored, opt_restored, expected_provenance=provenance)
    saved = mx.load(str(selected / "model.safetensors"))
    actual = dict(tree_flatten(restored.parameters()))
    if set(actual) != set(saved) or any(not bool(mx.all(actual[k] == saved[k]).item()) for k in saved):
        raise ValueError("Restored model tensor mismatch")
    restored_val = fixed_loss(restored, fixed_val, settings)
    difference = abs(restored_val["joint_loss"] - best_loss)
    if difference > 1e-5 or metadata["step"] != best_step:
        raise ValueError("Restored checkpoint loss or optimizer step mismatch")
    result = {"status": "completed", "completed_steps": completed, "best_step": best_step,
              "initial": curve[0], "best": next(x for x in curve if x["step"] == best_step), "final": curve[-1],
              "restoration": {"exact_model_tensors": len(saved), "validation_loss_difference": difference,
                              "optimizer_step": metadata["step"]},
              "parameter_count": parameter_count(model), "peak_mlx_bytes": mx.metal.get_peak_memory(),
              "test_file_opened": False, "provenance": provenance,
              "worker_wall_seconds": time.monotonic() - started}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Run directory exists")
    verify()
    load_plan()
    c = config()
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running", "preflight": before,
              "snapshots": [], "config_sha256": sha(V1 / "config.json"), "plan_sha256": sha(V1 / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    environment = os.environ.copy()
    environment["SLMA_SAFETY_PARENT"] = "extractive-qa-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", str(run_dir)],
                                cwd=SCRATCH, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            observed = snapshot(proc.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic() - started > c["training"]["wall_time_cap_seconds"]:
                reasons.append("parent wall-time cap exceeded")
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
    path = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    result["completed_steps"] = len(steps)
    result["tokens_processed"] = sum(x["tokens"] for x in steps)
    result["tokens_per_second"] = (result["tokens_processed"] / sum(x["duration_seconds"] for x in steps)) if steps else None
    result["mean_step_seconds"] = statistics.mean(x["duration_seconds"] for x in steps) if steps else None
    result["peak_worker_rss_bytes"] = max((s["process_rss_bytes"] or 0 for s in result["snapshots"]), default=None)
    result["minimum_available_bytes"] = min((s["available_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((s["memory_pressure_free_percent"] for s in result["snapshots"] if s["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((s["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for s in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((s["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((s["disk_free_bytes"] for s in result["snapshots"]), default=None)
    if (run_dir / "worker_result.json").exists():
        result["worker_result"] = json.loads((run_dir / "worker_result.json").read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and result.get("worker_result") and
                        len(steps) == result["worker_result"]["completed_steps"] else
                        "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if sum(x is not None and x is not False for x in (args.plan, args.preflight, args.execute, args.worker)) != 1:
        parser.error("Choose one action")
    if args.plan:
        print(json.dumps(plan(), indent=2))
    elif args.preflight:
        print(json.dumps(preflight("smoke"), indent=2))
    elif args.execute:
        print(json.dumps({k: v for k, v in execute(args.execute).items() if k != "snapshots"}, indent=2))
    else:
        print(json.dumps(worker(args.worker), indent=2))
