"""Frozen-data, answer-masked SLM-A synthetic QA experiment.

The train command uses only train/validation. The test partition is read only
by the separate final evaluation command, after checkpoint selection.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
sys.path[:0] = [str(HERE), str(SCRATCH)]
from prepare import V1, V3ByteBPE, answer_sequence, digest, prompt, verify  # noqa: E402
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Test cannot be opened during training")
    return [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]


def tokenizer():
    info = json.loads((V1 / "tokenizer.json").read_text())
    return V3ByteBPE(info["merges"], info["target_size"])


def make_plan():
    verify()
    config = json.loads((V1 / "config.json").read_text())
    train, validation = rows("train"), rows("validation")
    rng = random.Random(config["seed"])
    order = list(range(len(train)))
    rng.shuffle(order)
    def balanced(items, count):
        each = count // 2
        selected = [r["id"] for r in items if r["answerable"]][:each]
        selected += [r["id"] for r in items if not r["answerable"]][:each]
        assert len(selected) == count
        return selected
    plan = {"schema_version": 1, "config_sha256": digest(V1 / "config.json"),
            "manifest_sha256": digest(V1 / "manifest.json"),
            "sample_order": [train[i]["id"] for i in order[:config["training"]["maximum_optimizer_steps"]]],
            "fixed_train_ids": balanced(train, config["training"]["fixed_train_examples"]),
            "fixed_validation_ids": balanced(validation, config["training"]["fixed_validation_examples"]),
            "selection": "lowest fixed validation answer-only CE, earliest tie",
            "test_policy": "sealed until checkpoint selection"}
    path = V1 / "plan.json"
    with path.open("x") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")
    return plan


def load_plan():
    plan = json.loads((V1 / "plan.json").read_text())
    if plan["config_sha256"] != digest(V1 / "config.json") or plan["manifest_sha256"] != digest(V1 / "manifest.json"):
        raise ValueError("Plan inputs changed")
    return plan


def masked_loss(model, inputs, targets, mask):
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(inputs)
    per_token = nn.losses.cross_entropy(logits, targets, reduction="none")
    return mx.sum(per_token * mask) / mx.sum(mask)


def model_and_optimizer(config, *, fresh_seed=True):
    import mlx.core as mx
    import mlx.optimizers as optim
    from model import ScratchTransformer, parameter_count
    if fresh_seed:
        mx.random.seed(config["seed"])
    model = ScratchTransformer(config["model"], tokenizer().vocab_size)
    optimizer = optim.AdamW(learning_rate=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    mx.eval(model.parameters())
    if parameter_count(model) != config["parameter_count_expected"]:
        raise ValueError("Model size differs from preregistered size")
    return model, optimizer


def tensors(tok, row):
    import mlx.core as mx
    ids, mask, _, _ = answer_sequence(tok, row)
    return (mx.array([ids[:-1]], dtype=mx.int32),
            mx.array([ids[1:]], dtype=mx.int32),
            mx.array([mask], dtype=mx.float32))


def evaluate_loss(model, tok, fixed):
    import mlx.core as mx
    values = []
    model.eval()
    for row in fixed:
        value = masked_loss(model, *tensors(tok, row))
        mx.eval(value)
        values.append(float(value.item()))
    loss = sum(values) / len(values)
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite evaluation")
    return {"loss": loss, "perplexity": math.exp(loss), "examples": len(values)}


def train_worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "synthetic-qa-v1":
        raise RuntimeError("Live resource watchdog required")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from model import parameter_count
    config = json.loads((V1 / "config.json").read_text())
    settings = config["training"]
    if settings["maximum_optimizer_steps"] > 800 or settings["wall_time_cap_seconds"] > 180 or settings["batch_size"] != 1:
        raise ValueError("Hard QA training cap exceeded")
    verify()
    plan = load_plan()
    train, val = rows("train"), rows("validation")
    by_id = {r["id"]: r for r in train + val}
    tok = tokenizer()
    model, optimizer = model_and_optimizer(config)
    mx.metal.reset_peak_memory()
    fixed_train = [by_id[i] for i in plan["fixed_train_ids"]]
    fixed_val = [by_id[i] for i in plan["fixed_validation_ids"]]
    started = time.monotonic()
    curve = [{"step": 0, "train": evaluate_loss(model, tok, fixed_train),
              "validation": evaluate_loss(model, tok, fixed_val)}]
    objective = nn.value_and_grad(model, masked_loss)
    provenance = {"config_sha256": digest(V1 / "config.json"),
                  "manifest_sha256": digest(V1 / "manifest.json"),
                  "plan_sha256": digest(V1 / "plan.json"),
                  "model_sha256": digest(SCRATCH / "model.py"),
                  "train_sha256": digest(V1 / "train.jsonl"),
                  "validation_sha256": digest(V1 / "validation.jsonl"),
                  "tokenizer_sha256": digest(V1 / "tokenizer.json")}
    best_step, best_loss, stagnant = 0, curve[0]["validation"]["loss"], 0
    checkpoint_steps = []
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps(curve[0]) + "\n")
        for step, row_id in enumerate(plan["sample_order"], 1):
            if time.monotonic() - started >= settings["wall_time_cap_seconds"]:
                raise TimeoutError("QA worker wall-time cap")
            x, y, mask = tensors(tok, by_id[row_id])
            model.train()
            tick = time.monotonic()
            loss, grads = objective(model, x, y, mask)
            mx.eval(loss, grads)
            grad_sq = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(grads))
            if not math.isfinite(float(loss.item())) or not math.isfinite(grad_sq) or grad_sq <= 0:
                raise FloatingPointError("Invalid training loss or gradient")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            param_sq = sum(float(mx.sum(v.astype(mx.float32) ** 2).item()) for _, v in tree_flatten(model.parameters()))
            if not math.isfinite(param_sq):
                raise FloatingPointError("Nonfinite parameters")
            duration = time.monotonic() - tick
            token_count = int(x.shape[1])
            record = {"step": step, "row_id": row_id, "minibatch_loss": float(loss.item()),
                      "gradient_l2_norm": math.sqrt(grad_sq), "parameter_l2_norm": math.sqrt(param_sq),
                      "tokens": token_count, "duration_seconds": duration,
                      "tokens_per_second": token_count / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            steps.write(json.dumps(record) + "\n")
            steps.flush()
            if step % settings["evaluation_interval_steps"] == 0:
                assessment = {"step": step, "train": evaluate_loss(model, tok, fixed_train),
                              "validation": evaluate_loss(model, tok, fixed_val)}
                curve.append(assessment)
                curves.write(json.dumps(assessment) + "\n")
                curves.flush()
                os.fsync(curves.fileno())
                print(json.dumps({"step": step, "train_loss": assessment["train"]["loss"],
                                  "validation_loss": assessment["validation"]["loss"]}), flush=True)
                if assessment["train"]["loss"] > 2 * curve[0]["train"]["loss"] or assessment["validation"]["loss"] > curve[0]["validation"]["loss"] + 0.5:
                    raise RuntimeError("Fixed-set divergence abort")
                if step % settings["checkpoint_interval_steps"] == 0:
                    location = run_dir / f"checkpoint-step-{step:04d}"
                    save_checkpoint(location, model, optimizer, step=step, cursor=step, provenance=provenance)
                    checkpoint_steps.append(step)
                if assessment["validation"]["loss"] < best_loss - 1e-9:
                    best_loss, best_step, stagnant = assessment["validation"]["loss"], step, 0
                else:
                    stagnant += 1
                if step >= settings["minimum_steps_before_early_stop"] and stagnant >= settings["early_stop_patience_evaluations"]:
                    break
    if best_step == 0:
        raise RuntimeError("No validation checkpoint improved on initialization")
    location = run_dir / f"checkpoint-step-{best_step:04d}"
    restored, restored_optimizer = model_and_optimizer(config, fresh_seed=False)
    metadata = load_checkpoint(location, restored, restored_optimizer, expected_provenance=provenance)
    saved = mx.load(str(location / "model.safetensors"))
    actual = dict(tree_flatten(restored.parameters()))
    if set(saved) != set(actual) or any(not bool(mx.all(saved[k] == actual[k]).item()) for k in saved):
        raise ValueError("Checkpoint tensor restoration mismatch")
    restored_val = evaluate_loss(restored, tok, fixed_val)
    if abs(restored_val["loss"] - best_loss) > 1e-5:
        raise ValueError("Checkpoint validation loss restoration mismatch")
    result = {"status": "completed", "completed_steps": len(plan["sample_order"]) if len(curve) == 1 else curve[-1]["step"],
              "best_step": best_step, "best_validation": curve[[x["step"] for x in curve].index(best_step)]["validation"],
              "initial": curve[0], "final": curve[-1], "checkpoint_steps": checkpoint_steps,
              "restoration": {"exact_model_tensors": len(saved), "validation_loss_difference": abs(restored_val["loss"] - best_loss),
                              "optimizer_step": metadata["step"]},
              "peak_mlx_bytes": mx.metal.get_peak_memory(), "parameter_count": parameter_count(model),
              "test_file_opened": False, "provenance": provenance, "worker_wall_seconds": time.monotonic() - started}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Run already exists")
    verify()
    load_plan()
    config = json.loads((V1 / "config.json").read_text())
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running", "preflight": before,
              "snapshots": [], "config_sha256": digest(V1 / "config.json"), "plan_sha256": digest(V1 / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    environment = os.environ.copy()
    environment["SLMA_SAFETY_PARENT"] = "synthetic-qa-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", str(run_dir)],
                                cwd=SCRATCH, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            snap = snapshot(proc.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, before["resources"], before["safety"])
            if time.monotonic() - started > config["training"]["wall_time_cap_seconds"]:
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
    step_path = run_dir / "steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text().splitlines()] if step_path.exists() else []
    result["completed_steps"] = len(records)
    result["target_tokens_processed"] = sum(x["tokens"] for x in records)
    result["tokens_per_second"] = (sum(x["tokens"] for x in records) / sum(x["duration_seconds"] for x in records)) if records else None
    result["mean_step_seconds"] = (sum(x["duration_seconds"] for x in records) / len(records)) if records else None
    result["peak_worker_rss_bytes"] = max((x["process_rss_bytes"] or 0 for x in result["snapshots"]), default=0)
    result["minimum_available_bytes"] = min((x["available_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((x["memory_pressure_free_percent"] for x in result["snapshots"] if x["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((x["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((x["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((x["disk_free_bytes"] for x in result["snapshots"]), default=None)
    if (run_dir / "worker_result.json").exists():
        result["worker_result"] = json.loads((run_dir / "worker_result.json").read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and result.get("worker_result") and len(records) == result["worker_result"]["completed_steps"]
                        else "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
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
        parser.error("Choose exactly one action")
    if args.plan:
        print(json.dumps(make_plan(), indent=2))
    elif args.preflight:
        print(json.dumps(preflight("smoke"), indent=2))
    elif args.execute:
        print(json.dumps({k: v for k, v in execute(args.execute).items() if k != "snapshots"}, indent=2))
    else:
        print(json.dumps(train_worker(args.worker), indent=2))
