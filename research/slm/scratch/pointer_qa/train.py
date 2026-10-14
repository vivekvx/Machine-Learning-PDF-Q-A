"""Frozen-plan, bounded, guarded answerable-only pointer training."""
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
V1 = HERE / "v1"
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from real_text.v3_tokenizer import V3ByteBPE  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402
from pointer_qa.model import PointerReader, parameter_count, pointer_loss, pointer_predict  # noqa: E402
from pointer_qa.prepare import sha, verify  # noqa: E402


def config():
    return json.loads((V1 / "config.json").read_text())


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Only training and validation can be opened for selection")
    return [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]


def tokenizer():
    info = json.loads((V1 / "tokenizer.json").read_text())
    return V3ByteBPE(info["merges"], info["target_size"])


def fixed_ids(subset, count):
    if count % 4:
        raise ValueError("Fixed set must contain complete four-question documents")
    docs = []
    for row in subset:
        if row["document_id"] not in docs:
            docs.append(row["document_id"])
    selected = set(docs[:count // 4])
    result = [row["id"] for row in subset if row["document_id"] in selected]
    if len(result) != count:
        raise ValueError("Fixed set incomplete")
    return result


def plan():
    verify()
    if (V1 / "plan.json").exists():
        raise FileExistsError("Pointer plan already frozen")
    c = config()
    training, validation = rows("train"), rows("validation")
    order = [row["id"] for row in training]
    rng = random.Random(c["seed"])
    rounds = []
    while len(rounds) < c["training"]["main_steps_cap"]:
        rng.shuffle(order)
        rounds.extend(order)
    order = rounds[:c["training"]["main_steps_cap"]]
    tiny = fixed_ids(training, c["training"]["tiny_questions"])
    output = {"schema_version": 1, "config_sha256": sha(V1 / "config.json"),
              "manifest_sha256": sha(V1 / "manifest.json"),
              "train_sha256": sha(V1 / "train.jsonl"),
              "validation_sha256": sha(V1 / "validation.jsonl"),
              "tokenizer_sha256": sha(V1 / "tokenizer.json"),
              "model_sha256": sha(HERE / "model.py"),
              "prepare_sha256": sha(HERE / "prepare.py"),
              "shared_encoder_sha256": sha(SCRATCH / "working_qa" / "v2" / "model.py"),
              "tiny_ids": tiny, "main_order": order,
              "fixed_train_ids": fixed_ids(training, c["training"]["fixed_train_questions"]),
              "fixed_validation_ids": fixed_ids(validation, c["training"]["fixed_validation_questions"]),
              "expected_parameters": parameter_count(PointerReader(c["model"], tokenizer().vocab_size)),
              "selection": c["checkpoint_selection"], "test_sealed": True}
    (V1 / "plan.json").write_text(json.dumps(output, indent=2) + "\n")
    return {key: value for key, value in output.items() if key not in
            ("tiny_ids", "main_order", "fixed_train_ids", "fixed_validation_ids")}


def load_plan():
    p = json.loads((V1 / "plan.json").read_text())
    expected = {"config_sha256": sha(V1 / "config.json"),
                "manifest_sha256": sha(V1 / "manifest.json"),
                "train_sha256": sha(V1 / "train.jsonl"),
                "validation_sha256": sha(V1 / "validation.jsonl"),
                "tokenizer_sha256": sha(V1 / "tokenizer.json"),
                "model_sha256": sha(HERE / "model.py"),
                "prepare_sha256": sha(HERE / "prepare.py"),
                "shared_encoder_sha256": sha(SCRATCH / "working_qa" / "v2" / "model.py")}
    if any(p[key] != value for key, value in expected.items()):
        raise ValueError("Frozen pointer plan input hash mismatch")
    return p


def provenance():
    p = load_plan()
    return {key: p[key] for key in ("config_sha256", "manifest_sha256", "train_sha256",
                                    "validation_sha256", "tokenizer_sha256", "model_sha256",
                                    "prepare_sha256", "shared_encoder_sha256")} | {"plan_sha256": sha(V1 / "plan.json")}


def fresh_model(c, p):
    mx.random.seed(c["seed"])
    model = PointerReader(c["model"], tokenizer().vocab_size)
    mx.eval(model.parameters())
    if parameter_count(model) != p["expected_parameters"]:
        raise ValueError("Parameter count changed")
    return model


def encoded_row(tok, row):
    pairs, candidates = pair_inputs(tok, row["passage"], row["question"])
    if len(pairs) != 4 or len(candidates) != 4:
        raise ValueError("Expected four source span candidates")
    if [(x["start"], x["end"]) for x in candidates] != [(x["start"], x["end"]) for x in row["facts"]]:
        raise ValueError("Candidate offsets differ from source facts")
    if row["passage"][row["answer_start"]:row["answer_end"]] != row["expected_answer"]:
        raise ValueError("Gold answer does not match source")
    cap = config()["training"]["maximum_pair_tokens"]
    if max(len(pair["ids"]) for pair in pairs) > cap:
        raise ValueError("Pair exceeds registered token cap")
    span_cap = config()["training"]["maximum_answer_tokens"]
    if max(pair["value_end"] - pair["value_start"] for pair in pairs) > span_cap:
        raise ValueError("Candidate answer span exceeds registered token cap")
    return pairs, candidates


def assess(model, entries):
    model.eval()
    losses, groups, by_doc = [], defaultdict(lambda: [0, 0]), defaultdict(list)
    for row, pairs, candidates in entries:
        nll = pointer_loss(model, pairs, row["gold_candidate_index"])
        mx.eval(nll)
        prediction = pointer_predict(model, pairs, candidates)
        correct = prediction["prediction"] == row["expected_answer"] and prediction["candidate_index"] == row["gold_candidate_index"]
        copy_ok = row["passage"][prediction["evidence"]["start"]:prediction["evidence"]["end"]] == prediction["prediction"]
        losses.append(float(nll.item()))
        by_doc[row["document_id"]].append(prediction["candidate_index"])
        for group in ("all", "paraphrase" if row["question_paraphrase"] else "direct", row["asked_field"]):
            groups[group][0] += int(correct)
            groups[group][1] += 1
        groups["copy"][0] += int(copy_ok)
        groups["copy"][1] += 1
    changed_docs = sum(len(set(indices)) > 1 for indices in by_doc.values())
    return {"nll": statistics.mean(losses),
            "exact": groups["all"][0] / groups["all"][1],
            "paraphrase_exact": groups["paraphrase"][0] / groups["paraphrase"][1],
            "copy_support": groups["copy"][0] / groups["copy"][1],
            "question_changed_doc_fraction": changed_docs / len(by_doc),
            "groups": {key: {"correct": good, "count": count, "accuracy": good / count}
                       for key, (good, count) in groups.items()}}


def worker(mode, run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "pointer-qa-v1":
        raise RuntimeError("Live resource watchdog required")
    verify()
    c, p = config(), load_plan()
    settings = c["training"]
    if (settings["batch_size"] != 1 or settings["tiny_steps_cap"] > 300 or
        settings["main_steps_cap"] > 2500 or settings["maximum_pointer_attempts"] != 2):
        raise ValueError("Pointer compute cap changed")
    if mode == "main":
        tiny = json.loads((V1 / "runs" / "tiny" / "worker_result.json").read_text())
        if tiny["status"] != "tiny_passed":
            raise RuntimeError("Tiny span-selection and sensitivity checks did not pass")
    train, validation = rows("train"), rows("validation")
    by_id = {row["id"]: row for row in train + validation}
    needed = set(p["tiny_ids"] if mode == "tiny" else p["main_order"] + p["fixed_train_ids"] + p["fixed_validation_ids"])
    tok = tokenizer()
    encoded = {id: encoded_row(tok, by_id[id]) for id in needed}
    fixed_train_ids = p["tiny_ids"] if mode == "tiny" else p["fixed_train_ids"]
    fixed_val_ids = [] if mode == "tiny" else p["fixed_validation_ids"]
    entries = lambda ids: [(by_id[id], *encoded[id]) for id in ids]
    fixed_train, fixed_val = entries(fixed_train_ids), entries(fixed_val_ids)
    order = (p["tiny_ids"] * math.ceil(settings["tiny_steps_cap"] / len(p["tiny_ids"])))[:settings["tiny_steps_cap"]] if mode == "tiny" else p["main_order"]
    model = fresh_model(c, p)
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(model, pointer_loss)
    mx.metal.reset_peak_memory()
    started = time.monotonic()
    initial = {"step": 0, "train": assess(model, fixed_train)}
    if fixed_val:
        initial["validation"] = assess(model, fixed_val)
    best_key, best_step, stagnant, completed = (-1.0, -math.inf, 0), None, 0, 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps(initial) + "\n")
        curves.flush()
        for step, id in enumerate(order, 1):
            if time.monotonic() - started > (settings["tiny_wall_seconds"] if mode == "tiny" else settings["main_wall_seconds"]):
                raise TimeoutError("Worker wall-time ceiling")
            row, (pairs, _) = by_id[id], encoded[id]
            model.train()
            tick = time.monotonic()
            value, gradients = objective(model, pairs, row["gold_candidate_index"])
            mx.eval(value, gradients)
            gradients, norm = optim.clip_grad_norm(gradients, settings["gradient_clip_l2"])
            mx.eval(gradients, norm)
            loss_value, norm_value = float(value.item()), float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value <= 0:
                raise FloatingPointError("Invalid loss or gradient")
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            seconds = time.monotonic() - tick
            tokens = sum(len(pair["ids"]) for pair in pairs)
            steps.write(json.dumps({"step": step, "id": id, "document_id": row["document_id"],
                                    "minibatch_nll": loss_value, "gradient_l2": norm_value,
                                    "tokens": tokens, "seconds": seconds,
                                    "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            interval = settings["tiny_eval_every"] if mode == "tiny" else settings["main_eval_every"]
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
                if (assessment["train"]["exact"] == 1.0 and
                    assessment["train"]["question_changed_doc_fraction"] == 1.0):
                    save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                    step=step, cursor=step, provenance=provenance())
                    best_step = step
                    break
            else:
                save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                                step=step, cursor=step, provenance=provenance())
                val = assessment["validation"]
                key = (val["exact"], -val["nll"], -step)
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
        restored = fresh_model(c, p)
        restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
        path = run_dir / f"checkpoint-step-{best_step:04d}"
        metadata = load_checkpoint(path, restored, restored_optimizer, expected_provenance=provenance())
        saved = mx.load(str(path / "model.safetensors"))
        actual = dict(tree_flatten(restored.parameters()))
        exact = set(saved) == set(actual) and all(bool(mx.all(saved[key] == actual[key]).item()) for key in saved)
        verified = assess(restored, fixed_train if mode == "tiny" else fixed_val)
        original = next(x for x in curve if x["step"] == best_step)["train" if mode == "tiny" else "validation"]
        if not exact or abs(verified["nll"] - original["nll"]) > 1e-5 or verified["exact"] != original["exact"]:
            raise ValueError("Checkpoint restoration failed")
        restoration = {"exact_tensors": len(saved), "nll_difference": verified["nll"] - original["nll"],
                       "optimizer_step": metadata["step"]}
    result = {"status": status, "steps": completed, "best_step": best_step,
              "initial": initial, "final": curve[-1],
              "best": next((x for x in curve if x["step"] == best_step), None),
              "restoration": restoration, "parameter_count": parameter_count(model),
              "peak_mlx_bytes": mx.metal.get_peak_memory(), "worker_wall_seconds": time.monotonic() - started,
              "test_file_opened": False, "provenance": provenance()}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(mode):
    c, p = config(), load_plan()
    run_dir = V1 / "runs" / mode
    if run_dir.exists():
        raise FileExistsError("Versioned run already exists")
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "plan_sha256": sha(V1 / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "pointer-qa-v1"
    limit = c["training"]["tiny_wall_seconds" if mode == "tiny" else "main_wall_seconds"]
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker", mode],
                                   cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            observed = snapshot(process.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic() - started > limit:
                reasons.append("Parent wall-time limit")
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
    file = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    result["completed_steps"] = len(steps)
    result["tokens_processed"] = sum(row["tokens"] for row in steps)
    result["tokens_per_second"] = (result["tokens_processed"] / sum(row["seconds"] for row in steps)) if steps else None
    result["mean_step_seconds"] = statistics.mean(row["seconds"] for row in steps) if steps else None
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
    print(json.dumps(plan() if args.plan else execute(args.execute) if args.execute else worker(args.worker, V1 / "runs" / args.worker), indent=2))
