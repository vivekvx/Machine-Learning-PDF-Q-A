"""Guarded, bounded training of a separate support classifier on frozen-pointer features."""
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
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from pointer_qa.prepare import sha  # noqa: E402
from pointer_qa.answerability.prepare import verify  # noqa: E402
from pointer_qa.answerability.model import SufficiencyHead, binary_loss, frozen_features  # noqa: E402
from pointer_qa.v2.model import AlignedPointer  # noqa: E402
from pointer_qa.v2.train import config as pointer_config, provenance as pointer_provenance, tokenizer  # noqa: E402


def config():
    return json.loads((V1 / "config.json").read_text())


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Sealed test unavailable during training and threshold selection")
    return [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]


def selected_pointer():
    selection = json.loads((HERE.parent / "v2" / "validation" / "pointer_selection.json").read_text())
    step = selection["checkpoint_step"]
    checkpoint = HERE.parent / "v2" / "runs" / "main" / f"checkpoint-step-{step:04d}"
    if sha(checkpoint / "model.safetensors") != selection["checkpoint_sha256"]:
        raise ValueError("Frozen pointer checkpoint changed")
    tok = tokenizer()
    model = AlignedPointer(pointer_config()["model"], tok.vocab_size)
    settings = pointer_config()["training"]
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    load_checkpoint(checkpoint, model, optimizer, expected_provenance=pointer_provenance())
    model.eval()
    return model, tok


def plan():
    verify()
    path = V1 / "plan.json"
    if path.exists():
        raise FileExistsError("Sufficiency plan already frozen")
    c = config()
    train, validation = rows("train"), rows("validation")
    if {r["document_id"] for r in train} & {r["document_id"] for r in validation}:
        raise ValueError("Train/validation document overlap")
    docs = sorted({r["document_id"] for r in validation})
    ncal = c["training"]["calibration_validation_documents"]
    nassess = c["training"]["assessment_validation_documents"]
    if len(docs) != ncal + nassess:
        raise ValueError("Unexpected validation document count")
    order = [r["id"] for r in train]
    random.Random(c["seed"]).shuffle(order)
    order = order[:c["training"]["steps_cap"]]
    plan = {"schema_version": 1, "config_sha256": sha(V1 / "config.json"),
            "dataset_manifest_sha256": sha(V1 / "manifest.json"),
            "train_sha256": sha(V1 / "train.jsonl"),
            "validation_sha256": sha(V1 / "validation.jsonl"),
            "head_model_sha256": sha(HERE / "model.py"),
            "pointer_selection_sha256": sha(HERE.parent / "v2" / "validation" / "pointer_selection.json"),
            "pointer_checkpoint_sha256": json.loads((HERE.parent / "v2" / "validation" / "pointer_selection.json").read_text())["checkpoint_sha256"],
            "tokenizer_sha256": sha(HERE.parent / "v1" / "tokenizer.json"),
            "train_order": order, "calibration_document_ids": docs[:ncal],
            "assessment_document_ids": docs[ncal:], "test_sealed": True}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return {"train_steps": len(order), "calibration_documents": ncal,
            "assessment_documents": nassess, "plan_sha256": sha(path)}


def load_plan():
    p = json.loads((V1 / "plan.json").read_text())
    checks = {"config_sha256": V1 / "config.json", "dataset_manifest_sha256": V1 / "manifest.json",
              "train_sha256": V1 / "train.jsonl", "validation_sha256": V1 / "validation.jsonl",
              "head_model_sha256": HERE / "model.py",
              "pointer_selection_sha256": HERE.parent / "v2" / "validation" / "pointer_selection.json",
              "tokenizer_sha256": HERE.parent / "v1" / "tokenizer.json"}
    if any(p[key] != sha(path) for key, path in checks.items()):
        raise ValueError("Sufficiency plan input hash mismatch")
    return p


def provenance():
    p = load_plan()
    return {key: p[key] for key in ("config_sha256", "dataset_manifest_sha256", "train_sha256",
                                    "validation_sha256", "head_model_sha256", "pointer_selection_sha256",
                                    "pointer_checkpoint_sha256", "tokenizer_sha256")} | {"plan_sha256": sha(V1 / "plan.json")}


def _threshold(calibration, grid):
    ranked = []
    for threshold in grid:
        positive = [r for r in calibration if r["label_supported"]]
        negative = [r for r in calibration if not r["label_supported"]]
        sensitivity = sum(r["probability"] >= threshold for r in positive) / len(positive)
        specificity = sum(r["probability"] < threshold for r in negative) / len(negative)
        ranked.append(((sensitivity + specificity) / 2, -abs(threshold - .5), -threshold, threshold))
    return max(ranked)[-1]


def metrics(records, threshold):
    positive = [r for r in records if r["label_supported"]]
    negative = [r for r in records if not r["label_supported"]]
    if not positive or not negative:
        raise ValueError("Both support classes required")
    tp = sum(r["probability"] >= threshold for r in positive)
    tn = sum(r["probability"] < threshold for r in negative)
    categories = {}
    for name in sorted({r["category"] for r in records}):
        subset = [r for r in records if r["category"] == name]
        categories[name] = {"correct": sum((r["probability"] >= threshold) == r["label_supported"] for r in subset),
                            "count": len(subset)}
        categories[name]["accuracy"] = categories[name]["correct"] / len(subset)
    return {"count": len(records), "supported_count": len(positive), "unsupported_count": len(negative),
            "supported_accuracy": tp / len(positive), "unsupported_rejection": tn / len(negative),
            "balanced_accuracy": .5 * (tp / len(positive) + tn / len(negative)),
            "false_supported": len(negative) - tn, "false_unsupported": len(positive) - tp,
            "confusion": {"TP": tp, "FN": len(positive) - tp, "TN": tn, "FP": len(negative) - tn},
            "by_category": categories}


def assess(head, features, records, p, c):
    head.eval()
    probabilities = []
    losses = []
    for row in records:
        f = features[row["id"]]
        z = head(f)
        loss = binary_loss(head, f, row["label_supported"])
        mx.eval(z, loss)
        probabilities.append({"id": row["id"], "document_id": row["document_id"],
                              "scenario_id": row["scenario_id"], "category": row["category"],
                              "label_supported": row["label_supported"],
                              "probability": float(mx.sigmoid(z).item())})
        losses.append(float(loss.item()))
    calset = set(p["calibration_document_ids"])
    assessmentset = set(p["assessment_document_ids"])
    cal = [r for r in probabilities if r["document_id"] in calset]
    evaluation = [r for r in probabilities if r["document_id"] in assessmentset]
    threshold = _threshold(cal, c["training"]["threshold_grid"])
    return {"bce": statistics.mean(losses), "threshold": threshold,
            "calibration": metrics(cal, threshold),
            "assessment": metrics(evaluation, threshold),
            "all_validation": metrics(probabilities, threshold)}, probabilities


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "pointer-answerability-v1":
        raise RuntimeError("Live resource watchdog required")
    verify()
    p, c = load_plan(), config()
    settings = c["training"]
    if settings["batch_size"] != 1 or settings["steps_cap"] > 1200 or settings["wall_seconds_cap"] > 180:
        raise ValueError("Bounded checker training cap changed")
    train, validation = rows("train"), rows("validation")
    all_rows = train + validation
    pointer, tok = selected_pointer()
    frozen_hash_before = sha(HERE.parent / "v2" / "runs" / "main" /
                             f"checkpoint-step-{json.loads((HERE.parent / 'v2' / 'validation' / 'pointer_selection.json').read_text())['checkpoint_step']:04d}" /
                             "model.safetensors")
    mx.metal.reset_peak_memory()
    started = time.monotonic()
    original = {}
    tokens = {"full": 0, "pair": 0, "max_full": 0}
    for idx, row in enumerate(all_rows):
        if idx % 16 == 0 and time.monotonic() - started > settings["wall_seconds_cap"]:
            raise TimeoutError("Feature extraction exceeded worker wall limit")
        feature, lengths = frozen_features(pointer, tok, row["passage"], row["question"],
                                           row["proposed_candidate_index"], settings["maximum_full_input_tokens"])
        original[row["id"]] = feature
        tokens["full"] += lengths["full_input_tokens"]
        tokens["pair"] += lengths["pair_input_tokens"]
        tokens["max_full"] = max(tokens["max_full"], lengths["full_input_tokens"])
        if idx % 400 == 0:
            print(json.dumps({"features_done": idx, "total": len(all_rows)}), flush=True)
    train_features = mx.array([original[row["id"]] for row in train], dtype=mx.float32)
    mean = mx.mean(train_features, axis=0)
    std = mx.maximum(mx.sqrt(mx.mean((train_features - mean) ** 2, axis=0)), .01)
    mx.eval(mean, std)
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train_only",
              "fit_records": len(train)}
    (run_dir / "scaler.json").write_text(json.dumps(scaler) + "\n")
    features = {id: (mx.array(feature, dtype=mx.float32) - mean) / std
                for id, feature in original.items()}
    del original, train_features
    mx.random.seed(c["seed"])
    head = SufficiencyHead(settings["hidden_dim"])
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(head, binary_loss)
    by_id = {row["id"]: row for row in train}
    initial, _ = assess(head, features, validation, p, c)
    best_key, best_step, stagnant, completed = (-1.0, -1.0, -math.inf, -math.inf), None, 0, 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps({"step": 0, "validation": initial}) + "\n")
        curves.flush()
        for step, id in enumerate(p["train_order"], 1):
            if time.monotonic() - started > settings["wall_seconds_cap"]:
                raise TimeoutError("Worker wall-time ceiling")
            row = by_id[id]
            head.train()
            tick = time.monotonic()
            value, gradient = objective(head, features[id], row["label_supported"])
            mx.eval(value, gradient)
            gradient, norm = optim.clip_grad_norm(gradient, 1.0)
            mx.eval(gradient, norm)
            loss_value, norm_value = float(value.item()), float(norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value <= 0:
                raise FloatingPointError("Invalid checker loss or gradient")
            optimizer.update(head, gradient)
            mx.eval(head.parameters(), optimizer.state)
            duration = time.monotonic() - tick
            steps.write(json.dumps({"step": step, "id": id, "document_id": row["document_id"],
                                    "label_supported": row["label_supported"],
                                    "minibatch_bce": loss_value, "gradient_l2": norm_value,
                                    "seconds": duration, "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            if step % settings["evaluation_interval_steps"]:
                continue
            result, _ = assess(head, features, validation, p, c)
            curves.write(json.dumps({"step": step, "validation": result}) + "\n")
            curves.flush()
            print(json.dumps({"step": step, "threshold": result["threshold"],
                              "assessment": result["assessment"]}), flush=True)
            save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", head, optimizer,
                            step=step, cursor=step, provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
            assessment = result["assessment"]
            key = (assessment["balanced_accuracy"], min(assessment["supported_accuracy"],
                                                        assessment["unsupported_rejection"]), -result["bce"], -step)
            if key > best_key:
                best_key, best_step, stagnant = key, step, 0
            else:
                stagnant += 1
            if step >= settings["minimum_steps_before_early_stop"] and stagnant >= settings["early_stop_patience_evaluations"]:
                break
    selected = next(json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()
                    if json.loads(line)["step"] == best_step)
    restored = SufficiencyHead(settings["hidden_dim"])
    restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    metadata = load_checkpoint(run_dir / f"checkpoint-step-{best_step:04d}", restored, restored_optimizer,
                               expected_provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
    saved = mx.load(str(run_dir / f"checkpoint-step-{best_step:04d}" / "model.safetensors"))
    actual = dict(tree_flatten(restored.parameters()))
    exact = set(saved) == set(actual) and all(bool(mx.all(saved[k] == actual[k]).item()) for k in saved)
    verified, predictions = assess(restored, features, validation, p, c)
    if (not exact or abs(verified["bce"] - selected["validation"]["bce"]) > 1e-5 or
        verified["threshold"] != selected["validation"]["threshold"]):
        raise ValueError("Checker checkpoint restoration mismatch")
    with (run_dir / "validation_predictions.jsonl").open("x") as output:
        for row in predictions:
            output.write(json.dumps(row) + "\n")
    if sha(HERE.parent / "v2" / "runs" / "main" /
           f"checkpoint-step-{json.loads((HERE.parent / 'v2' / 'validation' / 'pointer_selection.json').read_text())['checkpoint_step']:04d}" /
           "model.safetensors") != frozen_hash_before:
        raise ValueError("Frozen pointer weights changed")
    gate = c["validation_gate"]
    assessment = verified["assessment"]
    passed = (assessment["supported_accuracy"] >= gate["supported_accuracy"] and
              assessment["unsupported_rejection"] >= gate["unsupported_rejection"])
    result = {"status": "checker_gate_passed" if passed else "checker_gate_failed", "steps": completed,
              "best_step": best_step, "initial": initial, "selected_validation": verified,
              "parameter_count": sum(v.size for _, v in tree_flatten(restored.parameters())),
              "restoration": {"exact_tensors": len(saved), "optimizer_step": metadata["step"],
                              "bce_difference": verified["bce"] - selected["validation"]["bce"]},
              "feature_tokens": tokens, "peak_mlx_bytes": mx.metal.get_peak_memory(),
              "worker_wall_seconds": time.monotonic() - started, "pointer_checkpoint_sha256": frozen_hash_before,
              "test_file_opened": False, "provenance": provenance(),
              "validation_predictions_sha256": sha(run_dir / "validation_predictions.jsonl")}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute():
    c, p = config(), load_plan()
    run_dir = V1 / "run"
    if run_dir.exists():
        raise FileExistsError("Versioned checker run already exists")
    before = preflight("smoke")
    run_dir.mkdir()
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "plan_sha256": sha(V1 / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "pointer-answerability-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker"],
                                   cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            observed = snapshot(process.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic() - started > c["training"]["wall_seconds_cap"]:
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
            time.sleep(.5)
        result["exit_code"] = process.wait()
    result["wall_seconds"] = time.monotonic() - started
    result["after"] = snapshot()
    steps_path = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()] if steps_path.exists() else []
    result["completed_steps"] = len(steps)
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
    return {k: v for k, v in result.items() if k != "snapshots"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan() if args.plan else execute() if args.execute else worker(V1 / "run"), indent=2))
