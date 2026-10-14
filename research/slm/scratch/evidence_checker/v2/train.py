"""Guarded checker-only MLX training with a disjoint final validation gate."""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
from pointer_qa.answerability.train import selected_pointer  # noqa: E402
from evidence_checker.v2.model import FEATURE_WIDTH, SupportCalibrator, loss, scenario_features  # noqa: E402
from evidence_checker.v2.prepare import config, sha, verify  # noqa: E402


def rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Sealed test cannot be opened for training or selection")
    return [json.loads(line) for line in (HERE / f"{split}.jsonl").read_text().splitlines()]


def _balanced_order(train, steps, seed):
    by_category = defaultdict(list)
    for row in train:
        by_category[row["category"]].append(row["id"])
    positive = ("supported_original", "supported_current_value")
    negative = ("wrong_entity_same_field", "wrong_field_same_entity",
                "obsolete_conflicting_result", "required_fact_absent")
    if steps % 8:
        raise ValueError("Step cap must balance both classes and six source categories")
    npositive, nnegative = steps // 4, steps // 8
    rng = random.Random(seed)
    selected = []
    for category in positive + negative:
        sample = by_category[category][:]
        rng.shuffle(sample)
        count = npositive if category in positive else nnegative
        if len(sample) < count:
            raise ValueError("Insufficient category records for balanced sample")
        selected.extend(sample[:count])
    rng.shuffle(selected)
    if len(selected) != steps or len(set(selected)) != steps:
        raise ValueError("Training order count or uniqueness failed")
    return selected


def plan():
    verify()
    path = HERE / "plan.json"
    if path.exists():
        raise FileExistsError("Checker v2 plan already frozen")
    c = config()
    train, validation = rows("train"), rows("validation")
    train_docs = {r["document_id"] for r in train}
    docs = sorted({r["document_id"] for r in validation})
    if train_docs & set(docs):
        raise ValueError("Document split leakage")
    allocation = c["data"]["validation_document_allocation"]
    a, b, g = (allocation[k] for k in ("threshold_calibration", "checkpoint_selection", "final_gate"))
    if len(docs) != a + b + g:
        raise ValueError("Validation document allocation invalid")
    selection = json.loads((SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json").read_text())
    checkpoint = SCRATCH / "pointer_qa" / "v2" / "runs" / "main" / f"checkpoint-step-{selection['checkpoint_step']:04d}"
    if not selection["pointer_gate_passed"] or sha(checkpoint / "model.safetensors") != selection["checkpoint_sha256"]:
        raise ValueError("Frozen pointer selection or checkpoint changed")
    order = _balanced_order(train, c["training"]["optimizer_steps_cap"], c["seed"])
    result = {"schema_version": 2, "config_sha256": sha(HERE / "config.json"),
              "manifest_sha256": sha(HERE / "manifest.json"),
              "train_sha256": sha(HERE / "train.jsonl"),
              "validation_sha256": sha(HERE / "validation.jsonl"),
              "model_sha256": sha(HERE / "model.py"),
              "prepare_sha256": sha(HERE / "prepare.py"),
              "pointer_selection_sha256": sha(SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json"),
              "pointer_checkpoint_sha256": selection["checkpoint_sha256"],
              "tokenizer_sha256": sha(SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"),
              "train_order": order,
              "calibration_document_ids": docs[:a],
              "selection_document_ids": docs[a:a+b],
              "gate_document_ids": docs[a+b:],
              "test_sealed": True}
    path.write_text(json.dumps(result, indent=2) + "\n")
    category_by_id = {r["id"]: r["category"] for r in train}
    return {"plan_sha256": sha(path), "training_steps_cap": len(order),
            "category_sample_counts": dict(Counter(category_by_id[id] for id in order)),
            "validation_documents": [a, b, g]}


def load_plan():
    p = json.loads((HERE / "plan.json").read_text())
    checks = {"config_sha256": HERE / "config.json", "manifest_sha256": HERE / "manifest.json",
              "train_sha256": HERE / "train.jsonl", "validation_sha256": HERE / "validation.jsonl",
              "model_sha256": HERE / "model.py", "prepare_sha256": HERE / "prepare.py",
              "pointer_selection_sha256": SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json",
              "tokenizer_sha256": SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"}
    if any(p[key] != sha(path) for key, path in checks.items()):
        raise ValueError("Checker v2 plan input hash changed")
    return p


def provenance():
    p = load_plan()
    return {k: p[k] for k in ("config_sha256", "manifest_sha256", "train_sha256",
                              "validation_sha256", "model_sha256", "prepare_sha256",
                              "pointer_selection_sha256", "pointer_checkpoint_sha256",
                              "tokenizer_sha256")} | {"plan_sha256": sha(HERE / "plan.json")}


def class_metrics(records, threshold):
    positive = [r for r in records if r["label_supported"]]
    negative = [r for r in records if not r["label_supported"]]
    if not positive or not negative:
        raise ValueError("Both support classes required")
    tp = sum(r["probability"] >= threshold for r in positive)
    tn = sum(r["probability"] < threshold for r in negative)
    by_category = {}
    for category, items in sorted(group_by(records, "category").items()):
        correct = sum((r["probability"] >= threshold) == r["label_supported"] for r in items)
        by_category[category] = {"count": len(items), "correct": correct,
                                 "accuracy": correct / len(items)}
    return {"count": len(records), "supported_count": len(positive), "unsupported_count": len(negative),
            "supported_acceptance": tp / len(positive), "unsupported_rejection": tn / len(negative),
            "balanced_accuracy": .5 * (tp / len(positive) + tn / len(negative)),
            "confusion": {"TP": tp, "FN": len(positive) - tp,
                          "TN": tn, "FP": len(negative) - tn},
            "by_category": by_category}


def group_by(records, field):
    result = defaultdict(list)
    for row in records:
        result[row[field]].append(row)
    return result


def select_threshold(probabilities, step):
    grid = [i * step for i in range(1, int(round(1 / step)))]
    ranked = []
    for threshold in grid:
        metrics = class_metrics(probabilities, threshold)
        key = (min(metrics["supported_acceptance"], metrics["unsupported_rejection"]),
               metrics["balanced_accuracy"], -abs(threshold - .5), -threshold)
        ranked.append((key, threshold))
    return max(ranked)[1]


def infer_probabilities(model, features, examples):
    model.eval()
    output = []
    losses = []
    for row in examples:
        feature = features[row["id"]]
        z = model(feature)
        bce = loss(model, feature, row["label_supported"])
        mx.eval(z, bce)
        output.append({"id": row["id"], "scenario_id": row["scenario_id"],
                       "document_id": row["document_id"], "category": row["category"],
                       "label_supported": row["label_supported"],
                       "probability": float(mx.sigmoid(z).item())})
        losses.append(float(bce.item()))
    return output, statistics.mean(losses)


def assess_development(model, features, validation, p, c):
    caldocs = set(p["calibration_document_ids"])
    seldocs = set(p["selection_document_ids"])
    development = [r for r in validation if r["document_id"] in caldocs | seldocs]
    probabilities, bce = infer_probabilities(model, features, development)
    cal = [r for r in probabilities if r["document_id"] in caldocs]
    selection = [r for r in probabilities if r["document_id"] in seldocs]
    threshold = select_threshold(cal, c["training"]["threshold_grid_step"])
    return {"threshold": threshold, "bce": bce,
            "calibration": class_metrics(cal, threshold),
            "selection": class_metrics(selection, threshold)}, probabilities


def make_feature_cache(pointer, tok, train, validation, cap, deadline):
    scenarios = {}
    for row in train + validation:
        prior = scenarios.get(row["scenario_id"])
        if prior and (prior["passage"] != row["passage"] or prior["question"] != row["question"] or
                      prior["supported_indices"] != row["supported_indices"]):
            raise ValueError("Scenario context or support labels disagree")
        scenarios[row["scenario_id"]] = row
    output, selected, total_tokens = {}, {}, 0
    for index, (sid, row) in enumerate(scenarios.items()):
        if time.monotonic() > deadline:
            raise TimeoutError("Feature extraction exceeded wall limit")
        features, _, argmax, tokens = scenario_features(pointer, tok, row["passage"], row["question"], cap)
        selected[sid] = argmax
        total_tokens += tokens
        output[sid] = features
        if index % 500 == 0:
            print(json.dumps({"scenarios_encoded": index, "total_scenarios": len(scenarios)}), flush=True)
    by_id = {r["id"]: output[r["scenario_id"]][r["candidate_index"]] for r in train + validation}
    if any(len(v) != FEATURE_WIDTH for v in by_id.values()):
        raise ValueError("Feature width changed")
    return by_id, output, selected, total_tokens


def predicted_scenarios(model, scenario_features, rows, selected, threshold):
    seen, records = set(), []
    for row in rows:
        sid = row["scenario_id"]
        if sid in seen:
            continue
        seen.add(sid)
        index = selected[sid]
        from working_qa.prepare import enumerate_candidates
        candidate = enumerate_candidates(row["passage"])[index]
        z = model(scenario_features[sid][index])
        mx.eval(z)
        probability = float(mx.sigmoid(z).item())
        records.append({"scenario_id": sid, "source_question_id": row["source_question_id"],
                        "document_id": row["document_id"], "scenario_kind": row["scenario_kind"],
                        "candidate_index": index, "candidate_value": candidate["value"],
                        "candidate_start": candidate["start"],
                        "candidate_end": candidate["end"],
                        "label_supported": index in row["supported_indices"],
                        "category": row["scenario_kind"], "probability": probability,
                        "accepted": probability >= threshold})
    return records


def end_to_end(predicted, threshold):
    groups = group_by(predicted, "scenario_kind")
    original = groups["original"]
    correction = groups["correction"]
    missing = [r for key, group in groups.items() if key.startswith("missing-") for r in group]
    if not original or not correction or not missing:
        raise ValueError("Missing an evaluation condition")
    unsupported = [r for r in predicted if not r["label_supported"]]
    return {"answerable_original_count": len(original),
            "pointer_only_original_exact": sum(r["label_supported"] for r in original) / len(original),
            "combined_answerable_exact_match": sum(r["label_supported"] and r["accepted"] for r in original) / len(original),
            "answerable_coverage": sum(r["accepted"] for r in original) / len(original),
            "correction_current_value_exact_match": sum(r["label_supported"] and r["accepted"] for r in correction) / len(correction),
            "missing_fact_count": len(missing),
            "missing_fact_abstention_accuracy": sum(not r["accepted"] for r in missing) / len(missing),
            "unsupported_candidate_count": len(unsupported),
            "unsupported_answer_rate": sum(r["accepted"] for r in unsupported) / len(unsupported),
            "overall_coverage": sum(r["accepted"] for r in predicted) / len(predicted),
            "predicted_support_classification": class_metrics(predicted, threshold)}


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "evidence-checker-v2":
        raise RuntimeError("Live resource watchdog required")
    verify()
    p, c = load_plan(), config()
    settings = c["training"]
    if settings["batch_size"] != 1 or settings["optimizer_steps_cap"] > 1200 or settings["wall_seconds_cap"] > 120:
        raise ValueError("Checker training cap changed")
    train, validation = rows("train"), rows("validation")
    gate_docs = set(p["gate_document_ids"])
    if gate_docs & ({r["document_id"] for r in train} | set(p["calibration_document_ids"]) |
                    set(p["selection_document_ids"])):
        raise ValueError("Final gate document leakage")
    pointer, tok = selected_pointer()
    selection = json.loads((SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json").read_text())
    pointer_checkpoint = SCRATCH / "pointer_qa" / "v2" / "runs" / "main" / f"checkpoint-step-{selection['checkpoint_step']:04d}" / "model.safetensors"
    if sha(pointer_checkpoint) != p["pointer_checkpoint_sha256"]:
        raise ValueError("Frozen pointer checkpoint changed before training")
    started = time.monotonic()
    mx.metal.reset_peak_memory()
    raw, scenario_raw, selected, feature_tokens = make_feature_cache(pointer, tok, train, validation,
                                                                      settings["maximum_pair_tokens"], started + settings["wall_seconds_cap"])
    means = [statistics.mean(raw[row["id"]][i] for row in train) for i in range(FEATURE_WIDTH)]
    stds = [max(statistics.pstdev(raw[row["id"]][i] for row in train), .01) for i in range(FEATURE_WIDTH)]
    scaler = {"fit_split": "train_only", "mean": means, "std": stds,
              "feature_width": FEATURE_WIDTH, "fit_records": len(train)}
    (run_dir / "scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")
    features = {id: mx.array([(value - means[i]) / stds[i] for i, value in enumerate(values)], dtype=mx.float32)
                for id, values in raw.items()}
    scenario_scaled = {sid: [mx.array([(value - means[i]) / stds[i] for i, value in enumerate(candidate)],
                                      dtype=mx.float32) for candidate in candidates]
                       for sid, candidates in scenario_raw.items()}
    del raw, scenario_raw
    mx.random.seed(c["seed"])
    head = SupportCalibrator()
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(head, loss)
    by_id = {r["id"]: r for r in train}
    initial, _ = assess_development(head, features, validation, p, c)
    best_key, best_step, stagnant, completed = (-1.0, -1.0, -math.inf, -math.inf), None, 0, 0
    with (run_dir / "curve.jsonl").open("x") as curves, (run_dir / "steps.jsonl").open("x") as steps:
        curves.write(json.dumps({"step": 0, "development": initial}) + "\n")
        curves.flush()
        for step, id in enumerate(p["train_order"], 1):
            if time.monotonic() - started > settings["wall_seconds_cap"]:
                raise TimeoutError("Checker worker wall limit")
            row = by_id[id]
            head.train()
            tick = time.monotonic()
            bce, gradient = objective(head, features[id], row["label_supported"])
            mx.eval(bce, gradient)
            gradient, gradnorm = optim.clip_grad_norm(gradient, 1.0)
            mx.eval(gradient, gradnorm)
            value, norm = float(bce.item()), float(gradnorm.item())
            if not math.isfinite(value) or not math.isfinite(norm) or norm <= 0:
                raise FloatingPointError("Nonfinite or zero checker training signal")
            optimizer.update(head, gradient)
            mx.eval(head.parameters(), optimizer.state)
            steps.write(json.dumps({"step": step, "id": id, "category": row["category"],
                                    "label_supported": row["label_supported"],
                                    "minibatch_bce": value, "gradient_l2": norm,
                                    "seconds": time.monotonic() - tick,
                                    "mlx_active_bytes": mx.metal.get_active_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}) + "\n")
            steps.flush()
            completed = step
            if step % settings["evaluation_interval_steps"]:
                continue
            development, _ = assess_development(head, features, validation, p, c)
            curves.write(json.dumps({"step": step, "development": development}) + "\n")
            curves.flush()
            print(json.dumps({"step": step, "threshold": development["threshold"],
                              "selection": development["selection"]}), flush=True)
            save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", head, optimizer,
                            step=step, cursor=step,
                            provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
            sel = development["selection"]
            key = (min(sel["supported_acceptance"], sel["unsupported_rejection"]),
                   sel["balanced_accuracy"], -development["bce"], -step)
            if key > best_key:
                best_key, best_step, stagnant = key, step, 0
            else:
                stagnant += 1
            if step >= settings["minimum_steps_before_early_stop"] and stagnant >= settings["early_stop_patience_evaluations"]:
                break
    if best_step is None:
        raise RuntimeError("No checkpoint was selected")
    curve = [json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()]
    selected_development = next(item["development"] for item in curve if item["step"] == best_step)
    restored = SupportCalibrator()
    restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    checkpoint = run_dir / f"checkpoint-step-{best_step:04d}"
    metadata = load_checkpoint(checkpoint, restored, restored_optimizer,
                               expected_provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
    saved = mx.load(str(checkpoint / "model.safetensors"))
    actual = dict(tree_flatten(restored.parameters()))
    exact = set(saved) == set(actual) and all(bool(mx.all(saved[k] == actual[k]).item()) for k in saved)
    verified_development, _ = assess_development(restored, features, validation, p, c)
    if (not exact or verified_development["threshold"] != selected_development["threshold"] or
        abs(verified_development["bce"] - selected_development["bce"]) > 1e-6):
        raise ValueError("Selected checker checkpoint did not restore exactly")
    threshold = verified_development["threshold"]
    gate_rows = [r for r in validation if r["document_id"] in gate_docs]
    gold_predictions, gate_bce = infer_probabilities(restored, features, gate_rows)
    gate = class_metrics(gold_predictions, threshold)
    with (run_dir / "gate_gold_predictions.jsonl").open("x") as output:
        for row in gold_predictions:
            output.write(json.dumps(row) + "\n")
    predicted = predicted_scenarios(restored, scenario_scaled, gate_rows, selected, threshold)
    with (run_dir / "gate_pointer_predictions.jsonl").open("x") as output:
        for row in predicted:
            output.write(json.dumps(row) + "\n")
    end_to_end_result = end_to_end(predicted, threshold)
    required = c["final_validation_gate"]
    passed = (gate["supported_acceptance"] >= required["supported_acceptance"] and
              gate["unsupported_rejection"] >= required["unsupported_rejection"])
    if sha(pointer_checkpoint) != p["pointer_checkpoint_sha256"]:
        raise ValueError("Frozen pointer checkpoint changed during checker training")
    result = {"status": "checker_gate_passed" if passed else "checker_gate_failed",
              "steps": completed, "best_step": best_step, "threshold": threshold,
              "initial": initial, "selected_development": verified_development,
              "final_gate_gold": gate, "final_gate_bce": gate_bce,
              "final_gate_pointer_selected": end_to_end_result,
              "parameter_count": sum(v.size for _, v in tree_flatten(restored.parameters())),
              "restoration": {"exact_tensors": len(saved), "optimizer_step": metadata["step"],
                              "bce_difference": verified_development["bce"] - selected_development["bce"]},
              "feature_pair_tokens": feature_tokens,
              "peak_mlx_bytes": mx.metal.get_peak_memory(), "worker_wall_seconds": time.monotonic() - started,
              "pointer_checkpoint_sha256": p["pointer_checkpoint_sha256"],
              "test_file_opened": False,
              "gate_gold_predictions_sha256": sha(run_dir / "gate_gold_predictions.jsonl"),
              "gate_pointer_predictions_sha256": sha(run_dir / "gate_pointer_predictions.jsonl"),
              "provenance": provenance()}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    if passed:
        selection = {"checker_gate_passed": True, "checkpoint_step": best_step,
                     "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
                     "threshold": threshold, "scaler_sha256": sha(run_dir / "scaler.json"),
                     "pointer_checkpoint_sha256": p["pointer_checkpoint_sha256"],
                     "plan_sha256": sha(HERE / "plan.json"),
                     "gate_gold_predictions_sha256": result["gate_gold_predictions_sha256"],
                     "gate_pointer_predictions_sha256": result["gate_pointer_predictions_sha256"],
                     "test_sealed": True}
        (HERE / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return result


def execute():
    c, p = config(), load_plan()
    run_dir = HERE / "run"
    if run_dir.exists():
        raise FileExistsError("Versioned checker v2 run already exists")
    before = preflight("smoke")
    run_dir.mkdir()
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "plan_sha256": sha(HERE / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "evidence-checker-v2"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker"],
                                   cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            observed = snapshot(process.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic() - started > c["training"]["wall_seconds_cap"]:
                reasons.append("Parent wall-time cap")
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
    print(json.dumps(plan() if args.plan else execute() if args.execute else worker(HERE / "run"), indent=2))
