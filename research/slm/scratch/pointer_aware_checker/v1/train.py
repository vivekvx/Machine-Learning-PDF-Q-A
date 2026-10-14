"""One preregistered guarded checker experiment; the pointer stays frozen."""
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
from checkpoint import save_checkpoint, load_checkpoint  # noqa: E402
from run import preflight, snapshot, guard_reasons  # noqa: E402
from pointer_qa.answerability.train import selected_pointer  # noqa: E402
from pointer_qa.prepare import sha  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402
from pointer_aware_checker.v1.prepare import config, verify  # noqa: E402
from pointer_aware_checker.v1.model import ContextChecker, FEATURE_WIDTH, FEATURE_NAMES, all_features, loss  # noqa: E402


def rows(partition):
    if partition not in ("train", "calibration", "selection", "final_gate"):
        raise ValueError("Sealed test inaccessible")
    return [json.loads(line) for line in (HERE / f"{partition}.jsonl").read_text().splitlines()]


def pointer_checkpoint():
    selection = json.loads((SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json").read_text())
    path = SCRATCH / "pointer_qa" / "v2" / "runs" / "main" / f"checkpoint-step-{selection['checkpoint_step']:04d}" / "model.safetensors"
    if sha(path) != selection["checkpoint_sha256"]:
        raise ValueError("Frozen pointer weights changed")
    return path


def plan():
    verify()
    path = HERE / "plan.json"
    if path.exists():
        raise FileExistsError("Plan already frozen")
    c = config()
    sets = {name: rows(name) for name in ("train", "calibration", "selection", "final_gate")}
    docs = {name: {r["document_id"] for r in subset} for name, subset in sets.items()}
    if any(docs[a] & docs[b] for a in docs for b in docs if a < b):
        raise ValueError("Document leakage")
    result = {"schema_version": 1,
        "config_sha256": sha(HERE / "config.json"), "manifest_sha256": sha(HERE / "manifest.json"),
        "model_sha256": sha(HERE / "model.py"), "prepare_sha256": sha(HERE / "prepare.py"),
        "train_script_sha256": sha(HERE / "train.py"),
        "pointer_checkpoint_sha256": sha(pointer_checkpoint()),
        "tokenizer_sha256": sha(SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"),
        "split_sha256": {name: sha(HERE / f"{name}.jsonl") for name in sets},
        "split_documents": {name: sorted(items) for name, items in docs.items()},
        "seed": c["seed"], "steps_cap": c["training"]["steps_cap"],
        "wall_seconds_cap": c["training"]["wall_seconds_cap"], "test_sealed": True}
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"plan_sha256": sha(path), "split_documents": {name: len(items) for name, items in docs.items()}}


def load_plan():
    p = json.loads((HERE / "plan.json").read_text())
    fixed = {"config_sha256": HERE / "config.json", "manifest_sha256": HERE / "manifest.json",
        "model_sha256": HERE / "model.py", "prepare_sha256": HERE / "prepare.py",
        "train_script_sha256": HERE / "train.py",
        "tokenizer_sha256": SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"}
    if any(p[key] != sha(path) for key, path in fixed.items()):
        raise ValueError("Plan input changed")
    if p["pointer_checkpoint_sha256"] != sha(pointer_checkpoint()):
        raise ValueError("Pointer changed")
    if any(p["split_sha256"][key] != sha(HERE / f"{key}.jsonl") for key in p["split_sha256"]):
        raise ValueError("Split data changed")
    return p


def feature_cache(pointer, tok, partition, cap, deadline, output):
    scenarios = rows(partition)
    records = []
    with output.open("x") as stream:
        for n, scenario in enumerate(scenarios):
            if time.monotonic() > deadline:
                raise TimeoutError("Feature extraction wall cap")
            pairs, candidates = pair_inputs(tok, scenario["passage"], scenario["question"])
            if max(len(pair["ids"]) for pair in pairs) > cap:
                raise ValueError("Frozen pointer pair context exceeded")
            logits = pointer(pairs)
            mx.eval(logits)
            scores = [float(x) for x in logits.tolist()]
            if not all(math.isfinite(x) for x in scores):
                raise FloatingPointError("Nonfinite pointer logit")
            features = all_features(scenario["passage"], scenario["question"], scores)
            selected = max(range(len(scores)), key=lambda i: scores[i])
            for i, candidate in enumerate(candidates):
                row = {"id": scenario["scenario_id"] + f":{i}",
                    "scenario_id": scenario["scenario_id"], "source_question_id": scenario["source_question_id"],
                    "document_id": scenario["document_id"], "condition": scenario["condition"],
                    "candidate_index": i, "candidate_value": candidate["value"],
                    "candidate_start": candidate["start"], "candidate_end": candidate["end"],
                    "candidate_sentence_start": candidate["sentence_start"],
                    "candidate_sentence_end": candidate["sentence_end"],
                    "label_supported": i in scenario["supported_indices"],
                    "pointer_selected": i == selected, "pointer_logit": scores[i],
                    "features": features[i]}
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                records.append(row)
            if n % 200 == 0:
                print(json.dumps({"partition": partition, "scenarios_encoded": n,
                                  "total": len(scenarios)}), flush=True)
    return records


def fit_scaler(records):
    raw = [r["features"] for r in records]
    means = [statistics.mean(row[i] for row in raw) if i < 8 else 0.0 for i in range(FEATURE_WIDTH)]
    stds = [max(statistics.pstdev(row[i] for row in raw), .01) if i < 8 else 1.0
            for i in range(FEATURE_WIDTH)]
    return {"mean": means, "std": stds, "fit_split": "train_only", "feature_names": FEATURE_NAMES}


def scaled(row, scaler):
    return mx.array([(value - scaler["mean"][i]) / scaler["std"][i]
                     for i, value in enumerate(row["features"])], dtype=mx.float32)


def training_order(records, steps, seed):
    """Balance classes; keep genuine pointer selections and paired candidates."""
    rng = random.Random(seed)
    by_label = {label: [r for r in records if r["label_supported"] == label]
                for label in (True, False)}
    if steps % 2:
        raise ValueError("Class-balanced step cap required")
    order = []
    for label in (True, False):
        selected = [r for r in by_label[label] if r["pointer_selected"]]
        others = [r for r in by_label[label] if not r["pointer_selected"]]
        rng.shuffle(selected)
        rng.shuffle(others)
        n = steps // 2
        # A majority of each class is actual pointer output where available.
        count = min(len(selected), n)
        chosen = selected[:count] + others[:n-count]
        if len(chosen) != n:
            raise ValueError("Not enough distinct training records")
        order.extend(r["id"] for r in chosen)
    rng.shuffle(order)
    return order


def score(model, records, scaler):
    model.eval()
    out = []
    for row in records:
        z = model(scaled(row, scaler))
        mx.eval(z)
        out.append({key: row[key] for key in row if key != "features"} |
                   {"probability": float(mx.sigmoid(z).item())})
    return out


def selected_only(records):
    return [r for r in records if r["pointer_selected"]]


def class_metrics(records, threshold):
    pos = [r for r in records if r["label_supported"]]
    neg = [r for r in records if not r["label_supported"]]
    if not pos or not neg:
        raise ValueError("Both classes required")
    tp = sum(r["probability"] >= threshold for r in pos)
    tn = sum(r["probability"] < threshold for r in neg)
    return {"supported_count": len(pos), "unsupported_count": len(neg),
        "supported_acceptance": tp / len(pos), "unsupported_rejection": tn / len(neg),
        "balanced_accuracy": (tp / len(pos) + tn / len(neg)) / 2,
        "confusion": {"TP": tp, "FN": len(pos)-tp, "TN": tn, "FP": len(neg)-tn}}


def choose_threshold(records, increment):
    choices = []
    for t in [i * increment for i in range(1, round(1 / increment))]:
        metric = class_metrics(records, t)
        key = (min(metric["supported_acceptance"], metric["unsupported_rejection"]),
               metric["balanced_accuracy"], -abs(t-.5), -t)
        choices.append((key, t))
    return max(choices)[1]


def scenario_metrics(records, threshold):
    chosen = selected_only(records)
    by_condition = defaultdict(list)
    for row in chosen:
        by_condition[row["condition"]].append(row)
    unsupported = [r for r in chosen if not r["label_supported"]]
    original = by_condition["original"]
    correction = by_condition["correction"]
    missing = by_condition["missing"]
    obsolete = [r for r in correction if not r["label_supported"]]
    counts = {key: {"count": len(items), "pointer_correct": sum(x["label_supported"] for x in items),
                    "accepted": sum(x["probability"] >= threshold for x in items),
                    "correct_accepted": sum(x["label_supported"] and x["probability"] >= threshold for x in items),
                    "unsupported_accepted": sum(not x["label_supported"] and x["probability"] >= threshold for x in items)}
              for key, items in by_condition.items()}
    return {"pointer_selected": class_metrics(chosen, threshold), "conditions": counts,
        "pointer_only_original_exact": sum(x["label_supported"] for x in original) / len(original),
        "combined_original_exact": sum(x["label_supported"] and x["probability"] >= threshold for x in original) / len(original),
        "correct_answer_retention": (sum(x["label_supported"] and x["probability"] >= threshold for x in original) /
                                     sum(x["label_supported"] for x in original)),
        "missing_fact_abstention": sum(x["probability"] < threshold for x in missing) / len(missing),
        "obsolete_value_rejection": sum(x["probability"] < threshold for x in obsolete) / len(obsolete) if obsolete else None,
        "unsupported_answer_rate": sum(x["probability"] >= threshold for x in unsupported) / len(unsupported),
        "overall_coverage": sum(x["probability"] >= threshold for x in chosen) / len(chosen)}


def provenance():
    p = load_plan()
    return {"plan_sha256": sha(HERE / "plan.json"), "pointer_checkpoint_sha256": p["pointer_checkpoint_sha256"]}


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "pointer-aware-checker-v1":
        raise RuntimeError("Parent resource watchdog required")
    p, c = load_plan(), config()
    settings = c["training"]
    if settings["steps_cap"] > 800 or settings["wall_seconds_cap"] > 180 or settings["batch_size"] != 1:
        raise ValueError("Bounded budget changed")
    started = time.monotonic()
    deadline = started + settings["wall_seconds_cap"]
    mx.metal.reset_peak_memory()
    pointer, tok = selected_pointer()
    before_pointer = sha(pointer_checkpoint())
    datasets = {name: feature_cache(pointer, tok, name, settings["maximum_pair_tokens"], deadline,
                                    run_dir / f"{name}_features.jsonl")
                for name in ("train", "calibration", "selection", "final_gate")}
    scaler = fit_scaler(datasets["train"])
    (run_dir / "scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")
    order = training_order(datasets["train"], settings["steps_cap"], c["seed"])
    (run_dir / "training_order.json").write_text(json.dumps(order, indent=2) + "\n")
    by_id = {r["id"]: r for r in datasets["train"]}
    mx.random.seed(c["seed"])
    model = ContextChecker()
    optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    objective = nn.value_and_grad(model, loss)
    best_key, best_step, stagnant, completed = (-1., -1., -math.inf), None, 0, 0
    curves = []
    with (run_dir / "steps.jsonl").open("x") as steps_file:
        for step, id in enumerate(order, 1):
            if time.monotonic() > deadline:
                raise TimeoutError("Checker wall cap")
            row = by_id[id]
            model.train()
            tick = time.monotonic()
            bce, grads = objective(model, scaled(row, scaler), row["label_supported"])
            mx.eval(bce, grads)
            grads, norm = optim.clip_grad_norm(grads, settings["gradient_clip_l2"])
            mx.eval(grads, norm)
            if not math.isfinite(float(bce.item())) or not math.isfinite(float(norm.item())) or float(norm.item()) <= 0:
                raise FloatingPointError("Nonfinite or zero training signal")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            step_row = {"step": step, "id": id, "label_supported": row["label_supported"],
                "pointer_selected": row["pointer_selected"], "condition": row["condition"],
                "bce": float(bce.item()), "gradient_l2": float(norm.item()),
                "seconds": time.monotonic()-tick, "mlx_peak_bytes": mx.metal.get_peak_memory()}
            steps_file.write(json.dumps(step_row) + "\n")
            steps_file.flush()
            completed = step
            if step % settings["eval_interval"]:
                continue
            cal_scores = score(model, datasets["calibration"], scaler)
            sel_scores = score(model, datasets["selection"], scaler)
            threshold = choose_threshold(selected_only(cal_scores), settings["threshold_grid_step"])
            cal_metrics = scenario_metrics(cal_scores, threshold)
            sel_metrics = scenario_metrics(sel_scores, threshold)
            curve = {"step": step, "threshold": threshold,
                "calibration": cal_metrics, "selection": sel_metrics}
            curves.append(curve)
            with (run_dir / "curve.jsonl").open("a") as stream:
                stream.write(json.dumps(curve) + "\n")
            save_checkpoint(run_dir / f"checkpoint-step-{step:04d}", model, optimizer,
                            step=step, cursor=step,
                            provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
            metric = sel_metrics["pointer_selected"]
            key = (min(metric["supported_acceptance"], metric["unsupported_rejection"]),
                   metric["balanced_accuracy"], -step)
            if key > best_key:
                best_key, best_step, stagnant = key, step, 0
            else:
                stagnant += 1
            print(json.dumps({"step": step, "threshold": threshold,
                              "selection": metric}), flush=True)
            if step >= settings["minimum_steps"] and stagnant >= settings["patience_evaluations"]:
                break
    if best_step is None:
        raise RuntimeError("No checkpoint selected")
    saved_curve = next(x for x in curves if x["step"] == best_step)
    restored = ContextChecker()
    restored_optimizer = optim.AdamW(learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"])
    checkpoint = run_dir / f"checkpoint-step-{best_step:04d}"
    metadata = load_checkpoint(checkpoint, restored, restored_optimizer,
        expected_provenance=provenance() | {"scaler_sha256": sha(run_dir / "scaler.json")})
    saved = mx.load(str(checkpoint / "model.safetensors"))
    actual = dict(tree_flatten(restored.parameters()))
    exact = set(saved) == set(actual) and all(bool(mx.all(saved[k] == actual[k]).item()) for k in saved)
    cal_scores = score(restored, datasets["calibration"], scaler)
    sel_scores = score(restored, datasets["selection"], scaler)
    threshold = choose_threshold(selected_only(cal_scores), settings["threshold_grid_step"])
    if not exact or threshold != saved_curve["threshold"] or scenario_metrics(sel_scores, threshold) != saved_curve["selection"]:
        raise ValueError("Checkpoint restoration changed predictions")
    gate_scores = score(restored, datasets["final_gate"], scaler)
    with (run_dir / "gate_predictions.jsonl").open("x") as out:
        for row in gate_scores:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    gate = scenario_metrics(gate_scores, threshold)
    gold = class_metrics(gate_scores, threshold)
    required = c["gate"]
    selected = gate["pointer_selected"]
    passed = (selected["supported_acceptance"] >= required["supported_acceptance_minimum"] and
        selected["unsupported_rejection"] >= required["unsupported_rejection_minimum"] and
        gate["combined_original_exact"] >= required["original_answerable_exact_minimum"] and
        gate["obsolete_value_rejection"] is not None and gate["obsolete_value_rejection"] >= required["obsolete_value_rejection_minimum"] and
        gate["correct_answer_retention"] >= required["correct_answer_retention_vs_pointer_minimum"])
    if before_pointer != sha(pointer_checkpoint()):
        raise ValueError("Frozen pointer changed during run")
    result = {"status": "validation_gate_passed" if passed else "validation_gate_failed",
        "completed_steps": completed, "selected_step": best_step, "threshold": threshold,
        "calibration": scenario_metrics(cal_scores, threshold),
        "selection": scenario_metrics(sel_scores, threshold),
        "final_gate": gate, "final_gate_all_candidates": gold,
        "parameters": sum(v.size for _, v in tree_flatten(restored.parameters())),
        "checkpoint_restoration_exact": exact, "checkpoint_optimizer_step": metadata["step"],
        "pointer_checkpoint_sha256": before_pointer,
        "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
        "gate_predictions_sha256": sha(run_dir / "gate_predictions.jsonl"),
        "peak_mlx_bytes": mx.metal.get_peak_memory(), "worker_wall_seconds": time.monotonic()-started,
        "test_file_opened": False}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute():
    p, c = load_plan(), config()
    run_dir = HERE / "run"
    if run_dir.exists():
        raise FileExistsError("Versioned run exists")
    before = preflight("smoke")
    run_dir.mkdir()
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
        "preflight": before, "snapshots": [], "plan_sha256": sha(HERE / "plan.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "pointer-aware-checker-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(__file__), "--worker"],
            cwd=SCRATCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            observed = snapshot(process.pid)
            result["snapshots"].append(observed)
            reasons = guard_reasons(observed, before["resources"], before["safety"])
            if time.monotonic()-started > c["training"]["wall_seconds_cap"]:
                reasons.append("Parent wall cap")
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
    result["wall_seconds"] = time.monotonic()-started
    result["after"] = snapshot()
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text().splitlines()] if (run_dir / "steps.jsonl").exists() else []
    result["completed_steps"] = len(steps)
    result["mean_step_seconds"] = statistics.mean(x["seconds"] for x in steps) if steps else None
    result["peak_worker_rss_bytes"] = max((x["process_rss_bytes"] or 0 for x in result["snapshots"]), default=None)
    result["minimum_available_bytes"] = min((x["available_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((x["memory_pressure_free_percent"] for x in result["snapshots"] if x["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((x["swap_used_bytes"]-before["resources"]["swap_used_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((x["swap_out_total_bytes"]-before["resources"]["swap_out_total_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((x["disk_free_bytes"] for x in result["snapshots"]), default=None)
    if (run_dir / "worker_result.json").exists():
        result["worker_result"] = json.loads((run_dir / "worker_result.json").read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and "worker_result" in result else
                        "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "snapshots"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan() if args.plan else execute() if args.execute else worker(HERE / "run"), indent=2))
