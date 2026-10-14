"""Fresh-weight, bounded v3 CPython pilot. Launch only through v3_pilot."""
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent
SCRATCH = ROOT.parent
sys.path[:0] = [str(ROOT), str(SCRATCH)]
from checkpoint import digest, load_checkpoint  # noqa: E402
from model import ScratchTransformer, next_token_loss, parameter_count  # noqa: E402
from expanded_worker import checkpoint_at, probe_values, generate  # noqa: E402
from v3_data import V3, load_verified, load_plan  # noqa: E402


def stats(tree):
    leaves = [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]
    if not leaves:
        raise ValueError("No tensor leaves")
    squared = 0.0
    max_abs = 0.0
    for value in leaves:
        mx.eval(value)
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise FloatingPointError("Non-finite tensor")
        squared += float(mx.sum(mx.square(value.astype(mx.float32))).item())
        max_abs = max(max_abs, float(mx.max(mx.abs(value)).item()))
    return {"l2_norm": math.sqrt(squared), "max_absolute": max_abs, "tensor_count": len(leaves)}


def evaluate(model, fixed):
    by_doc = defaultdict(list)
    values = []
    for window in fixed:
        ids = window["ids"]
        loss = next_token_loss(model, mx.array([ids[:-1]], dtype=mx.int32), mx.array([ids[1:]], dtype=mx.int32))
        mx.eval(loss)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite fixed-window loss")
        values.append(value)
        by_doc[window["document_id"]].append(value)
    per_doc = {doc: sum(items) / len(items) for doc, items in by_doc.items()}
    balanced = sum(per_doc.values()) / len(per_doc)
    window_mean = sum(values) / len(values)
    return {"loss": window_mean, "perplexity": math.exp(window_mean),
            "document_balanced_loss": balanced, "document_balanced_perplexity": math.exp(balanced),
            "windows": len(values), "documents": len(per_doc), "per_document_loss": per_doc}


def restore_exact(path, model_config, tokenizer, config, provenance, fixed_train, fixed_val, curve):
    model = ScratchTransformer(model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    metadata = load_checkpoint(path, model, optimizer, expected_provenance=provenance)
    saved_model = mx.load(str(path / "model.safetensors"))
    saved_optimizer = mx.load(str(path / "adamw.safetensors"))
    actual_model = dict(tree_flatten(model.parameters()))
    actual_optimizer = dict(tree_flatten(optimizer.state))
    if set(actual_model) != set(saved_model) or set(actual_optimizer) != set(saved_optimizer):
        raise ValueError("Restored checkpoint tensor keys differ")
    for key, value in saved_model.items():
        if not bool(mx.all(actual_model[key] == value).item()):
            raise ValueError("Model tensor restoration differs")
    for key, value in saved_optimizer.items():
        if not bool(mx.all(actual_optimizer[key] == value).item()):
            raise ValueError("Optimizer tensor restoration differs")
    if int(actual_optimizer["step"].item()) != metadata["step"]:
        raise ValueError("Restored optimizer step differs")
    model.eval()
    train_result, val_result = evaluate(model, fixed_train), evaluate(model, fixed_val)
    if abs(train_result["loss"] - curve["train"]["loss"]) > 1e-5 or abs(val_result["loss"] - curve["validation"]["loss"]) > 1e-5:
        raise ValueError("Restored fixed-window loss differs")
    probe = json.loads((path / "probe_logits.json").read_text())["values"]
    observed = probe_values(model, fixed_val[0])
    probe_error = max(abs(a - b) for a, b in zip(probe, observed))
    if probe_error > 1e-6:
        raise ValueError("Restored probe differs")
    return model, {"step": metadata["step"], "exact_model_tensors": len(saved_model),
                   "exact_optimizer_tensors": len(saved_optimizer), "probe_max_abs_difference": probe_error,
                   "model_sha256": digest(path / "model.safetensors"),
                   "optimizer_sha256": digest(path / "adamw.safetensors")}


def worker(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "cpython-v3":
        raise RuntimeError("Live memory watchdog required")
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    plan = load_plan(config, train_windows, val_windows)
    if config["optimizer_steps_cap"] > 100 or config["wall_time_cap_seconds"] > 180 or config["batch_size"] != 1:
        raise ValueError("v3 hard safety cap exceeded")
    base_model = json.loads((SCRATCH / "config.v1.json").read_text())["model"]
    model_config = {**base_model, **config["architecture"]}
    if any(model_config[k] != base_model[k] for k in ("layers", "embedding_dim", "attention_heads", "context_length", "feed_forward_multiplier", "normalization_epsilon")):
        raise ValueError("Architecture dimensions changed")
    mx.random.seed(config["seed"])
    model = ScratchTransformer(model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    mx.eval(model.parameters())
    if parameter_count(model) != config["parameter_count_expected"]:
        raise ValueError("Unexpected parameter count")
    model.eval()
    initial = {"step": 0, "train": evaluate(model, plan["fixed_train"]),
               "validation": evaluate(model, plan["fixed_validation"])}
    model.train()
    mx.metal.reset_peak_memory()
    objective = nn.value_and_grad(model, next_token_loss)
    provenance = {"pilot_config_sha256": digest(V3 / "pilot_config.v1.json"),
                  "pilot_plan_sha256": digest(V3 / "pilot_plan.v1.json"),
                  "corpus_manifest_sha256": digest(V3 / "manifest.json"),
                  "train_sha256": digest(V3 / "train.jsonl"), "validation_sha256": digest(V3 / "validation.jsonl"),
                  "tokenizer_sha256": digest(V3 / "tokenizer.json"),
                  "model_source_sha256": digest(SCRATCH / "model.py"), "worker_source_sha256": digest(__file__)}
    curve = [initial]
    checkpoints = {}
    started = time.monotonic()
    with (run_dir / "steps.jsonl").open("x") as step_file, (run_dir / "curve.jsonl").open("x") as curve_file:
        curve_file.write(json.dumps(initial, sort_keys=True) + "\n")
        curve_file.flush()
        for step, window in enumerate(plan["training_sample_order"], 1):
            if time.monotonic() - started >= config["wall_time_cap_seconds"]:
                raise TimeoutError("Worker wall-time cap")
            ids = window["ids"]
            x = mx.array([ids[:-1]], dtype=mx.int32)
            y = mx.array([ids[1:]], dtype=mx.int32)
            tick = time.monotonic()
            loss, grads = objective(model, x, y)
            mx.eval(loss, grads)
            grad = stats(grads)
            if not bool(mx.isfinite(loss).item()) or grad["l2_norm"] <= 0:
                raise FloatingPointError("Invalid loss/gradient")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            param = stats(model.parameters())
            stats(optimizer.state)
            duration = time.monotonic() - tick
            record = {"step": step, "document_id": window["document_id"], "group_id": window["group_id"],
                      "window_offset": window["offset"], "tokens": config["sequence_length"],
                      "minibatch_loss_before_update": float(loss.item()),
                      "gradient_l2_norm": grad["l2_norm"], "gradient_max_absolute": grad["max_absolute"],
                      "parameter_l2_norm": param["l2_norm"], "parameter_max_absolute": param["max_absolute"],
                      "duration_seconds": duration, "tokens_per_second": config["sequence_length"] / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(), "mlx_cache_bytes": mx.metal.get_cache_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            step_file.write(json.dumps(record, sort_keys=True) + "\n")
            step_file.flush()
            if step in config["evaluation_steps"]:
                model.eval()
                assessment = {"step": step, "train": evaluate(model, plan["fixed_train"]),
                              "validation": evaluate(model, plan["fixed_validation"])}
                model.train()
                curve.append(assessment)
                curve_file.write(json.dumps(assessment, sort_keys=True) + "\n")
                curve_file.flush()
                os.fsync(curve_file.fileno())
                print(json.dumps({"step": step, "fixed_train_loss": assessment["train"]["loss"],
                                  "validation_document_balanced_loss": assessment["validation"]["document_balanced_loss"]}), flush=True)
                if assessment["train"]["loss"] > 2 * initial["train"]["loss"] or assessment["validation"]["document_balanced_loss"] > initial["validation"]["document_balanced_loss"] + 0.5:
                    raise RuntimeError("Preregistered fixed-window divergence abort")
                if step in config["checkpoint_steps"]:
                    checkpoints[step], _ = checkpoint_at(run_dir, step, model, optimizer,
                                                         plan["fixed_validation"][0], provenance, step)
    if len(curve) != len(config["evaluation_steps"]) or len(checkpoints) != len(config["checkpoint_steps"]):
        raise RuntimeError("Pilot did not complete preregistered step budget")
    best = min(curve[1:], key=lambda row: row["validation"]["document_balanced_loss"])
    best_model, best_restore = restore_exact(checkpoints[best["step"]], model_config, tokenizer, config,
                                             provenance, plan["fixed_train"], plan["fixed_validation"], best)
    final_model, final_restore = restore_exact(checkpoints[config["optimizer_steps_cap"]], model_config, tokenizer,
                                               config, provenance, plan["fixed_train"], plan["fixed_validation"], curve[-1])
    in_memory_error = max(abs(a - b) for a, b in zip(probe_values(model, plan["fixed_validation"][0]),
                                                     probe_values(final_model, plan["fixed_validation"][0])))
    if in_memory_error > 1e-6:
        raise ValueError("Final checkpoint differs from in-memory model")
    generations = {"policy": "First 100 characters of fixed train/validation page, 40-token prompt, greedy 24-token continuation",
                   "best": [generate(best_model, tokenizer, row, 24) for row in (train[0], val[0])],
                   "final": [generate(final_model, tokenizer, row, 24) for row in (train[0], val[0])]}
    (run_dir / "generations.json").write_text(json.dumps(generations, indent=2) + "\n")
    result = {"status": "completed", "completed_steps": config["optimizer_steps_cap"],
              "training_target_tokens_processed": config["target_token_budget"],
              "sampled_unique_windows": len(plan["training_sample_order"]),
              "sampled_documents": len({w["document_id"] for w in plan["training_sample_order"]}),
              "sampled_groups": len({w["group_id"] for w in plan["training_sample_order"]}),
              "training_corpus_token_fraction_exposed": config["target_token_budget"] / manifest["split_stats"]["train"]["unique_text_tokens"],
              "fresh_random_initialization": True, "prior_checkpoint_loaded": False, "test_file_opened": False,
              "parameter_count": parameter_count(model), "initial": initial, "best": best, "final": curve[-1],
              "best_restore": best_restore, "final_restore": final_restore,
              "final_in_memory_probe_max_abs_difference": in_memory_error,
              "worker_wall_seconds": time.monotonic() - started, "mlx_peak_bytes": mx.metal.get_peak_memory(),
              "mlx_active_bytes": mx.metal.get_active_memory(), "provenance": provenance}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: v3_worker.py RUN_DIRECTORY")
    print(json.dumps(worker(Path(sys.argv[1]))))
