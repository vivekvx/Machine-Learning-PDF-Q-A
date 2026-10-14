"""Fresh-weight, bounded CPython v2 worker. Launch only through expanded_pilot."""
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "v2"
sys.path.insert(0, str(ROOT.parent))
from checkpoint import digest, load_checkpoint, save_checkpoint  # noqa: E402
from model import ScratchTransformer, next_token_loss, parameter_count  # noqa: E402
from tokenizer import ByteBPE, SPECIAL  # noqa: E402
from expanded_prepare import rights_gate, windows  # noqa: E402


def load_verified():
    config = json.loads((V2 / "config.json").read_text())
    manifest = json.loads((V2 / "manifest.json").read_text())
    rights_gate(config, include_test_sources=False)
    if manifest["source_commit"] != config["source_commit"] or manifest["source_sha256"] != config["expected_source_sha256"]:
        raise ValueError("Source provenance differs from corpus manifest")
    if not manifest["test_sealed_for_training_and_model_selection"]:
        raise ValueError("Test split is not sealed")
    # Deliberately never open test.sealed.jsonl or any held-out test source here.
    for name in ("config.json", "train.jsonl", "validation.jsonl", "tokenizer.json", "leakage_audit.json"):
        if digest(V2 / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Training-visible v2 artifact hash mismatch: {name}")
    leakage = json.loads((V2 / "leakage_audit.json").read_text())
    if any(leakage[key] for key in ("cross_split_exact_paragraph_pairs", "cross_split_near_paragraph_pairs", "cross_split_near_document_pairs")):
        raise ValueError("Unresolved cross-split duplicates")
    train = [json.loads(x) for x in (V2 / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (V2 / "validation.jsonl").read_text().splitlines()]
    if len(train) != 6 or len(val) != 3 or {r["document_id"] for r in train} & {r["document_id"] for r in val}:
        raise ValueError("Train/validation document split failure")
    tokdata = json.loads((V2 / "tokenizer.json").read_text())
    if tokdata["type"] != "byte_bpe_from_cpython_v2_train_documents_only" or tokdata["training_document_ids"] != [r["document_id"] for r in train]:
        raise ValueError("Tokenizer used unexpected documents")
    tokenizer = ByteBPE(tokdata["merges"], tokdata["target_size"])
    if tokenizer.vocab_size != tokdata["vocab_size"] or tokenizer.vocab_size != 1024:
        raise ValueError("Changed model vocabulary/parameter scale")
    train_windows = windows(tokenizer, train, config["sequence_length"], config["max_train_windows"])
    val_windows = windows(tokenizer, val, config["sequence_length"], config["max_validation_windows_per_document"], cap_per_document=True)
    if len(train_windows) != manifest["fixed_windows"]["train"] or len(val_windows) != manifest["fixed_windows"]["validation"]:
        raise ValueError("Fixed window counts differ")
    return config, manifest, tokenizer, train, val, train_windows, val_windows


def evaluate(model, windows):
    weighted = []
    grouped = {}
    for w in windows:
        ids = w["ids"]
        loss = next_token_loss(model, mx.array([ids[:-1]], dtype=mx.int32), mx.array([ids[1:]], dtype=mx.int32))
        mx.eval(loss)
        value = float(loss.item())
        weighted.append(value)
        grouped.setdefault(w["document_id"], []).append(value)
    mean = sum(weighted) / len(weighted)
    return {"loss": mean, "perplexity": math.exp(mean),
            "per_document_loss": {doc: sum(values) / len(values) for doc, values in grouped.items()},
            "windows": len(weighted)}


def probe_values(model, validation_window):
    ids = validation_window["ids"][:-1]
    logits = model(mx.array([ids], dtype=mx.int32))[0, -1]
    mx.eval(logits)
    return [float(x) for x in logits.tolist()]


def generate(model, tokenizer, row, n_tokens):
    prefix = tokenizer.encode(row["text"][:100], bos=True)[:40]
    current = list(prefix)
    new = []
    for _ in range(n_tokens):
        token = int(mx.argmax(model(mx.array([current[-model.context_length:]], dtype=mx.int32))[0, -1]).item())
        if token == SPECIAL["eos"]:
            break
        current.append(token)
        new.append(token)
    return {"document_id": row["document_id"], "prompt": tokenizer.decode(prefix),
            "continuation": tokenizer.decode(new), "token_ids": new}


def checkpoint_at(run_dir, step, model, optimizer, val_window, provenance, cursor):
    path = run_dir / "checkpoints" / f"step-{step:04d}"
    metadata = save_checkpoint(path, model, optimizer, step=step, cursor=cursor, provenance=provenance)
    values = probe_values(model, val_window)
    with (path / "probe_logits.json").open("x") as stream:
        json.dump({"validation_document_id": val_window["document_id"], "values": values}, stream, separators=(",", ":"))
        stream.write("\n")
    return path, metadata


def restore_and_verify(path, base_model_config, tokenizer, config, provenance, val_window):
    model = ScratchTransformer(base_model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    metadata = load_checkpoint(path, model, optimizer, expected_provenance=provenance)
    saved = json.loads((path / "probe_logits.json").read_text())
    if saved["validation_document_id"] != val_window["document_id"]:
        raise ValueError("Checkpoint probe source mismatch")
    observed = probe_values(model, val_window)
    error = max(abs(a - b) for a, b in zip(observed, saved["values"]))
    if len(observed) != len(saved["values"]) or error > 1e-6:
        raise RuntimeError("Restored checkpoint changed validation probe logits")
    return model, {"step": metadata["step"], "max_absolute_probe_logit_difference": error,
                   "model_sha256": metadata["file_sha256"]["model.safetensors"],
                   "optimizer_sha256": metadata["file_sha256"]["adamw.safetensors"]}


def bootstrap_delta(best, final, seed):
    docs = sorted(best)
    rng = random.Random(seed)
    deltas = []
    for _ in range(1000):
        picks = [rng.choice(docs) for _ in docs]
        deltas.append(sum(final[doc] - best[doc] for doc in picks) / len(picks))
    deltas.sort()
    return {"unit": "validation_document", "resamples": 1000,
            "delta_final_minus_best": sum(final[d] - best[d] for d in docs) / len(docs),
            "bootstrap_95_percent_interval": [deltas[24], deltas[974]],
            "limits": "Only three selected validation documents; descriptive resampling, not an independent test."}


def run(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "expanded-cpython-v2":
        raise RuntimeError("Live parent memory watchdog is required")
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    if config["max_optimizer_steps"] > 100 or config["batch_size"] != 1 or config["sequence_length"] > 256 or config["wall_time_cap_seconds"] > 180:
        raise ValueError("Expanded pilot hard safety cap exceeded")
    base = json.loads((ROOT.parent / "config.v1.json").read_text())
    model_config = base["model"]
    if (model_config["layers"], model_config["embedding_dim"], model_config["attention_heads"], model_config["context_length"]) != (4, 128, 4, 256):
        raise ValueError("Existing architecture dimensions changed")
    mx.random.seed(config["seed"])
    model = ScratchTransformer(model_config, tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    mx.eval(model.parameters())
    count = parameter_count(model)
    if count != 1082496:
        raise ValueError("Model parameter scale changed")
    initial_train = evaluate(model, train_windows)
    initial_val = evaluate(model, val_windows)
    mx.metal.reset_peak_memory()
    objective = nn.value_and_grad(model, next_token_loss)
    provenance = {"config_sha256": digest(V2 / "config.json"), "manifest_sha256": digest(V2 / "manifest.json"),
                  "train_sha256": digest(V2 / "train.jsonl"), "validation_sha256": digest(V2 / "validation.jsonl"),
                  "tokenizer_sha256": digest(V2 / "tokenizer.json"),
                  "base_model_config_sha256": digest(ROOT.parent / "config.v1.json"),
                  "model_source_sha256": digest(ROOT.parent / "model.py"), "worker_source_sha256": digest(__file__)}
    curve = [{"step": 0, "train": initial_train, "validation": initial_val}]
    checkpoints = {}
    best_step = None
    best_loss = float("inf")
    nonimproving = 0
    started = time.monotonic()
    completed = 0
    early_stop_reason = None
    with (run_dir / "steps.jsonl").open("x") as steps, (run_dir / "curve.jsonl").open("x") as curves:
        curves.write(json.dumps(curve[0], sort_keys=True) + "\n")
        curves.flush()
        for step in range(1, config["max_optimizer_steps"] + 1):
            if time.monotonic() - started >= config["wall_time_cap_seconds"]:
                early_stop_reason = "worker_wall_time_cap"
                break
            w = train_windows[(step - 1) % len(train_windows)]
            ids = w["ids"]
            x = mx.array([ids[:-1]], dtype=mx.int32)
            y = mx.array([ids[1:]], dtype=mx.int32)
            tick = time.monotonic()
            loss, grads = objective(model, x, y)
            mx.eval(loss, grads)
            if not bool(mx.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            duration = time.monotonic() - tick
            completed = step
            record = {"step": step, "train_document_id": w["document_id"], "tokens": config["sequence_length"],
                      "loss_before_update": float(loss.item()), "duration_seconds": duration,
                      "tokens_per_second": config["sequence_length"] / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(), "mlx_cache_bytes": mx.metal.get_cache_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            steps.write(json.dumps(record, sort_keys=True) + "\n")
            steps.flush()
            if step % config["evaluation_interval_steps"] == 0:
                assessment = {"step": step, "train": evaluate(model, train_windows), "validation": evaluate(model, val_windows)}
                curve.append(assessment)
                curves.write(json.dumps(assessment, sort_keys=True) + "\n")
                curves.flush()
                os.fsync(curves.fileno())
                print(json.dumps({"step": step, "train_loss": assessment["train"]["loss"],
                                  "validation_loss": assessment["validation"]["loss"]}), flush=True)
                if assessment["train"]["loss"] > initial_train["loss"] * 2:
                    raise RuntimeError("Training divergence abort")
                if step % config["checkpoint_interval_steps"] == 0:
                    path, metadata = checkpoint_at(run_dir, step, model, optimizer, val_windows[0], provenance, step % len(train_windows))
                    checkpoints[step] = path
                if assessment["validation"]["loss"] < best_loss - 1e-9:
                    best_loss = assessment["validation"]["loss"]
                    best_step = step
                    nonimproving = 0
                else:
                    nonimproving += 1
                if step >= 40 and nonimproving >= 4:
                    early_stop_reason = "four_validation_evaluations_without_improvement"
                    break
    if completed == 0:
        raise RuntimeError("No optimizer step completed")
    if completed not in checkpoints:
        # Covers a worker wall-time stop between scheduled evaluations.
        assessment = {"step": completed, "train": evaluate(model, train_windows), "validation": evaluate(model, val_windows)}
        curve.append(assessment)
        with (run_dir / "curve.jsonl").open("a") as curves:
            curves.write(json.dumps(assessment, sort_keys=True) + "\n")
        path, metadata = checkpoint_at(run_dir, completed, model, optimizer, val_windows[0], provenance, completed % len(train_windows))
        checkpoints[completed] = path
        if assessment["validation"]["loss"] < best_loss:
            best_step = completed
            best_loss = assessment["validation"]["loss"]
    final_step = completed
    final_model, final_restore = restore_and_verify(checkpoints[final_step], model_config, tokenizer, config, provenance, val_windows[0])
    best_model, best_restore = restore_and_verify(checkpoints[best_step], model_config, tokenizer, config, provenance, val_windows[0])
    if max(abs(a - b) for a, b in zip(probe_values(model, val_windows[0]), probe_values(final_model, val_windows[0]))) > 1e-6:
        raise RuntimeError("Final checkpoint differs from in-memory model")
    samples = {"prompt_policy": "First 100 characters of frozen train/validation document, 40-token encoded prefix, greedy 32-token continuation",
               "best_validation_step": best_step, "final_step": final_step,
               "best": [generate(best_model, tokenizer, train[0], config["generation_tokens"]),
                        generate(best_model, tokenizer, val[0], config["generation_tokens"]),
                        generate(best_model, tokenizer, val[2], config["generation_tokens"])],
               "final": [generate(final_model, tokenizer, train[0], config["generation_tokens"]),
                         generate(final_model, tokenizer, val[0], config["generation_tokens"]),
                         generate(final_model, tokenizer, val[2], config["generation_tokens"])]}
    (run_dir / "generations.json").write_text(json.dumps(samples, indent=2) + "\n")
    best_curve = next(x for x in curve if x["step"] == best_step)
    final_curve = next(x for x in reversed(curve) if x["step"] == final_step)
    diagnosis = {"best_validation_step": best_step, "final_step": final_step,
                 "validation_loss_difference_final_minus_best": final_curve["validation"]["loss"] - best_curve["validation"]["loss"],
                 "train_loss_difference_final_minus_best": final_curve["train"]["loss"] - best_curve["train"]["loss"],
                 "per_document_validation_delta_final_minus_best": {doc: final_curve["validation"]["per_document_loss"][doc] - best_curve["validation"]["per_document_loss"][doc]
                                                              for doc in best_curve["validation"]["per_document_loss"]},
                 "document_bootstrap": bootstrap_delta(best_curve["validation"]["per_document_loss"],
                                                      final_curve["validation"]["per_document_loss"], config["seed"]),
                 "interpretation_limit": "Fixed windows give deterministic within-run measurements; three documents cannot reliably separate general overfitting from validation-sample heterogeneity. Best-step selection itself uses this validation set."}
    (run_dir / "checkpoint_comparison.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    result = {"status": "completed", "completed_steps": completed, "step_cap": config["max_optimizer_steps"],
              "early_stop_reason": early_stop_reason, "fresh_random_initialization": True,
              "prior_checkpoint_loaded": False, "test_file_opened": False,
              "parameter_count": count, "vocabulary_size": tokenizer.vocab_size,
              "initial": curve[0], "best_validation": best_curve, "final": final_curve,
              "best_checkpoint": str(checkpoints[best_step].relative_to(run_dir)),
              "final_checkpoint": str(checkpoints[final_step].relative_to(run_dir)),
              "best_restore": best_restore, "final_restore": final_restore,
              "checkpoint_count": len(checkpoints), "worker_wall_seconds": time.monotonic() - started,
              "mlx_peak_bytes": mx.metal.get_peak_memory(), "mlx_active_bytes": mx.metal.get_active_memory(),
              "provenance": provenance}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: expanded_worker.py RUN_DIRECTORY")
    print(json.dumps(run(Path(sys.argv[1]))))
