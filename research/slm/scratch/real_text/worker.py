"""Bounded real-text pilot worker; only the parent watchdog may launch it."""
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "v1"
sys.path.insert(0, str(ROOT.parent))
from checkpoint import digest, load_checkpoint, save_checkpoint  # noqa: E402
from model import ScratchTransformer, next_token_loss, parameter_count  # noqa: E402
from tokenizer import ByteBPE, SPECIAL  # noqa: E402
from prepare import encode_windows, rights_gate  # noqa: E402


def load_verified():
    config = json.loads((V1 / "config.json").read_text())
    manifest = json.loads((V1 / "manifest.json").read_text())
    rights_gate(config)
    if manifest["rights_status"] != "official_documentation_prose_only_verified_with_pinned_license":
        raise ValueError("Corpus rights status is not verified")
    for name, expected in manifest["file_sha256"].items():
        if digest(V1 / name) != expected:
            raise ValueError(f"Real-text artifact hash mismatch: {name}")
    if manifest["source_sha256"] != config["expected_source_sha256"]:
        raise ValueError("Source hash ledger mismatch")
    train = [json.loads(x) for x in (V1 / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (V1 / "validation.jsonl").read_text().splitlines()]
    if len(train) != 3 or len(val) != 1 or ({r["document_id"] for r in train} & {r["document_id"] for r in val}):
        raise ValueError("Document split isolation failed")
    tokdata = json.loads((V1 / "tokenizer.json").read_text())
    if tokdata["type"] != "byte_bpe_from_cpython_3_12_15_train_only" or tokdata["training_document_ids"] != [r["document_id"] for r in train]:
        raise ValueError("Tokenizer has unexpected training sources")
    tokenizer = ByteBPE(tokdata["merges"], tokdata["target_size"])
    if tokenizer.vocab_size != tokdata["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size mismatch")
    train_windows = encode_windows(tokenizer, train, config["sequence_length"], config["max_train_windows"])
    val_windows = encode_windows(tokenizer, val, config["sequence_length"], config["max_validation_windows"])
    if {"train": len(train_windows), "validation": len(val_windows)} != manifest["fixed_sequence_windows"]:
        raise ValueError("Window counts differ from manifest")
    return config, manifest, tokenizer, train, val, train_windows, val_windows


def mean_loss(model, windows):
    values = []
    for row in windows:
        ids = row["ids"]
        loss = next_token_loss(model, mx.array([ids[:-1]], dtype=mx.int32), mx.array([ids[1:]], dtype=mx.int32))
        mx.eval(loss)
        values.append(float(loss.item()))
    return sum(values) / len(values)


def generation(model, tokenizer, row, n_tokens):
    prefix = tokenizer.encode(row["text"][:100], bos=True)[:40]
    current = list(prefix)
    output = []
    for _ in range(n_tokens):
        token = int(mx.argmax(model(mx.array([current[-model.context_length:]], dtype=mx.int32))[0, -1]).item())
        if token == SPECIAL["eos"]:
            break
        current.append(token)
        output.append(token)
    return {"document_id": row["document_id"], "prompt": tokenizer.decode(prefix),
            "continuation": tokenizer.decode(output), "generated_token_ids": output}


def run(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "real-text-v1":
        raise RuntimeError("The live memory watchdog is required")
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    if config["max_optimizer_steps"] > 100 or config["batch_size"] != 1 or config["sequence_length"] > 256 or config["wall_time_cap_seconds"] > 180:
        raise ValueError("Real-text pilot hard cap exceeded")
    base = json.loads((ROOT.parent / "config.v1.json").read_text())
    if base["model"]["layers"] != 4 or base["model"]["embedding_dim"] != 128 or base["model"]["attention_heads"] != 4:
        raise ValueError("Existing architecture changed")
    mx.random.seed(config["seed"])
    model = ScratchTransformer(base["model"], tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    mx.eval(model.parameters())
    count = parameter_count(model)
    initial_train = mean_loss(model, train_windows)
    initial_val = mean_loss(model, val_windows)
    mx.metal.reset_peak_memory()
    objective = nn.value_and_grad(model, next_token_loss)
    provenance = {"config_sha256": digest(V1 / "config.json"), "manifest_sha256": digest(V1 / "manifest.json"),
                  "tokenizer_sha256": digest(V1 / "tokenizer.json"), "base_model_config_sha256": digest(ROOT.parent / "config.v1.json"),
                  "model_source_sha256": digest(ROOT.parent / "model.py"), "worker_source_sha256": digest(__file__)}
    curve = [{"step": 0, "train_loss": initial_train, "validation_loss": initial_val,
              "train_perplexity": math.exp(initial_train), "validation_perplexity": math.exp(initial_val)}]
    started = time.monotonic()
    checkpoint_paths = []
    completed = 0
    with (run_dir / "steps.jsonl").open("x") as steps, (run_dir / "curve.jsonl").open("x") as losses:
        losses.write(json.dumps(curve[0], sort_keys=True) + "\n")
        losses.flush()
        for step in range(1, config["max_optimizer_steps"] + 1):
            if time.monotonic() - started > config["wall_time_cap_seconds"]:
                break
            ids = train_windows[(step - 1) % len(train_windows)]["ids"]
            inputs = mx.array([ids[:-1]], dtype=mx.int32)
            targets = mx.array([ids[1:]], dtype=mx.int32)
            tick = time.monotonic()
            loss, grads = objective(model, inputs, targets)
            mx.eval(loss, grads)
            if not bool(mx.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            duration = time.monotonic() - tick
            completed = step
            steps.write(json.dumps({"step": step, "train_window_document_id": train_windows[(step - 1) % len(train_windows)]["document_id"],
                                    "loss_before_update": float(loss.item()), "duration_seconds": duration,
                                    "tokens": config["sequence_length"], "tokens_per_second": config["sequence_length"] / duration,
                                    "mlx_active_bytes": mx.metal.get_active_memory(), "mlx_cache_bytes": mx.metal.get_cache_memory(),
                                    "mlx_peak_bytes": mx.metal.get_peak_memory()}, sort_keys=True) + "\n")
            steps.flush()
            if step % config["evaluation_interval_steps"] == 0 or step == config["max_optimizer_steps"]:
                train_loss = mean_loss(model, train_windows)
                val_loss = mean_loss(model, val_windows)
                assessment = {"step": step, "train_loss": train_loss, "validation_loss": val_loss,
                              "train_perplexity": math.exp(train_loss), "validation_perplexity": math.exp(val_loss)}
                curve.append(assessment)
                losses.write(json.dumps(assessment, sort_keys=True) + "\n")
                losses.flush()
                os.fsync(losses.fileno())
                print(json.dumps(assessment), flush=True)
                if train_loss > initial_train * 2:
                    raise RuntimeError("Training divergence early abort")
            if step % config["checkpoint_interval_steps"] == 0:
                path = run_dir / "checkpoints" / f"step-{step:04d}"
                save_checkpoint(path, model, optimizer, step=step, cursor=step % len(train_windows), provenance=provenance)
                checkpoint_paths.append(path)
    if completed == 0:
        raise RuntimeError("No optimizer steps completed")
    if completed % config["checkpoint_interval_steps"]:
        path = run_dir / "checkpoints" / f"step-{completed:04d}"
        save_checkpoint(path, model, optimizer, step=completed, cursor=completed % len(train_windows), provenance=provenance)
        checkpoint_paths.append(path)
    if curve[-1]["step"] != completed:
        train_loss = mean_loss(model, train_windows)
        val_loss = mean_loss(model, val_windows)
        assessment = {"step": completed, "train_loss": train_loss, "validation_loss": val_loss,
                      "train_perplexity": math.exp(train_loss), "validation_perplexity": math.exp(val_loss)}
        curve.append(assessment)
        with (run_dir / "curve.jsonl").open("a") as losses:
            losses.write(json.dumps(assessment, sort_keys=True) + "\n")
    restored = ScratchTransformer(base["model"], tokenizer.vocab_size)
    restored_optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    metadata = load_checkpoint(checkpoint_paths[-1], restored, restored_optimizer, expected_provenance=provenance)
    probe = mx.array([val_windows[0]["ids"][:-1]], dtype=mx.int32)
    max_error = float(mx.max(mx.abs(model(probe) - restored(probe))).item())
    if max_error > 1e-6 or metadata["step"] != completed:
        raise RuntimeError("Checkpoint restore failed logit comparison")
    examples = [generation(restored, tokenizer, train[0], config["generation_tokens"]),
                generation(restored, tokenizer, val[0], config["generation_tokens"])]
    (run_dir / "generations.json").write_text(json.dumps(examples, indent=2) + "\n")
    result = {"status": "completed", "completed_steps": completed, "step_cap": config["max_optimizer_steps"],
              "fresh_random_initialization": True, "prior_synthetic_weights_loaded": False,
              "actual_parameter_count": count, "actual_vocab_size": tokenizer.vocab_size,
              "initial": curve[0], "final": curve[-1], "worker_wall_seconds": time.monotonic() - started,
              "mlx_peak_bytes": mx.metal.get_peak_memory(), "mlx_active_bytes": mx.metal.get_active_memory(),
              "checkpoint_count": len(checkpoint_paths), "checkpoint_last": str(checkpoint_paths[-1].relative_to(run_dir)),
              "checkpoint_restore_max_absolute_logit_difference": max_error,
              "provenance": provenance}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py RUN_DIRECTORY")
    print(json.dumps(run(Path(sys.argv[1]))))
