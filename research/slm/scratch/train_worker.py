"""Bounded SLM-A trainer. Must be launched by the resource watchdog in run.py."""
import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from checkpoint import digest, load_checkpoint, save_checkpoint
from data import DATA, load_data, windows
from model import ScratchTransformer, next_token_loss, parameter_count

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.v1.json"


def provenance():
    return {"config_sha256": digest(CONFIG),
            "data_manifest_sha256": digest(DATA / "manifest.json"),
            "tokenizer_sha256": digest(DATA / "tokenizer.json"),
            "model_source_sha256": digest(ROOT / "model.py"),
            "worker_source_sha256": digest(__file__)}


def mean_loss(model, blocks, limit=2):
    values = []
    for row in blocks[:limit]:
        x = mx.array([row[:-1]], dtype=mx.int32)
        y = mx.array([row[1:]], dtype=mx.int32)
        value = next_token_loss(model, x, y)
        mx.eval(value)
        values.append(value.item())
    return sum(values) / len(values)


def run(mode, run_dir, resume=None):
    if os.environ.get("SLMA_SAFETY_PARENT") != "1":
        raise RuntimeError("Launch through run.py to enforce memory watchdog")
    config = json.loads(CONFIG.read_text())
    tokenizer, records, data_manifest = load_data(config)
    train_blocks = windows(records["train"], tokenizer, config["model"]["context_length"])
    valid_blocks = windows(records["validation"], tokenizer, config["model"]["context_length"])
    mx.random.seed(config["seed"])
    model = ScratchTransformer(config["model"], tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    checkpoint_provenance = provenance()
    start_step = 0
    if resume:
        info = load_checkpoint(resume, model, optimizer, expected_provenance=checkpoint_provenance)
        start_step = info["step"]
        if mode != "overfit" or start_step != config["training"]["smoke_steps"]:
            raise ValueError("Overfit stage must resume exactly the five-step smoke checkpoint")
    elif mode != "smoke":
        raise ValueError("Overfit stage requires a verified smoke checkpoint")
    mx.eval(model.parameters())
    initial_train_loss = mean_loss(model, train_blocks[:1])
    initial_validation_loss = mean_loss(model, valid_blocks)
    mx.metal.reset_peak_memory()
    maximum = config["training"]["smoke_steps"] if mode == "smoke" else config["training"]["overfit_additional_steps_cap"]
    threshold = config["training"]["overfit_early_stop_train_loss"]
    objective = nn.value_and_grad(model, next_token_loss)
    step_file = Path(run_dir) / "steps.jsonl"
    completed = 0
    early_stop = False
    with step_file.open("x", encoding="utf-8") as log:
        for local_index in range(maximum):
            global_step = start_step + local_index + 1
            block = train_blocks[local_index % len(train_blocks)] if mode == "smoke" else train_blocks[0]
            x = mx.array([block[:-1]], dtype=mx.int32)
            y = mx.array([block[1:]], dtype=mx.int32)
            started = time.monotonic()
            loss, grads = objective(model, x, y)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            duration = time.monotonic() - started
            loss_value = loss.item()
            record = {"mode": mode, "local_step": local_index + 1, "global_step": global_step,
                      "train_loss_before_update": loss_value, "duration_seconds": duration,
                      "tokens": len(block) - 1, "tokens_per_second": (len(block)-1) / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(),
                      "mlx_cache_bytes": mx.metal.get_cache_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
            completed += 1
            print(json.dumps(record), flush=True)
            if mode == "overfit" and loss_value <= threshold:
                early_stop = True
                break
    final_train_loss = mean_loss(model, train_blocks[:1])
    final_validation_loss = mean_loss(model, valid_blocks)
    checkpoint = save_checkpoint(Path(run_dir) / "checkpoint", model, optimizer,
                                 step=start_step + completed,
                                 cursor=completed % len(train_blocks) if mode == "smoke" else 0,
                                 provenance=checkpoint_provenance)
    result = {"status": "completed", "mode": mode, "completed_steps": completed,
              "global_step": start_step + completed, "early_stopped": early_stop,
              "actual_parameter_count": parameter_count(model), "actual_vocab_size": tokenizer.vocab_size,
              "train_blocks": len(train_blocks), "validation_blocks": len(valid_blocks),
              "initial_train_loss": initial_train_loss, "final_train_loss": final_train_loss,
              "initial_validation_loss": initial_validation_loss,
              "final_validation_loss": final_validation_loss,
              "mlx_peak_bytes": mx.metal.get_peak_memory(),
              "mlx_active_bytes": mx.metal.get_active_memory(),
              "mlx_cache_bytes": mx.metal.get_cache_memory(),
              "checkpoint_file_sha256": checkpoint["file_sha256"],
              "checkpoint_metadata_sha256": digest(Path(run_dir) / "checkpoint/metadata.json"),
              "data_manifest_sha256": digest(DATA / "manifest.json")}
    (Path(run_dir) / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"worker_result": result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "overfit"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.mode, args.run_dir, args.resume)
