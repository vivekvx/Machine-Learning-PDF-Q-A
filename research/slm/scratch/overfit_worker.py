"""Fresh-weight memorization probe. Launch only through overfit_run.py."""
import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from checkpoint import digest, load_checkpoint, save_checkpoint
from model import ScratchTransformer, next_token_loss, parameter_count
from tokenizer import ByteBPE, SPECIAL

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "config.v1.json"
EXPERIMENT = ROOT / "overfit/v1"
TOKENIZER = ROOT / "data/v1/tokenizer.json"


def provenance():
    return {"base_config_sha256": digest(BASE_CONFIG),
            "overfit_config_sha256": digest(EXPERIMENT / "config.json"),
            "fixture_manifest_sha256": digest(EXPERIMENT / "manifest.json"),
            "tokenizer_sha256": digest(TOKENIZER),
            "model_source_sha256": digest(ROOT / "model.py"),
            "worker_source_sha256": digest(__file__)}


def load_fixture():
    manifest = json.loads((EXPERIMENT / "manifest.json").read_text())
    if not manifest["synthetic_only"] or manifest["academic_qa_records"] != 0:
        raise ValueError("Overfit fixture must be synthetic only")
    if manifest["model_config_sha256"] != digest(BASE_CONFIG):
        raise ValueError("Base model configuration differs from fixture")
    if manifest["tokenizer_sha256"] != digest(TOKENIZER):
        raise ValueError("Tokenizer differs from fixture")
    if manifest["data_sha256"] != digest(EXPERIMENT / "data.jsonl"):
        raise ValueError("Overfit data hash mismatch")
    tokenizer = ByteBPE.load(TOKENIZER)
    rows = [json.loads(line) for line in (EXPERIMENT / "data.jsonl").read_text().splitlines()]
    if len(rows) != 6 or len({r["sequence_id"] for r in rows}) != 6 or len({r["text"] for r in rows}) != 6:
        raise ValueError("Overfit fixture must have six unique sequences")
    encoded = [tokenizer.encode(r["text"], bos=True, eos=True) for r in rows]
    if [len(ids) for ids in encoded] != manifest["token_lengths_with_bos_eos"]:
        raise ValueError("Token lengths differ from manifest")
    return tokenizer, rows, encoded


def full_set_loss(model, encoded):
    weighted = 0.0
    targets = 0
    per_sequence = []
    for ids in encoded:
        x = mx.array([ids[:-1]], dtype=mx.int32)
        y = mx.array([ids[1:]], dtype=mx.int32)
        loss = next_token_loss(model, x, y)
        mx.eval(loss)
        value = float(loss.item())
        per_sequence.append(value)
        weighted += value * len(ids[1:])
        targets += len(ids[1:])
    return weighted / targets, per_sequence


def continuations(model, tokenizer, rows, encoded):
    output = []
    for row, ids in zip(rows, encoded):
        prefix_length = max(6, (len(ids) - 1) // 2)
        current = list(ids[:prefix_length])
        expected = ids[prefix_length:]
        generated = []
        for _ in expected:
            logits = model(mx.array([current], dtype=mx.int32))
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            if token == SPECIAL["eos"]:
                break
            current.append(token)
        matching_prefix = 0
        for actual, gold in zip(generated, expected):
            if actual != gold:
                break
            matching_prefix += 1
        output.append({"sequence_id": row["sequence_id"],
                       "prompt": tokenizer.decode(ids[:prefix_length]),
                       "expected_suffix": tokenizer.decode(expected),
                       "generated_suffix": tokenizer.decode(generated),
                       "expected_token_ids": expected, "generated_token_ids": generated,
                       "exact": generated == expected,
                       "matching_prefix_tokens": matching_prefix,
                       "expected_token_count": len(expected)})
    return output


def gradient_l1_by_group(grads):
    flat = list(tree_flatten(grads))
    groups = {
        "token_embedding": lambda name: name.startswith("token."),
        "attention": lambda name: ".attn." in name,
        "feed_forward": lambda name: ".ff." in name,
        "output_head": lambda name: name.startswith("head."),
    }
    values = {}
    for group, predicate in groups.items():
        selected = [value for name, value in flat if predicate(name)]
        values[group] = sum(float(mx.sum(mx.abs(value)).item()) for value in selected)
    return values


def run(run_dir):
    if os.environ.get("SLMA_SAFETY_PARENT") != "overfit-v1":
        raise RuntimeError("Run through overfit_run.py so the memory watchdog remains active")
    base = json.loads(BASE_CONFIG.read_text())
    experiment = json.loads((EXPERIMENT / "config.json").read_text())
    if experiment["max_optimizer_steps"] > 200 or experiment["wall_time_cap_seconds"] > 120:
        raise ValueError("Initial overfit experiment exceeds hard step or time cap")
    if experiment["seed"] != base["seed"] or experiment["batch_size"] != base["model"]["batch_size"]:
        raise ValueError("Overfit fixture must retain model seed and batch size")
    tokenizer, rows, encoded = load_fixture()
    if any(len(ids) - 1 > base["model"]["context_length"] for ids in encoded):
        raise ValueError("Overfit sequence exceeds model context")
    mx.random.seed(experiment["seed"])
    model = ScratchTransformer(base["model"], tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=experiment["learning_rate"],
                            weight_decay=experiment["weight_decay"])
    mx.eval(model.parameters())
    first_token_weight = model.token.weight
    initial_loss, initial_per_sequence = full_set_loss(model, encoded)
    initial_continuations = continuations(model, tokenizer, rows, encoded)
    mx.metal.reset_peak_memory()
    objective = nn.value_and_grad(model, next_token_loss)
    started = time.monotonic()
    checkpoints = []
    evaluations = [{"step": 0, "full_set_loss": initial_loss,
                    "per_sequence_loss": initial_per_sequence,
                    "exact_reproductions": sum(r["exact"] for r in initial_continuations)}]
    (run_dir / "initial_continuations.json").write_text(json.dumps(initial_continuations, indent=2) + "\n")
    with (run_dir / "steps.jsonl").open("x") as step_log, (run_dir / "curve.jsonl").open("x") as curve_log:
        curve_log.write(json.dumps(evaluations[0], sort_keys=True) + "\n")
        curve_log.flush()
        os.fsync(curve_log.fileno())
        final_step = 0
        early_stopped = False
        latest_continuations = initial_continuations
        first_step_gradient_l1 = None
        first_step_embedding_update_l1 = None
        for step in range(1, experiment["max_optimizer_steps"] + 1):
            if time.monotonic() - started >= experiment["wall_time_cap_seconds"]:
                break
            ids = encoded[(step - 1) % len(encoded)]
            x = mx.array([ids[:-1]], dtype=mx.int32)
            y = mx.array([ids[1:]], dtype=mx.int32)
            tick = time.monotonic()
            loss, grads = objective(model, x, y)
            mx.eval(loss, grads)
            if not bool(mx.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            if step == 1:
                first_step_gradient_l1 = gradient_l1_by_group(grads)
                if any(value <= 0 for value in first_step_gradient_l1.values()):
                    raise RuntimeError("No gradient flow through a required model component")
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            if step == 1:
                first_step_embedding_update_l1 = float(mx.sum(mx.abs(first_token_weight - model.token.weight)).item())
                if first_step_embedding_update_l1 <= 0:
                    raise RuntimeError("Optimizer made no token-embedding update")
            duration = time.monotonic() - tick
            record = {"step": step, "sequence_id": rows[(step - 1) % len(rows)]["sequence_id"],
                      "loss_before_update": float(loss.item()), "duration_seconds": duration,
                      "tokens": len(ids) - 1, "tokens_per_second": (len(ids) - 1) / duration,
                      "mlx_active_bytes": mx.metal.get_active_memory(),
                      "mlx_cache_bytes": mx.metal.get_cache_memory(),
                      "mlx_peak_bytes": mx.metal.get_peak_memory()}
            step_log.write(json.dumps(record, sort_keys=True) + "\n")
            step_log.flush()
            final_step = step
            needs_loss_eval = step % experiment["loss_eval_interval_steps"] == 0
            needs_checkpoint = step % experiment["checkpoint_interval_steps"] == 0
            if needs_loss_eval or needs_checkpoint or step == experiment["max_optimizer_steps"]:
                mean_loss, per_sequence = full_set_loss(model, encoded)
                if needs_checkpoint or step == experiment["max_optimizer_steps"]:
                    latest_continuations = continuations(model, tokenizer, rows, encoded)
                assessment = {"step": step, "full_set_loss": mean_loss,
                              "per_sequence_loss": per_sequence,
                              "exact_reproductions": sum(r["exact"] for r in latest_continuations),
                              "continuations_evaluated": needs_checkpoint or step == experiment["max_optimizer_steps"]}
                evaluations.append(assessment)
                curve_log.write(json.dumps(assessment, sort_keys=True) + "\n")
                curve_log.flush()
                os.fsync(curve_log.fileno())
                print(json.dumps({"evaluation": assessment}), flush=True)
                if needs_checkpoint:
                    checkpoint_path = run_dir / "checkpoints" / f"step-{step:04d}"
                    saved = save_checkpoint(checkpoint_path, model, optimizer, step=step,
                                            cursor=step % len(rows), provenance=provenance())
                    checkpoints.append({"step": step, "path": str(checkpoint_path.relative_to(run_dir)),
                                        "file_sha256": saved["file_sha256"]})
                if (assessment["continuations_evaluated"] and
                        mean_loss <= experiment["early_stop_full_set_loss"] and
                        assessment["exact_reproductions"] == experiment["early_stop_exact_reproductions"]):
                    early_stopped = True
                    break
    if final_step == 0:
        raise RuntimeError("No optimizer step completed before worker time cap")
    if final_step % experiment["checkpoint_interval_steps"] != 0:
        checkpoint_path = run_dir / "checkpoints" / f"step-{final_step:04d}"
        saved = save_checkpoint(checkpoint_path, model, optimizer, step=final_step,
                                cursor=final_step % len(rows), provenance=provenance())
        checkpoints.append({"step": final_step, "path": str(checkpoint_path.relative_to(run_dir)),
                            "file_sha256": saved["file_sha256"]})
    else:
        checkpoint_path = run_dir / checkpoints[-1]["path"]
    final_loss, final_per_sequence = full_set_loss(model, encoded)
    final_continuations = continuations(model, tokenizer, rows, encoded)
    if evaluations[-1]["step"] != final_step:
        assessment = {"step": final_step, "full_set_loss": final_loss,
                      "per_sequence_loss": final_per_sequence,
                      "exact_reproductions": sum(r["exact"] for r in final_continuations),
                      "continuations_evaluated": True}
        with (run_dir / "curve.jsonl").open("a") as curve_log:
            curve_log.write(json.dumps(assessment, sort_keys=True) + "\n")
    (run_dir / "final_continuations.json").write_text(json.dumps(final_continuations, indent=2) + "\n")
    reloaded = ScratchTransformer(base["model"], tokenizer.vocab_size)
    reloaded_optimizer = optim.AdamW(learning_rate=experiment["learning_rate"],
                                    weight_decay=experiment["weight_decay"])
    metadata = load_checkpoint(checkpoint_path, reloaded, reloaded_optimizer,
                               expected_provenance=provenance())
    max_logit_error = 0.0
    for ids in encoded:
        x = mx.array([ids[:-1]], dtype=mx.int32)
        original = model(x)
        restored = reloaded(x)
        error = float(mx.max(mx.abs(original - restored)).item())
        max_logit_error = max(max_logit_error, error)
    restored_continuations = continuations(reloaded, tokenizer, rows, encoded)
    restore = {"checkpoint_step": metadata["step"],
               "max_absolute_logit_difference": max_logit_error,
               "generated_token_ids_identical": all(a["generated_token_ids"] == b["generated_token_ids"]
                                                    for a, b in zip(final_continuations, restored_continuations))}
    (run_dir / "restore_verification.json").write_text(json.dumps(restore, indent=2) + "\n")
    if max_logit_error > 1e-6 or not restore["generated_token_ids_identical"]:
        raise RuntimeError("Checkpoint restore changed logits or greedy continuations")
    result = {"status": "completed", "step_cap": experiment["max_optimizer_steps"],
              "completed_steps": final_step, "early_stopped": early_stopped,
              "actual_parameter_count": parameter_count(model), "actual_vocab_size": tokenizer.vocab_size,
              "initial_full_set_loss": initial_loss, "final_full_set_loss": final_loss,
              "final_per_sequence_loss": final_per_sequence,
              "initial_exact_reproductions": sum(r["exact"] for r in initial_continuations),
              "final_exact_reproductions": sum(r["exact"] for r in final_continuations),
              "first_step_gradient_l1_by_group": first_step_gradient_l1,
              "first_step_embedding_update_l1": first_step_embedding_update_l1,
              "worker_wall_seconds": time.monotonic() - started,
              "mlx_peak_bytes": mx.metal.get_peak_memory(),
              "mlx_active_bytes": mx.metal.get_active_memory(),
              "mlx_cache_bytes": mx.metal.get_cache_memory(),
              "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
              "restore_verification": restore,
              "provenance": provenance()}
    (run_dir / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"worker_result": result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)
