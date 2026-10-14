"""Read-only v2 checkpoint and update-integrity audit; never opens sealed test text."""
import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

SCRATCH = Path(__file__).resolve().parents[1]
REAL = SCRATCH / "real_text"
RUN = SCRATCH / "runs/v2-expanded-cpython"
sys.path[:0] = [str(SCRATCH), str(REAL)]
from checkpoint import digest, load_checkpoint  # noqa: E402
from model import ScratchTransformer, parameter_count  # noqa: E402
from expanded_worker import evaluate, load_verified, probe_values  # noqa: E402


def window_sha(windows):
    record = [{"document_id": w["document_id"], "offset": w["offset"], "ids": w["ids"]} for w in windows]
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_stats(tree):
    leaves = [v for _, v in tree_flatten(tree) if isinstance(v, mx.array)]
    if not leaves:
        raise ValueError("No tensor leaves")
    count = sum(v.size for v in leaves)
    squared = 0.0
    maximum = 0.0
    finite = True
    for v in leaves:
        mx.eval(v)
        finite &= bool(mx.all(mx.isfinite(v)).item())
        squared += float(mx.sum(mx.square(v.astype(mx.float32))).item())
        maximum = max(maximum, float(mx.max(mx.abs(v)).item()))
    return {"tensor_count": len(leaves), "scalar_count": count, "l2_norm": math.sqrt(squared),
            "max_absolute": maximum, "all_finite": finite}


def verify_ledger():
    ledger = json.loads((RUN / "artifact_hashes.json").read_text())
    mismatches = [name for name, expected in ledger["sha256"].items()
                  if digest(SCRATCH / name) != expected]
    if mismatches:
        raise ValueError(f"Frozen v2 files changed: {mismatches}")
    return {"ledger_sha256": digest(RUN / "artifact_hashes.json"),
            "verified_file_count": len(ledger["sha256"]), "mismatches": mismatches}


def reevaluate():
    ledger = verify_ledger()
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    if (len(train_windows), len(val_windows)) != (33, 23):
        raise ValueError("Fixed evaluation window count changed")
    model_config = json.loads((SCRATCH / "config.v1.json").read_text())["model"]
    recorded = {row["step"]: row for row in map(json.loads, (RUN / "curve.jsonl").read_text().splitlines())}
    reference_provenance = json.loads((RUN / "checkpoints/step-0060/metadata.json").read_text())["provenance"]
    results = {}
    for step in (60, 80):
        path = RUN / f"checkpoints/step-{step:04d}"
        model = ScratchTransformer(model_config, tokenizer.vocab_size)
        optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
        metadata = load_checkpoint(path, model, optimizer, expected_provenance=reference_provenance)
        if metadata["step"] != step or metadata["sample_cursor"] != step % len(train_windows):
            raise ValueError("Checkpoint step/cursor mismatch")
        state = dict(tree_flatten(optimizer.state))
        if int(state["step"].item()) != step:
            raise ValueError("Restored AdamW step counter mismatch")
        model.eval()
        train_eval = evaluate(model, train_windows)
        val_eval = evaluate(model, val_windows)
        probe = json.loads((path / "probe_logits.json").read_text())
        actual_probe = probe_values(model, val_windows[0])
        probe_error = max(abs(a - b) for a, b in zip(probe["values"], actual_probe))
        train_error = abs(train_eval["loss"] - recorded[step]["train"]["loss"])
        val_error = abs(val_eval["loss"] - recorded[step]["validation"]["loss"])
        if max(probe_error, train_error, val_error) > 1e-5:
            raise ValueError(f"Frozen step {step} evaluation failed reproduction")
        results[str(step)] = {"train": train_eval, "validation": val_eval,
                              "train_record_difference": train_error, "validation_record_difference": val_error,
                              "probe_max_abs_difference": probe_error, "model": tensor_stats(model.parameters()),
                              "optimizer": tensor_stats(optimizer.state),
                              "optimizer_step": int(state["step"].item()),
                              "model_sha256": metadata["file_sha256"]["model.safetensors"],
                              "optimizer_sha256": metadata["file_sha256"]["adamw.safetensors"],
                              "parameter_count": parameter_count(model)}
    return {"status": "verified", "frozen_ledger": ledger,
            "corpus_manifest_sha256": digest(REAL / "v2/manifest.json"),
            "train_window_sha256": window_sha(train_windows),
            "validation_window_sha256": window_sha(val_windows),
            "window_counts": {"train": len(train_windows), "validation": len(val_windows)},
            "sealed_test_opened": False, "checkpoints": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = reevaluate()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "window_counts": result["window_counts"],
                      "losses": {s: {k: x[k]["loss"] for k in ("train", "validation")}
                                 for s, x in result["checkpoints"].items()}}))
