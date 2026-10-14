"""Safe tensor-only model/AdamW checkpoints with integrity metadata."""
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save_checkpoint(path, model, optimizer, *, step, cursor, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Checkpoint directory already exists")
    path.mkdir(parents=True)
    model_file = path / "model.safetensors"
    optimizer_file = path / "adamw.safetensors"
    model.save_weights(str(model_file))
    state = dict(tree_flatten(optimizer.state))
    if not state or "step" not in state:
        raise RuntimeError("Optimizer state must contain a completed step")
    mx.save_safetensors(str(optimizer_file), state)
    metadata = {"schema_version": 1, "step": step, "sample_cursor": cursor,
                "parameter_count": sum(v.size for _, v in tree_flatten(model.parameters())),
                "provenance": provenance,
                "file_sha256": {"model.safetensors": digest(model_file),
                                "adamw.safetensors": digest(optimizer_file)}}
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_checkpoint(path, model, optimizer, *, expected_provenance):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["provenance"] != expected_provenance:
        raise ValueError("Checkpoint configuration or data provenance mismatch")
    for name, expected in metadata["file_sha256"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    model.load_weights(str(path / "model.safetensors"), strict=True)
    state = mx.load(str(path / "adamw.safetensors"))
    optimizer.state = tree_unflatten(list(state.items()))
    if sum(v.size for _, v in tree_flatten(model.parameters())) != metadata["parameter_count"]:
        raise ValueError("Checkpoint parameter count mismatch")
    mx.eval(model.parameters(), optimizer.state)
    return metadata
