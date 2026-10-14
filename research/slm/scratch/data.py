"""Immutable synthetic data loading and next-token windows."""
import json
from pathlib import Path

from prepare_data import digest
from tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/v1"


def load_data(config, data_dir=DATA):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["config_sha256"] != digest(ROOT / "config.v1.json"):
        raise ValueError("Synthetic config hash mismatch")
    for name, expected in manifest["file_sha256"].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"Synthetic artifact hash mismatch: {name}")
    if manifest["actual_vocab"] != config["model"]["vocabulary_target"]:
        raise ValueError("Tokenizer missed configured target vocabulary")
    tokenizer = ByteBPE.load(data_dir / "tokenizer.json")
    records = {}
    for name in ("train", "validation"):
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as stream:
            records[name] = [json.loads(line) for line in stream if line.strip()]
    if set(x["record_id"] for x in records["train"]) & set(x["record_id"] for x in records["validation"]):
        raise ValueError("Train/validation record overlap")
    if set(x["text"] for x in records["train"]) & set(x["text"] for x in records["validation"]):
        raise ValueError("Train/validation text overlap")
    return tokenizer, records, manifest


def windows(records, tokenizer, context_length):
    ids = [token for record in records for token in tokenizer.encode(record["text"], bos=True, eos=True)]
    size = context_length + 1
    result = [ids[start:start + size] for start in range(0, len(ids) - size + 1, context_length)]
    if not result or any(len(row) != size for row in result):
        raise ValueError("Not enough tokens for one complete context window")
    return result
