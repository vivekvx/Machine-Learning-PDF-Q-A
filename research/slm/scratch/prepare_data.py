"""Build deterministic synthetic-only, document-disjoint train and validation text."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.v1.json"
OUT = ROOT / "data/v1"
COLORS = ("amber", "blue", "copper", "green", "ivory", "violet", "silver", "ochre")
OBJECTS = ("robot", "sensor", "ledger", "beacon", "engine", "panel", "circuit", "relay")
PLACES = ("north", "south", "east", "west", "upper", "lower")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentence(i):
    color = COLORS[(i * 5 + 3) % len(COLORS)]
    object_name = OBJECTS[(i * 3 + 1) % len(OBJECTS)]
    place = PLACES[(i * 7 + 2) % len(PLACES)]
    value = (i * 17 + 11) % 97
    checksum = hashlib.sha256(f"synthetic-record-{i:04d}".encode()).hexdigest()[:32]
    return (f"Record {i:04d}. The {color} {object_name} moves {place}. "
            f"Its measured value is {value:02d}. The control log confirms this route. "
            f"Checksum {checksum}.\n")


def build(output_dir=OUT, config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    n_train = config["data"]["train_records"]
    n_valid = config["data"]["validation_records"]
    train = [{"record_id": f"synthetic-train-{i:04d}", "text": sentence(i)} for i in range(n_train)]
    valid = [{"record_id": f"synthetic-validation-{i:04d}", "text": sentence(i)}
             for i in range(n_train, n_train + n_valid)]
    if {r["record_id"] for r in train} & {r["record_id"] for r in valid} or {r["text"] for r in train} & {r["text"] for r in valid}:
        raise RuntimeError("Synthetic train/validation overlap")
    output_dir = Path(output_dir)
    outputs = [output_dir / name for name in ("train.jsonl", "validation.jsonl", "tokenizer.json", "manifest.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Synthetic data version already exists")
    tokenizer = ByteBPE.train([r["text"] for r in train], config["model"]["vocabulary_target"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("validation.jsonl", valid)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tokenizer.save(output_dir / "tokenizer.json")
    manifest = {"schema_version": 1, "synthetic_only": True, "train_count": len(train),
                "validation_count": len(valid), "train_validation_record_overlap": 0,
                "train_validation_text_overlap": 0, "tokenizer_trained_on": "train.jsonl only",
                "tokenizer_target_vocab": config["model"]["vocabulary_target"],
                "actual_vocab": tokenizer.vocab_size, "config_sha256": digest(config_path),
                "file_sha256": {name: digest(output_dir / name) for name in
                                ("train.jsonl", "validation.jsonl", "tokenizer.json")},
                "generation_script_sha256": digest(__file__),
                "tokenizer_script_sha256": digest(ROOT / "tokenizer.py")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))
