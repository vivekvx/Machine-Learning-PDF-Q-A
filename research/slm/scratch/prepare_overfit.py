"""Build an immutable six-sequence, synthetic-only memorization fixture."""
import argparse
import json
from pathlib import Path

from checkpoint import digest
from tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "config.v1.json"
TOKENIZER = ROOT / "data/v1/tokenizer.json"
DEFAULT_OUT = ROOT / "overfit/v1"
TEXTS = (
    "Record A: The amber robot moves north. The signal is blue.\n",
    "Record B: The violet sensor moves south. The signal is green.\n",
    "Record C: The copper relay moves east. The signal is silver.\n",
    "Record D: The green engine moves west. The signal is amber.\n",
    "Record E: The ivory panel moves upper. The signal is violet.\n",
    "Record F: The blue circuit moves lower. The signal is copper.\n",
)


def build(output_dir=DEFAULT_OUT):
    output_dir = Path(output_dir)
    data_file = output_dir / "data.jsonl"
    manifest_file = output_dir / "manifest.json"
    if data_file.exists() or manifest_file.exists():
        raise FileExistsError("Overfit fixture version exists; never overwrite it")
    config = json.loads(BASE_CONFIG.read_text())
    tokenizer = ByteBPE.load(TOKENIZER)
    if len(set(TEXTS)) != 6:
        raise ValueError("Synthetic fixture texts must be unique")
    rows = [{"sequence_id": f"synthetic-memorize-{i+1:02d}", "text": text}
            for i, text in enumerate(TEXTS)]
    lengths = [len(tokenizer.encode(row["text"], bos=True, eos=True)) for row in rows]
    if any(n < 3 or n - 1 > config["model"]["context_length"] for n in lengths):
        raise ValueError("Fixture sequence exceeds model context")
    output_dir.mkdir(parents=True, exist_ok=True)
    with data_file.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": 1, "synthetic_only": True,
                "academic_qa_records": 0, "sequence_count": len(rows),
                "token_lengths_with_bos_eos": lengths,
                "next_token_targets": sum(n - 1 for n in lengths),
                "model_config_sha256": digest(BASE_CONFIG),
                "tokenizer_sha256": digest(TOKENIZER),
                "data_sha256": digest(data_file),
                "builder_sha256": digest(__file__)}
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))
