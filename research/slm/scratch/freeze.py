"""Write a new, non-overwriting reproducibility manifest for SLM-A code/data."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from checkpoint import digest

ROOT = Path(__file__).resolve().parent
SOURCES = ("config.v1.json", "requirements.txt", "tokenizer.py", "prepare_data.py",
           "data.py", "model.py", "checkpoint.py", "train_worker.py", "run.py",
           "generate.py", "freeze.py", "tests/test_data_and_safety.py", "tests/test_mlx_model.py",
           "data/v1/train.jsonl", "data/v1/validation.jsonl", "data/v1/tokenizer.json",
           "data/v1/manifest.json")


def freeze(output):
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the isolated scratch environment")
    manifest = {"schema_version": 1, "project": "SLM-A-from-scratch",
                "python": sys.version.split()[0], "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name)
                             for name in ("mlx", "mlx-metal", "psutil")},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True).strip(),
                "file_sha256": {name: digest(ROOT / name) for name in SOURCES}}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2, sort_keys=True))
