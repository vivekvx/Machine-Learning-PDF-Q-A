"""Freeze hashes for this isolated, versioned synthetic overfit experiment."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from checkpoint import digest

ROOT = Path(__file__).resolve().parent
SOURCES = (
    "config.v1.json", "model.py", "tokenizer.py", "checkpoint.py", "run.py",
    "overfit_run.py", "overfit_worker.py", "prepare_overfit.py", "export_curve.py",
    "freeze_overfit.py", "overfit/v1/config.json", "overfit/v1/data.jsonl",
    "overfit/v1/manifest.json", "data/v1/tokenizer.json",
    "tests/test_overfit.py", "reproducibility.v1.json", "reproducibility.v2.json",
    "runs/v1-smoke-memory-audit/run.json",
)
RUN_FILES = (
    "preflight.json", "run.json", "worker.log", "worker_result.json",
    "steps.jsonl", "curve.jsonl", "curve.csv", "initial_continuations.json",
    "final_continuations.json", "restore_verification.json",
)


def freeze(run_dir):
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use isolated scratch environment")
    run_dir = Path(run_dir)
    paths = sorted(RUN_FILES + tuple(str(path.relative_to(run_dir))
                                     for path in (run_dir / "checkpoints").rglob("*") if path.is_file()))
    manifest = {"schema_version": 1, "experiment": "SLM-A-synthetic-overfit-v1",
                "python": sys.version.split()[0], "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name)
                             for name in ("mlx", "mlx-metal", "psutil")},
                "source_sha256": {name: digest(ROOT / name) for name in SOURCES},
                "run_sha256": {name: digest(run_dir / name) for name in paths}}
    with (run_dir / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"source_files": len(SOURCES), "run_files": len(paths),
            "provenance_sha256": digest(run_dir / "provenance.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.run_dir)))
