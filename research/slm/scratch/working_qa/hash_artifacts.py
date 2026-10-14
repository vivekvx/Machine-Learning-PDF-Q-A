"""Freeze/verify SHA-256 for new SLM-A QA artifacts, excluding cache files."""
import argparse
import json
from pathlib import Path

from prepare import sha

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
LEDGER = HERE / "ARTIFACT_HASHES.json"


def selected():
    files = [path for path in HERE.rglob("*") if path.is_file() and
             path != LEDGER and "__pycache__" not in path.parts and
             path.suffix != ".pyc" and path.name != ".DS_Store"]
    files += [SCRATCH / "FINAL_WORKING_SLM_REPORT.md", SCRATCH / "DEMO_AND_VIVA.md"]
    return sorted(files)


def paths_and_hashes():
    return {str(path.relative_to(SCRATCH)): sha(path) for path in selected()}


def write():
    if LEDGER.exists():
        raise FileExistsError("Artifact ledger already frozen")
    hashes = paths_and_hashes()
    LEDGER.write_text(json.dumps({"schema_version": 1, "algorithm": "SHA-256",
                                  "root": "research/slm/scratch", "files": hashes}, indent=2) + "\n")
    return {"files": len(hashes), "ledger": str(LEDGER)}


def verify():
    expected = json.loads(LEDGER.read_text())["files"]
    actual = paths_and_hashes()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
        raise ValueError(f"Artifact mismatch: missing={missing}, added={added}, changed={changed}")
    return {"verified_files": len(actual)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write() if args.write else verify(), indent=2))
