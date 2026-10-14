"""Freeze/verify evidence-checker v2 artifacts without touching pointer files."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ARTIFACT_HASHES.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def selected():
    return sorted(path for path in HERE.rglob("*") if path.is_file() and path != LEDGER
                  and "__pycache__" not in path.parts and path.suffix != ".pyc"
                  and path.name != ".DS_Store")


def current():
    return {str(path.relative_to(HERE)): sha(path) for path in selected()}


def write():
    if LEDGER.exists():
        raise FileExistsError("Evidence checker artifact ledger already frozen")
    hashes = current()
    LEDGER.write_text(json.dumps({"schema_version": 1, "root": "research/slm/scratch/evidence_checker",
                                  "algorithm": "SHA-256", "files": hashes}, indent=2, sort_keys=True) + "\n")
    return {"files": len(hashes)}


def verify():
    expected = json.loads(LEDGER.read_text())["files"]
    actual = current()
    if expected != actual:
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ValueError(f"Evidence artifact mismatch: added={sorted(set(actual)-set(expected))}, "
                         f"missing={sorted(set(expected)-set(actual))}, changed={changed}")
    return {"verified_files": len(actual)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write() if args.write else verify(), indent=2))
