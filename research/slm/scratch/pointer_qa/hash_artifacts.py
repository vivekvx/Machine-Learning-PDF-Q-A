"""Freeze and verify new pointer QA artifacts without opening sealed test answers."""
import argparse
import json
from pathlib import Path

from prepare import sha

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ARTIFACT_HASHES.json"


def selected():
    return sorted(path for path in HERE.rglob("*") if path.is_file() and path != LEDGER
                  and "__pycache__" not in path.parts and path.suffix != ".pyc"
                  and path.name not in (".DS_Store", "test.sealed.jsonl"))


def current():
    return {str(path.relative_to(HERE)): sha(path) for path in selected()}


def write():
    if LEDGER.exists():
        raise FileExistsError("Pointer QA hash ledger already frozen")
    hashes = current()
    LEDGER.write_text(json.dumps({"schema_version": 1, "algorithm": "SHA-256",
                                  "root": "research/slm/scratch/pointer_qa",
                                  "sealed_test_content_excluded": True,
                                  "files": hashes}, indent=2, sort_keys=True) + "\n")
    return {"files": len(hashes), "ledger": str(LEDGER)}


def verify():
    expected = json.loads(LEDGER.read_text())["files"]
    actual = current()
    if actual != expected:
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ValueError(f"Pointer artifact mismatch: added={sorted(set(actual)-set(expected))}, "
                         f"missing={sorted(set(expected)-set(actual))}, changed={changed}")
    return {"verified_files": len(actual)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write() if args.write else verify(), indent=2))
