"""Immutable SHA-256 ledger for the bounded pointer-aware checker v1 run."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = HERE / "v1"
LEDGER = VERSION / "ARTIFACT_HASHES.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paths():
    return sorted(p for p in VERSION.rglob("*") if p.is_file() and
                  "__pycache__" not in p.parts and p.name != ".DS_Store" and p != LEDGER)


def freeze():
    if LEDGER.exists():
        raise FileExistsError("Artifact ledger already frozen")
    hashes = {str(p.relative_to(VERSION)): digest(p) for p in paths()}
    result = {"schema_version": 1, "sealed_test_read": False,
              "files_sha256": hashes}
    LEDGER.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"frozen_files": len(hashes), "ledger_sha256": digest(LEDGER)}


def verify():
    ledger = json.loads(LEDGER.read_text())
    expected = ledger["files_sha256"]
    current = {str(p.relative_to(VERSION)): digest(p) for p in paths()}
    if current != expected:
        changed = sorted(set(current) ^ set(expected) | {key for key in current.keys() & expected.keys()
                     if current[key] != expected[key]})
        raise ValueError(f"Pointer-aware v1 artifacts changed: {changed}")
    return {"verified_files": len(current), "ledger_sha256": digest(LEDGER)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze() if args.freeze else verify(), indent=2))
