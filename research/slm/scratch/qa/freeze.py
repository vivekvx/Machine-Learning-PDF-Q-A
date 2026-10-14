"""Verify earlier frozen research and seal the synthetic QA experiment."""
import hashlib
import json
from pathlib import Path

QA = Path(__file__).resolve().parent
SCRATCH = QA.parent
OUTPUT = QA / "artifact_hashes.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_prior():
    result = {}
    for path in (SCRATCH / "runs/v1-python-docs-pilot/artifact_hashes.json",
                 SCRATCH / "runs/v2-expanded-cpython/artifact_hashes.json",
                 SCRATCH / "optimization/v1/artifact_hashes.json",
                 SCRATCH / "real_text/v3/artifact_hashes.json"):
        ledger = json.loads(path.read_text())
        bad = [name for name, expected in ledger["sha256"].items() if sha(SCRATCH / name) != expected]
        if bad:
            raise ValueError(f"Previously frozen artifacts changed: {bad}")
        result[str(path.relative_to(SCRATCH))] = sha(path)
    return result


def seal():
    if OUTPUT.exists():
        raise FileExistsError("QA ledger already exists")
    previous = verify_prior()
    files = [path for path in QA.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path != OUTPUT]
    files.append(SCRATCH / "SCRATCH_QA_REPORT.md")
    hashes = {str(path.relative_to(SCRATCH)): sha(path) for path in sorted(files)}
    ledger = {"schema_version": 1, "prior_frozen_ledgers_sha256": previous, "sha256": hashes}
    OUTPUT.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return len(hashes)


def verify():
    previous = verify_prior()
    ledger = json.loads(OUTPUT.read_text())
    if previous != ledger["prior_frozen_ledgers_sha256"]:
        raise ValueError("Prior frozen ledgers changed")
    bad = [name for name, expected in ledger["sha256"].items() if sha(SCRATCH / name) != expected]
    if bad:
        raise ValueError(f"QA artifacts changed: {bad}")
    return len(ledger["sha256"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seal == args.verify:
        parser.error("Choose exactly one action")
    print(f"Verified QA and prior files: {seal() if args.seal else verify()}")
