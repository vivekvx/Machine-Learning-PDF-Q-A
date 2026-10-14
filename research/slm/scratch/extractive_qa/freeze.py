"""Seal extractive artifacts after checking all previously frozen work."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
OUTPUT = HERE / "artifact_hashes.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prior():
    ledgers = (SCRATCH / "runs/v1-python-docs-pilot/artifact_hashes.json",
               SCRATCH / "runs/v2-expanded-cpython/artifact_hashes.json",
               SCRATCH / "optimization/v1/artifact_hashes.json",
               SCRATCH / "real_text/v3/artifact_hashes.json",
               SCRATCH / "qa/artifact_hashes.json")
    output = {}
    for path in ledgers:
        ledger = json.loads(path.read_text())
        bad = [name for name, expected in ledger["sha256"].items() if sha(SCRATCH / name) != expected]
        if bad:
            raise ValueError(f"Prior frozen artifact changed: {bad}")
        output[str(path.relative_to(SCRATCH))] = sha(path)
    return output


def seal():
    if OUTPUT.exists():
        raise FileExistsError("Extractive ledger exists")
    previous = prior()
    files = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p != OUTPUT]
    files.append(SCRATCH / "EXTRACTIVE_QA_REPORT.md")
    hashes = {str(p.relative_to(SCRATCH)): sha(p) for p in sorted(files)}
    OUTPUT.write_text(json.dumps({"schema_version": 1, "prior_ledgers_sha256": previous,
                                  "sha256": hashes}, indent=2, sort_keys=True) + "\n")
    return len(hashes)


def verify():
    previous = prior()
    ledger = json.loads(OUTPUT.read_text())
    if previous != ledger["prior_ledgers_sha256"]:
        raise ValueError("Prior ledger hash changed")
    bad = [name for name, expected in ledger["sha256"].items() if sha(SCRATCH / name) != expected]
    if bad:
        raise ValueError(f"Extractive artifact changed: {bad}")
    return len(ledger["sha256"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seal == args.verify:
        parser.error("Choose one action")
    print(f"Verified files: {seal() if args.seal else verify()}")
