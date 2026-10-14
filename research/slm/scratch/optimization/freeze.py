"""Write a one-time SHA-256 ledger for optimization audit outputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
OUTPUT = HERE / "v1/artifact_hashes.json"


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create():
    if OUTPUT.exists():
        raise FileExistsError("Optimization artifact ledger already exists")
    prior = {}
    for name in ("v1-python-docs-pilot", "v2-expanded-cpython"):
        ledger = SCRATCH / "runs" / name / "artifact_hashes.json"
        data = json.loads(ledger.read_text())
        changed = [relative for relative, expected in data["sha256"].items()
                   if sha(SCRATCH / relative) != expected]
        if changed:
            raise ValueError(f"Prior frozen artifacts changed: {name}: {changed}")
        prior[name] = sha(ledger)
    selected = [SCRATCH / "OPTIMIZATION_STABILITY_REPORT.md",
                SCRATCH / "tests/test_optimization_stability.py",
                SCRATCH / "checks/v6-optimization-stability-unittest.log"]
    selected.extend(path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path != OUTPUT)
    selected.extend(path for path in (SCRATCH / "runs/v3-optimization-low-lr").rglob("*") if path.is_file())
    files = {str(path.relative_to(SCRATCH)): sha(path) for path in sorted(set(selected))}
    OUTPUT.write_text(json.dumps({"schema_version": 1, "prior_frozen_ledger_sha256": prior,
                                  "sha256": files}, indent=2, sort_keys=True) + "\n")
    return len(files)


if __name__ == "__main__":
    print(f"Frozen optimization files: {create()}")
