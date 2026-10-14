"""One-time SHA-256 ledger for corpus v3 and its bounded pilot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = ROOT.parent
V3 = ROOT / "v3"
RUN = SCRATCH / "runs/v4-cpython-v3-pilot"
OUTPUT = V3 / "artifact_hashes.json"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def freeze():
    if OUTPUT.exists():
        raise FileExistsError("v3 artifact ledger already exists")
    prior = {}
    for name in ("runs/v1-python-docs-pilot", "runs/v2-expanded-cpython", "optimization/v1"):
        ledger = SCRATCH / name / "artifact_hashes.json"
        data = json.loads(ledger.read_text())
        changed = [path for path, expected in data["sha256"].items() if sha(SCRATCH / path) != expected]
        if changed:
            raise ValueError(f"Frozen prior artifacts changed in {name}: {changed}")
        prior[name] = sha(ledger)
    chosen = [SCRATCH / "CORPUS_V3_REPORT.md", SCRATCH / "tests/test_corpus_v3.py",
              SCRATCH / "checks/v7-corpus-v3-unittest.log", SCRATCH / "checks/v7-v2-regression-unittest.log"]
    chosen.extend(ROOT / name for name in ("v3_prepare.py", "v3_tokenizer.py", "v3_data.py", "v3_worker.py",
                                             "v3_pilot.py", "v3_analyze.py", "v3_quality.py", "v3_freeze.py"))
    chosen.extend(path for path in V3.rglob("*") if path.is_file() and path != OUTPUT)
    chosen.extend(path for path in RUN.rglob("*") if path.is_file())
    files = {str(path.relative_to(SCRATCH)): sha(path) for path in sorted(set(chosen))}
    OUTPUT.write_text(json.dumps({"schema_version": 3, "prior_frozen_ledger_sha256": prior,
                                  "sha256": files}, indent=2, sort_keys=True) + "\n")
    return len(files)


if __name__ == "__main__":
    print(f"Frozen v3 files: {freeze()}")
