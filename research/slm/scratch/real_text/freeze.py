"""Write a one-time checksum ledger for the v1 pilot without changing artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = ROOT.parent
CORPUS = ROOT / "v1"
RUN = SCRATCH / "runs/v1-python-docs-pilot"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    target = RUN / "artifact_hashes.json"
    if target.exists():
        raise FileExistsError("Frozen checksum ledger already exists")
    files = [p for p in (ROOT / "v1").rglob("*") if p.is_file()]
    files += [ROOT / "prepare.py", ROOT / "worker.py", ROOT / "pilot.py", ROOT / "freeze.py", ROOT / "quality_audit.py",
              ROOT / "RIGHTS_PROVENANCE.md", ROOT / "environment.json",
              SCRATCH / "model.py", SCRATCH / "tokenizer.py", SCRATCH / "checkpoint.py", SCRATCH / "run.py",
              SCRATCH / "REAL_TEXT_PILOT_REPORT.md", SCRATCH / "README.md",
              SCRATCH / "tests/test_real_text.py", SCRATCH / "checks/v4-real-text-unittest.log"]
    files += [p for p in RUN.rglob("*") if p.is_file() and p != target]
    data = {"schema_version": 1, "source_tag": "v3.12.15", "source_commit": "e848e4b09ca4bc67c0041c9ce7526dae471d4c36",
            "sha256": {str(p.relative_to(SCRATCH)): digest(p) for p in sorted(set(files))}}
    with target.open("x") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"files_hashed": len(data["sha256"]), "ledger_sha256": digest(target)}))


if __name__ == "__main__":
    freeze()
