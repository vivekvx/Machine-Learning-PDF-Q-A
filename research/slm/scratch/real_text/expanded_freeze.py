"""One-time v2 artifact checksum ledger; refuses to overwrite existing ledger."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = ROOT.parent
CORPUS = ROOT / "v2"
RUN = SCRATCH / "runs/v2-expanded-cpython"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    target = RUN / "artifact_hashes.json"
    if target.exists():
        raise FileExistsError("Frozen v2 checksum ledger already exists")
    files = [p for p in CORPUS.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [ROOT / "expanded_prepare.py", ROOT / "expanded_worker.py", ROOT / "expanded_pilot.py",
              ROOT / "expanded_freeze.py", ROOT / "prepare.py", ROOT / "environment.json",
              SCRATCH / "model.py", SCRATCH / "tokenizer.py", SCRATCH / "checkpoint.py",
              SCRATCH / "run.py", SCRATCH / "config.v1.json", SCRATCH / "README.md",
              SCRATCH / "EXPANDED_CORPUS_REPORT.md", SCRATCH / "tests/test_expanded_corpus.py",
              SCRATCH / "checks/v5-expanded-corpus-unittest.log"]
    files += [p for p in RUN.rglob("*") if p.is_file() and p != target and "__pycache__" not in p.parts]
    old_ledger = SCRATCH / "runs/v1-python-docs-pilot/artifact_hashes.json"
    record = {"schema_version": 2, "source_tag": "v3.12.15",
              "source_commit": "e848e4b09ca4bc67c0041c9ce7526dae471d4c36",
              "previous_v1_artifact_ledger_sha256": digest(old_ledger),
              "sha256": {str(p.relative_to(SCRATCH)): digest(p) for p in sorted(set(files))}}
    with target.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({"files_hashed": len(record["sha256"]), "ledger_sha256": digest(target),
                      "previous_v1_ledger_sha256": record["previous_v1_artifact_ledger_sha256"]}))


if __name__ == "__main__":
    freeze()
