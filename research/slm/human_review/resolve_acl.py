"""Rebuild a versioned train-only ACL provenance ledger from a pinned export."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from acl_metadata import SELECTED, load_selected, resolve
from review_core import HERE, sha256

OUT = HERE / "v1"
CONFIG = OUT / "resolver_config.json"
EXPORT = OUT / "sources/acl_anthology_2026-10-10.bib.gz"
POLICY = OUT / "sources/acl_copyright_policy_2026-10-11.html"


def build(output_dir=OUT, bib_path=EXPORT, policy_path=POLICY):
    config = json.loads(CONFIG.read_text())
    if sha256(bib_path) != config["source_export_sha256"] or sha256(policy_path) != config["copyright_policy_sha256"]:
        raise ValueError("Official metadata or policy snapshot hash changed")
    baseline = json.loads((HERE.parent / "training_data_v2/v1/manifest.json").read_text())
    if sha256(SELECTED) != baseline["file_sha256"]["selected_audit.jsonl"]:
        raise ValueError("Frozen 352-row audit changed")
    output_dir = Path(output_dir)
    ledger_path = output_dir / "provenance_ledger.jsonl"
    manifest_path = output_dir / "provenance_manifest.json"
    if ledger_path.exists() or manifest_path.exists():
        raise FileExistsError("Provenance version already exists; do not overwrite")
    ledger = resolve(load_selected(), bib_path)
    if len(ledger) != 352:
        raise ValueError("Unexpected provenance row count")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("x", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": 1, "status": "publication_metadata_only_not_qasper_text_rights_clearance",
                "source_selected_audit_sha256": sha256(SELECTED), "resolver_config_sha256": sha256(CONFIG),
                "acl_export_sha256": sha256(bib_path), "acl_policy_sha256": sha256(policy_path),
                "tool_sha256": {"acl_metadata.py": sha256(HERE / "acl_metadata.py"),
                                "resolve_acl.py": sha256(HERE / "resolve_acl.py")},
                "provenance_ledger_sha256": sha256(ledger_path), "document_count": len(ledger),
                "match_counts": dict(Counter(row["match_method"] for row in ledger)),
                "publication_license_class_counts": dict(Counter(row["acl_publication_license_class"] for row in ledger)),
                "qasper_text_training_rights_cleared": 0, "human_rights_decisions": 0,
                "test_answers_read": False, "training_executed": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
