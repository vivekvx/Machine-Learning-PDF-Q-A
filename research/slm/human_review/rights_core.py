"""Separate, append-only human source-rights decisions for train documents."""
import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from review_core import HERE, sha256

LEDGER = HERE / "v1/provenance_ledger.jsonl"
DECISIONS = HERE / "v1/rights_decisions.jsonl"
REVIEWER = re.compile(r"[A-Za-z0-9._-]{3,64}\Z")
VALUES = ("permissive_verified", "restricted", "unresolved")
YN = ("Yes", "No", "Uncertain")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        result = [json.loads(line) for line in stream if line.strip()]
        fcntl.flock(stream, fcntl.LOCK_UN)
        return result


def append_rights(decision, *, ledger_path=LEDGER, decisions_path=DECISIONS, timestamp=None):
    if Path(ledger_path).resolve() == LEDGER.resolve():
        manifest = json.loads((HERE / "v1/provenance_manifest.json").read_text())
        if sha256(ledger_path) != manifest["provenance_ledger_sha256"]:
            raise ValueError("Pinned provenance ledger changed")
    ledger = read_jsonl(ledger_path)
    by_doc = {r["document_id"]: r for r in ledger}
    if len(by_doc) != 352 or decision.get("document_id") not in by_doc:
        raise ValueError("Document outside frozen training provenance ledger")
    if not REVIEWER.fullmatch(decision.get("reviewer_id", "")):
        raise ValueError("Valid independent rights reviewer ID required")
    if decision.get("decision") not in VALUES:
        raise ValueError("Invalid rights decision")
    for field in ("source_version_matches", "third_party_material_cleared", "training_and_redistribution_allowed"):
        if decision.get(field) not in YN:
            raise ValueError(f"Explicit {field} review required")
    for field in ("license_evidence_url", "attribution", "notes"):
        if not isinstance(decision.get(field), str) or len(decision[field]) > 4000:
            raise ValueError(f"Invalid {field}")
    if decision["decision"] == "permissive_verified":
        if by_doc[decision["document_id"]]["acl_publication_license_class"] != "verified_permissive_acl_publication":
            raise ValueError("Official ACL publication license is not verified permissive")
        if any(decision[field] != "Yes" for field in ("source_version_matches", "third_party_material_cleared", "training_and_redistribution_allowed")):
            raise ValueError("All source-version, third-party, and use-scope checks must be Yes")
        if not decision["license_evidence_url"].startswith("https://") or not decision["attribution"].strip() or not decision["notes"].strip():
            raise ValueError("Permissive decision requires license evidence URL, attribution, and rationale")
    event = {key: decision[key] for key in ("document_id", "reviewer_id", "decision", "source_version_matches",
             "third_party_material_cleared", "training_and_redistribution_allowed", "license_evidence_url", "attribution", "notes")}
    event.update({"schema_version": 1, "event_id": str(uuid4()),
                  "submitted_at_utc": timestamp or datetime.now(timezone.utc).isoformat(),
                  "provenance_ledger_sha256": sha256(ledger_path)})
    path = Path(decisions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    return event


def latest_rights(events, ledger_hash, doc_ids):
    latest = {}
    for event in events:
        if event.get("provenance_ledger_sha256") != ledger_hash or event.get("document_id") not in doc_ids:
            raise ValueError("Rights event has stale provenance hash or unknown document")
        if not REVIEWER.fullmatch(event.get("reviewer_id", "")) or event.get("decision") not in VALUES:
            raise ValueError("Malformed rights reviewer or decision")
        if any(event.get(field) not in YN for field in ("source_version_matches", "third_party_material_cleared", "training_and_redistribution_allowed")):
            raise ValueError("Malformed rights scope fields")
        if event["decision"] == "permissive_verified" and (
            any(event[field] != "Yes" for field in ("source_version_matches", "third_party_material_cleared", "training_and_redistribution_allowed"))
            or not event.get("license_evidence_url", "").startswith("https://")
            or not event.get("attribution", "").strip()
            or not event.get("notes", "").strip()
        ):
            raise ValueError("Malformed permissive rights decision")
        latest[(event["document_id"], event["reviewer_id"])] = event
    return latest
