"""Train-only human evidence review storage and summaries; no model calls."""
import csv
import fcntl
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
SAMPLE = SLM / "EVIDENCE_REVIEW_SAMPLE.csv"
DECISIONS = HERE / "v1/reviews.jsonl"
JUDGMENTS = ("Supported", "Partially Supported", "Unsupported", "Ambiguous", "Needs Review")
ABSTENTION = ("Yes", "No", "Uncertain")
ID_RE = re.compile(r"[A-Za-z0-9._-]{3,64}\Z")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample(path=SAMPLE):
    if Path(path).resolve() == SAMPLE.resolve():
        manifest = json.loads((SLM / "training_data_v2/v1/manifest.json").read_text())
        if sha256(path) != manifest["file_sha256"]["EVIDENCE_REVIEW_SAMPLE.csv"]:
            raise ValueError("Frozen review sample hash changed")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 64 or len({row["question_id"] for row in rows}) != 64:
        raise ValueError("Expected unchanged 64-question review sample")
    human_fields = ("reviewer_id", "human_support_judgment", "human_evidence_condition",
                    "human_answerability_judgment", "human_contradiction_judgment",
                    "human_medical_domain_judgment", "human_rights_judgment", "review_notes", "reviewed_at")
    for row in rows:
        if row["split"] != "train" or any(row[field] for field in human_fields):
            raise ValueError("Review source contains human decisions or non-train data")
    return rows


def validate_decision(decision, sample_ids):
    if decision.get("question_id") not in sample_ids:
        raise ValueError("Question outside fixed review sample")
    if not ID_RE.fullmatch(decision.get("reviewer_id", "")):
        raise ValueError("Reviewer ID must be 3-64 letters, digits, dots, dashes, or underscores")
    if decision.get("judgment") not in JUDGMENTS:
        raise ValueError("Select one support judgment")
    if decision.get("abstention_appropriate") not in ABSTENTION:
        raise ValueError("Select whether abstention is appropriate for the supplied context")
    if decision.get("document_answerability_consistent") not in ABSTENTION:
        raise ValueError("Select whether the original document-level label is consistent")
    if decision.get("full_document_reviewed") not in ABSTENTION:
        raise ValueError("Record whether the full source document was reviewed")
    if decision.get("medical_domain") not in ABSTENTION:
        raise ValueError("Record medical-domain judgment")
    if len(decision.get("notes", "")) > 4000:
        raise ValueError("Notes exceed 4000 characters")
    if decision["judgment"] in ("Ambiguous", "Needs Review") and not decision.get("notes", "").strip():
        raise ValueError("Explain ambiguous or needs-review decisions")


def append_review(decision, *, sample_path=SAMPLE, decisions_path=DECISIONS, timestamp=None):
    sample_ids = {row["question_id"] for row in load_sample(sample_path)}
    validate_decision(decision, sample_ids)
    event = {key: decision[key] for key in ("question_id", "reviewer_id", "judgment", "abstention_appropriate", "document_answerability_consistent", "full_document_reviewed", "medical_domain")}
    event["notes"] = decision.get("notes", "")
    event.update({"event_id": str(uuid4()), "submitted_at_utc": timestamp or datetime.now(timezone.utc).isoformat(),
                  "sample_sha256": sha256(sample_path), "schema_version": 1})
    path = Path(decisions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    return event


def review_events(path=DECISIONS):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        result = [json.loads(line) for line in stream if line.strip()]
        fcntl.flock(stream, fcntl.LOCK_UN)
        return result


def latest_reviews(events, sample_ids, sample_hash):
    latest = {}
    for event in events:
        if event.get("sample_sha256") != sample_hash or event.get("question_id") not in sample_ids:
            raise ValueError("Review event points outside frozen sample/hash")
        validate_decision(event, sample_ids)
        latest[(event["question_id"], event["reviewer_id"])] = event
    return latest


def _agreement(pairs, field, choices):
    if not pairs:
        return {"overlap": 0, "raw_agreement": None, "cohens_kappa": None}
    left = [a[field] for a, _ in pairs]
    right = [b[field] for _, b in pairs]
    observed = sum(a == b for a, b in zip(left, right)) / len(pairs)
    expected = sum(left.count(label) * right.count(label) for label in choices) / len(pairs) ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"overlap": len(pairs), "raw_agreement": observed, "cohens_kappa": kappa}


def summarize(*, sample_path=SAMPLE, decisions_path=DECISIONS):
    sample = load_sample(sample_path)
    by_id = {r["question_id"]: r for r in sample}
    latest = latest_reviews(review_events(decisions_path), set(by_id), sha256(sample_path))
    by_question = defaultdict(dict)
    for (qid, reviewer), event in latest.items():
        by_question[qid][reviewer] = event
    reviewers = sorted({reviewer for _, reviewer in latest})
    pairwise = []
    for i, left in enumerate(reviewers):
        for right in reviewers[i + 1:]:
            pairs = [(by_question[q][left], by_question[q][right]) for q in by_question
                     if left in by_question[q] and right in by_question[q]]
            pairwise.append({"reviewers": [left, right],
                             "support": _agreement(pairs, "judgment", JUDGMENTS),
                             "abstention": _agreement(pairs, "abstention_appropriate", ABSTENTION)})
    agreed = []
    conflicts = []
    for qid, by_reviewer in by_question.items():
        values = list(by_reviewer.values())
        if len(values) < 2:
            continue
        pair_values = {(x["judgment"], x["abstention_appropriate"], x["document_answerability_consistent"], x["full_document_reviewed"], x["medical_domain"]) for x in values}
        if (len(pair_values) == 1 and values[0]["judgment"] not in ("Ambiguous", "Needs Review")
                and values[0]["abstention_appropriate"] != "Uncertain"
                and values[0]["document_answerability_consistent"] == "Yes"
                and values[0]["medical_domain"] == "No"
                and (by_id[qid]["answerability"] != "unanswerable" or values[0]["full_document_reviewed"] == "Yes")):
            agreed.append(qid)
        else:
            conflicts.append(qid)
    return {"sample_count": 64, "sample_sha256": sha256(sample_path), "review_event_count": len(review_events(decisions_path)),
            "unique_reviewer_question_count": len(latest), "reviewed_question_count": len(by_question),
            "reviewer_count": len(reviewers), "judgment_counts_latest": dict(Counter(x["judgment"] for x in latest.values())),
            "abstention_counts_latest": dict(Counter(x["abstention_appropriate"] for x in latest.values())),
            "agreement": pairwise, "agreed_question_ids": sorted(agreed), "conflict_or_unresolved_question_ids": sorted(conflicts),
            "unreviewed_question_count": 64 - len(by_question)}
