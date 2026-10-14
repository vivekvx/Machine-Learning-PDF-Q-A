"""Sidecar formatting audit; never rewrites the immutable baseline results."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ("baseline_validation.json", "baseline_faiss_validation.json")
MARKER = re.compile(r"\bUNANSWERABLE\b", re.I)
PROSE = re.compile(r"(?:cannot be answered|does not (?:provide|address|establish|specify)|not explicitly stated|specific .* (?:not provided|not specified))", re.I)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(text):
    strict = text.strip().upper().startswith("UNANSWERABLE")
    marker = bool(MARKER.search(text))
    prose = bool(PROSE.search(text))
    return {"strict_marker_at_start": strict, "marker_anywhere": marker,
            "abstention_language": marker or prose,
            "format_or_parser_miss": not strict and (marker or prose)}


def build():
    report = {}
    for name in FILES:
        path = ROOT / name
        source = json.loads(path.read_text())
        entries = [{"question_id": p["question_id"], "answerability": p["answerability"],
                    **classify(p["prediction"])} for p in source["predictions"]]
        report[name] = {"source_sha256": sha256(path), "entries": entries,
                        "unanswerable_marker_anywhere": sum(e["marker_anywhere"] for e in entries if e["answerability"] == "unanswerable"),
                        "unanswerable_abstention_language": sum(e["abstention_language"] for e in entries if e["answerability"] == "unanswerable")}
    (ROOT / "baseline_format_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps({k: {"marker_anywhere": v["unanswerable_marker_anywhere"],
                          "abstention_language": v["unanswerable_abstention_language"]}
                      for k, v in build().items()}, indent=2))
