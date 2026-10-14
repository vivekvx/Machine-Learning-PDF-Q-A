"""Score frozen responses against independent semantic labels, without inferring them.

Each evaluation row must include an independently adjudicated context_sufficient
boolean and a blinded human response_class: abstain, answer, or unclear.
"""
from collections import Counter
import re

RESPONSE_CLASSES = {"abstain", "answer", "unclear"}
CITATION = re.compile(r"\[section:\d+:paragraph:\d+\]")
ANSWER = re.compile(r"(?s)^(.+?)\s+(\[section:\d+:paragraph:\d+\](?:\s+\[section:\d+:paragraph:\d+\])*)$")


def strict_marker(raw_response):
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be text")
    return raw_response.strip() == "UNANSWERABLE"


def strict_output(raw_response, allowed_passage_ids):
    """Return exact-marker, cited-answer, or invalid without inferring support."""
    if strict_marker(raw_response):
        return "exact_abstention"
    value = raw_response.strip()
    if "<think>" in value or "</think>" in value or "UNANSWERABLE" in value:
        return "invalid"
    match = ANSWER.fullmatch(value)
    if not match or not match.group(1).strip():
        return "invalid"
    citations = [token[1:-1] for token in CITATION.findall(match.group(2))]
    if not citations or any(citation not in allowed_passage_ids for citation in citations):
        return "invalid"
    return "cited_answer"


def _rates(counts):
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if precision is not None and recall is not None else None
    return {"counts": dict(counts), "precision": precision, "recall": recall,
            "f1": f1, "accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else None}


def score(rows):
    """Separate semantic abstention from exact marker compliance.

    Refuses missing human labels; this scorer never invents semantic decisions.
    """
    semantic = Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    marker = Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    unclear = 0
    total = 0
    compliant = 0
    for row in rows:
        if type(row.get("context_sufficient")) is not bool:
            raise ValueError("independent boolean context_sufficient label required")
        label = row.get("human_response_class")
        if label not in RESPONSE_CLASSES:
            raise ValueError("blinded human_response_class required")
        raw = row["raw_response"]
        allowed = row.get("allowed_passage_ids")
        if not isinstance(allowed, (list, tuple, set)) or not all(isinstance(x, str) for x in allowed):
            raise ValueError("supplied allowed_passage_ids required for format compliance")
        target_abstain = not row["context_sufficient"]
        for counts, predicted in ((semantic, label == "abstain"), (marker, strict_marker(raw))):
            key = ("tp" if target_abstain else "fp") if predicted else ("fn" if target_abstain else "tn")
            counts[key] += 1
        unclear += label == "unclear"
        compliant += strict_output(raw, allowed) != "invalid"
        total += 1
    return {"n": total, "human_unclear_count": unclear,
            "semantic_abstention": _rates(semantic), "exact_marker": _rates(marker),
            "strict_format_compliance": compliant / total if total else None}
