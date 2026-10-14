"""Deterministic baseline metrics; no external evaluation model."""
import re
import string
from collections import Counter


def tokens(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.findall(r"\w+", text)


def exact_match(prediction, references):
    p = tokens(prediction)
    return max((int(p == tokens(r)) for r in references), default=0)


def token_f1(prediction, references):
    p = tokens(prediction)
    best = 0.0
    for reference in references:
        r = tokens(reference)
        overlap = sum((Counter(p) & Counter(r)).values())
        score = 2 * overlap / (len(p) + len(r)) if p or r else 1.0
        best = max(best, score)
    return best


def is_abstention(text):
    return text.strip().upper().startswith("UNANSWERABLE")


def summarize(predictions):
    n = len(predictions)
    if not n:
        return {"count": 0}
    return {"count": n,
            "exact_match": sum(p["exact_match"] for p in predictions) / n,
            "token_f1": sum(p["token_f1"] for p in predictions) / n,
            "abstention_accuracy": sum(p["abstention_correct"] for p in predictions) / n,
            "mean_latency_seconds": sum(p["latency_seconds"] for p in predictions) / n,
            "peak_process_rss_bytes": max(p["process_rss_bytes"] for p in predictions),
            "peak_mlx_active_bytes": max(p["mlx_active_bytes"] for p in predictions),
            "peak_mlx_bytes": max(p["mlx_peak_bytes"] for p in predictions)}
