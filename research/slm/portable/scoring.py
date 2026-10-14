"""Deterministic QA, abstention, citation, and evidence-support measurements."""
import re
from collections import Counter

CITATION = re.compile(r"\[section:(\d+):paragraph:(\d+)\]")
PAGE_REFERENCE = re.compile(r"\b(?:page|p\.)\s*\d+\b", re.I)


def words(text):
    return re.findall(r"\w+", text.casefold())


def strip_citations(text):
    return CITATION.sub("", text).strip()


def exact_match(prediction, references):
    p = words(strip_citations(prediction))
    return max((int(p == words(r)) for r in references), default=0)


def token_f1(prediction, references):
    p = Counter(words(strip_citations(prediction)))
    scores = []
    for reference in references:
        r = Counter(words(reference))
        overlap = sum((p & r).values())
        scores.append(2 * overlap / (sum(p.values()) + sum(r.values())) if p or r else 1.0)
    return max(scores, default=0.0)


def citation_ids(text):
    return [f"section:{s}:paragraph:{p}" for s, p in CITATION.findall(text)]


def exact_abstention(text):
    return text.strip().upper() == "UNANSWERABLE"


def audit_prediction(row, prediction, latency_seconds, memory):
    citations = citation_ids(prediction)
    allowed = {p["passage_id"] for p in row["selected_passages"]}
    label = row["answerability"]
    # Lexical checks are proxies, never a claim of semantic entailment.
    reference_in_context = any(" ".join(words(row["reference_answer"])) in " ".join(words(p["text"]))
                               for p in row["selected_passages"]) if label == "answerable" else None
    answer_text = strip_citations(prediction)
    prediction_in_context = any(" ".join(words(answer_text)) in " ".join(words(p["text"]))
                                for p in row["selected_passages"]) if words(answer_text) and label == "answerable" else None
    return {"question_id": row["question_id"], "document_id": row["document_id"],
            "answerability": label, "prediction": prediction,
            "references": row["expected_answers"],
            "exact_match": exact_match(prediction, row["expected_answers"]) if label == "answerable" else None,
            "token_f1": token_f1(prediction, row["expected_answers"]) if label == "answerable" else None,
            "strict_abstention": exact_abstention(prediction),
            "citations": citations, "citation_is_source": bool(citations) and all(c in allowed for c in citations),
            "invented_page_reference": bool(PAGE_REFERENCE.search(prediction)),
            "reference_citation_present": row.get("reference_citation") in citations if label == "answerable" else None,
            "reference_text_in_context": reference_in_context,
            "prediction_text_in_context": prediction_in_context,
            "gold_passage_retrieved": row.get("gold_passage_retrieved"),
            "selected_passage_ids": sorted(allowed),
            "latency_seconds": latency_seconds, **memory}


def aggregate(results):
    answerable = [r for r in results if r["answerability"] == "answerable"]
    unanswerable = [r for r in results if r["answerability"] == "unanswerable"]
    def avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else None
    latencies = sorted(r["latency_seconds"] for r in results)
    return {"count": len(results), "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "answerable_exact_match": avg(answerable, "exact_match"),
            "answerable_token_f1": avg(answerable, "token_f1"),
            "answerable_false_abstention_rate": avg(answerable, "strict_abstention"),
            "unanswerable_strict_abstention_accuracy": avg(unanswerable, "strict_abstention"),
            "answerable_reference_citation_rate": avg(answerable, "reference_citation_present"),
            "unsupported_citation_predictions": sum(bool(r["citations"]) and not r["citation_is_source"] for r in results),
            "page_reference_predictions": sum(r["invented_page_reference"] for r in results),
            "answerable_reference_text_in_context_rate": avg(answerable, "reference_text_in_context"),
            "answerable_prediction_text_in_context_rate": avg(answerable, "prediction_text_in_context"),
            "answerable_gold_passage_recall": avg([r for r in answerable if r["gold_passage_retrieved"] is not None],
                                                  "gold_passage_retrieved"),
            "mean_latency_seconds": avg(results, "latency_seconds"),
            "p95_latency_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
            "peak_process_rss_bytes": max((r["process_rss_bytes"] for r in results), default=None),
            "peak_model_allocation_bytes": max((r["model_peak_bytes"] for r in results), default=None),
            "minimum_system_available_bytes": min((r["system_available_bytes"] for r in results), default=None)}


def manual_audit_rows(input_rows, scored_rows, per_label=5):
    by_id = {r["question_id"]: r for r in scored_rows}
    seen = Counter()
    output = []
    for row in input_rows:
        label = row["answerability"]
        if seen[label] >= per_label or row["question_id"] not in by_id:
            continue
        seen[label] += 1
        output.append({"question_id": row["question_id"], "document_id": row["document_id"],
                       "question": row["question"], "answerability": label,
                       "reference_answers": row["expected_answers"],
                       "prediction": by_id[row["question_id"]]["prediction"],
                       "passages": row["selected_passages"],
                       "manual_support_judgment": None,
                       "manual_citation_judgment": None})
    return output
