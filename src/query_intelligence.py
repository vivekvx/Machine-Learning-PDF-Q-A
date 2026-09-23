"""
query_intelligence.py
---------------------
Smart query classification and expansion for academic RAG.

Converts a raw user question into:
  - A classified intent type (broad_summary, methodology, etc.)
  - A set of expanded retrieval queries to cast a wider semantic net
  - A recommended top_k value for retrieval

This module is pure Python with no external dependencies beyond stdlib.
"""

from __future__ import annotations
import re
import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Intent types
# ─────────────────────────────────────────────────────────

INTENT_BROAD_SUMMARY    = "broad_summary"
INTENT_METHODOLOGY      = "methodology"
INTENT_FINDINGS         = "findings"
INTENT_METRICS          = "metrics"
INTENT_SPECIFIC_FACT    = "specific_fact"
INTENT_COMPARISON       = "comparison"
INTENT_UNKNOWN          = "unknown"


@dataclass
class QueryIntelligence:
    """Output of classify_and_expand."""
    original_query: str
    intent: str
    expanded_queries: List[str]   # including original
    recommended_top_k: int
    is_broad: bool


# ─────────────────────────────────────────────────────────
# Keyword signal tables
# ─────────────────────────────────────────────────────────

_BROAD_SUMMARY_SIGNALS = [
    "summarize", "summary", "overview", "key finding", "main finding",
    "conclusion", "what does the paper", "what is this paper about",
    "what does this document", "main point", "key takeaway", "key point",
    "abstract", "briefly describe", "describe the paper",
    "what does the study", "what is the article about",
    "overall finding", "central argument", "primary argument",
    "what is presented", "what are the findings",
]

_METHODOLOGY_SIGNALS = [
    "methodology", "method", "approach", "how was", "how did", "technique",
    "procedure", "framework", "pipeline", "study design", "research design",
    "what method", "what approach", "how the study", "how the paper",
    "systematic review", "literature review", "review article",
    "experimental design", "algorithm", "model used",
]

_METRICS_SIGNALS = [
    "metric", "metrics", "accuracy", "performance", "evaluation", "f1",
    "precision", "recall", "score", "scores", "benchmark", "measurement",
    "measurements", "statistical", "p value", "p-value", "effect size",
    "how many", "percentage", "percent", "results reported",
]

_FINDINGS_SIGNALS = [
    "result", "results", "finding", "findings", "outcome", "outcomes",
    "what was found", "what did they find", "evidence", "reported", "showed",
    "demonstrated", "conclusion", "conclusions", "key takeaway", "implication",
]

_COMPARISON_SIGNALS = [
    "compare", "comparison", "vs", "versus", "difference between",
    "better than", "worse than", "contrast", "relative to", "against",
]

# ─────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────

def _contains(text: str, signals: List[str]) -> bool:
    t = text.lower()
    return any(s in t for s in signals)


def classify_intent(question: str) -> str:
    """Classify question intent into one of the INTENT_* constants."""
    q = question.lower().strip()

    if any(phrase in q for phrase in ("information is missing", "what is missing", "what is absent")):
        return INTENT_UNKNOWN

    # Check narrow intent before broad wording such as "key findings".
    if _contains(q, _COMPARISON_SIGNALS):
        return INTENT_COMPARISON
    if _contains(q, _METRICS_SIGNALS):
        return INTENT_METRICS
    if _contains(q, _FINDINGS_SIGNALS):
        return INTENT_FINDINGS
    if _contains(q, _METHODOLOGY_SIGNALS):
        return INTENT_METHODOLOGY
    if _contains(q, _BROAD_SUMMARY_SIGNALS):
        return INTENT_BROAD_SUMMARY

    # Heuristic: long questions with "what" / "how" tend to be broad
    words = q.split()
    if len(words) >= 8 and any(q.startswith(w) for w in ["what", "how", "describe", "explain"]):
        return INTENT_BROAD_SUMMARY

    return INTENT_SPECIFIC_FACT


# ─────────────────────────────────────────────────────────
# Query expander
# ─────────────────────────────────────────────────────────

def expand_query(question: str, intent: str) -> List[str]:
    """
    Return a list of retrieval queries including the original.
    For broad intents, adds semantically related variants to widen recall.
    """
    queries = [question]  # always include the original

    if intent == INTENT_BROAD_SUMMARY:
        queries += [
            "key findings conclusions main results summary",
            "abstract introduction overview purpose of the study",
            "conclusion discussion implications",
            "research in context interpretation future directions",
            "central argument thesis main contribution",
        ]

    elif intent == INTENT_METHODOLOGY:
        queries += [
            "methodology approach systematic review literature review",
            "study design research method framework procedure",
            "how the study was conducted experimental setup",
            "review article overview mechanism analysis approach",
        ]

    elif intent == INTENT_FINDINGS:
        queries += [
            "primary findings results evidence outcomes",
            "data conclusions demonstrated showed evidence",
            "experimental findings reported values outcomes",
            "key evidence findings implications clinical epidemiological",
            "animal model human tissue genetic findings biomarker",
        ]

    elif intent == INTENT_METRICS:
        queries += [
            "metrics results performance evaluation reported values",
            "tables figures statistics accuracy precision recall",
            "experimental results measurements effect sizes",
        ]

    elif intent == INTENT_COMPARISON:
        queries += [
            "comparison contrast difference between approaches",
            "relative performance versus comparison analysis",
        ]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ─────────────────────────────────────────────────────────
# Top-K recommendation
# ─────────────────────────────────────────────────────────

_TOP_K_BY_INTENT = {
    INTENT_BROAD_SUMMARY:  10,
    INTENT_METHODOLOGY:     8,
    INTENT_FINDINGS:        8,
    INTENT_METRICS:         8,
    INTENT_COMPARISON:      8,
    INTENT_SPECIFIC_FACT:   5,
    INTENT_UNKNOWN:         6,
}


def recommend_top_k(intent: str, index_size: int) -> int:
    """Return a recommended top_k bounded by the actual index size."""
    k = _TOP_K_BY_INTENT.get(intent, 6)
    return min(k, max(index_size, 1))


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────

def classify_and_expand(question: str, index_size: int = 999) -> QueryIntelligence:
    """
    Classify a user question and produce expanded retrieval queries.

    Parameters
    ----------
    question    : raw user question
    index_size  : total vectors in the FAISS index (to cap top_k)

    Returns
    -------
    QueryIntelligence dataclass
    """
    intent = classify_intent(question)
    expanded = expand_query(question, intent)
    top_k = recommend_top_k(intent, index_size)
    is_broad = intent in (INTENT_BROAD_SUMMARY, INTENT_METHODOLOGY,
                          INTENT_FINDINGS, INTENT_METRICS, INTENT_COMPARISON)

    logger.info(
        f"[QueryIntelligence] intent={intent}, top_k={top_k}, "
        f"expanded_queries={len(expanded)}, is_broad={is_broad}"
    )
    return QueryIntelligence(
        original_query=question,
        intent=intent,
        expanded_queries=expanded,
        recommended_top_k=top_k,
        is_broad=is_broad,
    )
