"""
evaluation.py
-------------
Lightweight evaluation module for the RAG pipeline.

Metrics tracked per query:
  - retrieval_latency   : seconds to embed + search
  - generation_latency  : seconds for LLM inference
  - total_latency       : end-to-end response time
  - num_chunks          : how many chunks were retrieved
  - avg_similarity      : average cosine similarity of retrieved chunks
  - top_similarity      : highest cosine similarity score
  - model_used          : which LLM generated the answer

Optional RAG-quality metrics (stubs for research paper discussion):
  - faithfulness        : does the answer stay within the retrieved context?
  - answer_relevancy    : how well does the answer address the question?
  - context_precision   : fraction of retrieved chunks that are actually useful
  - context_recall      : fraction of relevant chunks that were retrieved

Note: Full faithfulness / relevancy evaluation typically requires an LLM judge
(like RAGAS). The stubs below let you discuss them in your paper and extend
later without changing the interface.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import csv
import time
import logging
from pathlib import Path

from src.retriever import RetrievalResult
from src.generator import GenerationResult
from src.config import EVAL_RESULTS_FILE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Per-query evaluation record
# ─────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    """One row in the evaluation log."""
    query: str
    answer: str
    retrieval_latency: float        # seconds
    generation_latency: float       # seconds
    total_latency: float            # seconds
    num_chunks: int
    avg_similarity: float
    top_similarity: float
    source_pages: str               # comma-separated page numbers
    source_files: str               # comma-separated filenames
    model_used: str
    # ── Optional RAG-quality metrics ───────────────────────
    faithfulness: Optional[float]      = None   # 0–1 (stub)
    answer_relevancy: Optional[float]  = None   # 0–1 (stub)
    context_precision: Optional[float] = None   # 0–1 (stub)
    context_recall: Optional[float]    = None   # 0–1 (stub)
    notes: str = ""


# ─────────────────────────────────────────────────────────
# Evaluation tracker (session-level)
# ─────────────────────────────────────────────────────────

class EvaluationTracker:
    """
    Collects EvalRecord objects for every query answered in the session.

    Usage:
        tracker = EvaluationTracker()
        record  = tracker.log(retrieval_result, generation_result)
        summary = tracker.summary()
        tracker.export_csv("results.csv")
    """

    def __init__(self):
        self.records: List[EvalRecord] = []

    def log(
        self,
        retrieval: RetrievalResult,
        generation: GenerationResult,
        notes: str = "",
    ) -> EvalRecord:
        """
        Create and store an EvalRecord from retrieval + generation results.

        Parameters
        ----------
        retrieval  : RetrievalResult from retriever.retrieve()
        generation : GenerationResult from generator.generate_answer()
        notes      : optional free-text note (e.g., "manual test case #3")

        Returns
        -------
        EvalRecord
        """
        record = EvalRecord(
            query              = retrieval.query,
            answer             = generation.answer,
            retrieval_latency  = retrieval.retrieval_time,
            generation_latency = generation.generation_time,
            total_latency      = round(retrieval.retrieval_time + generation.generation_time, 4),
            num_chunks         = len(retrieval.chunks),
            avg_similarity     = retrieval.avg_similarity,
            top_similarity     = retrieval.top_score,
            source_pages       = ", ".join(str(p) for p in generation.source_pages),
            source_files       = ", ".join(generation.source_files),
            model_used         = generation.model_used,
            notes              = notes,
        )
        self.records.append(record)
        logger.info(
            f"Eval logged | latency={record.total_latency:.2f}s "
            f"| sim={record.avg_similarity:.3f} | chunks={record.num_chunks}"
        )
        return record

    def summary(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all recorded queries.

        Returns a dict suitable for display in the Streamlit sidebar.
        """
        if not self.records:
            return {"total_queries": 0}

        n = len(self.records)
        avg = lambda key: round(sum(getattr(r, key) for r in self.records) / n, 4)

        return {
            "total_queries":         n,
            "avg_retrieval_latency": avg("retrieval_latency"),
            "avg_generation_latency":avg("generation_latency"),
            "avg_total_latency":     avg("total_latency"),
            "avg_num_chunks":        avg("num_chunks"),
            "avg_similarity":        avg("avg_similarity"),
            "avg_top_similarity":    avg("top_similarity"),
        }

    def export_csv(self, filepath: str = EVAL_RESULTS_FILE) -> str:
        """
        Export all evaluation records to a CSV file.

        Parameters
        ----------
        filepath : destination CSV path

        Returns
        -------
        str : absolute path of the created file
        """
        if not self.records:
            logger.warning("No evaluation records to export.")
            return ""

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(self.records[0]).keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

        logger.info(f"Evaluation results exported to {filepath}")
        return str(Path(filepath).resolve())

    def clear(self):
        """Reset the evaluation log."""
        self.records.clear()
        logger.info("Evaluation log cleared.")


# ─────────────────────────────────────────────────────────
# RAG-quality metric stubs
# ─────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, context_chunks: list) -> float:
    """
    Estimate how faithful the answer is to the retrieved context.

    Full implementation would use an NLI model or an LLM judge (RAGAS style).
    This stub returns a placeholder value of -1.0 to indicate "not computed".

    For your research paper, discuss this as:
    "We measured faithfulness qualitatively by manual inspection of 20 sample
    Q&A pairs and found that the system stayed within the document context in
    X% of cases."
    """
    # TODO: Implement using cross-encoder NLI model or RAGAS
    return -1.0


def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Estimate how well the answer addresses the question.

    Stub returns -1.0. A full implementation would embed both the question
    and the answer and compute their cosine similarity, or use an LLM judge.
    """
    # TODO: Embed query and answer → cosine similarity
    return -1.0


def compute_context_precision(
    retrieved_chunks: list,
    relevant_chunk_ids: list,
) -> float:
    """
    Precision = |retrieved ∩ relevant| / |retrieved|

    Requires ground-truth annotations of which chunks are truly relevant.
    Returns -1.0 if no ground truth is provided.
    """
    if not relevant_chunk_ids:
        return -1.0
    retrieved_ids = {c.chunk_index for c in retrieved_chunks}
    relevant_set  = set(relevant_chunk_ids)
    precision = len(retrieved_ids & relevant_set) / len(retrieved_ids)
    return round(precision, 4)


def compute_context_recall(
    retrieved_chunks: list,
    relevant_chunk_ids: list,
) -> float:
    """
    Recall = |retrieved ∩ relevant| / |relevant|

    Requires ground-truth annotations. Returns -1.0 if not provided.
    """
    if not relevant_chunk_ids:
        return -1.0
    retrieved_ids = {c.chunk_index for c in retrieved_chunks}
    relevant_set  = set(relevant_chunk_ids)
    recall = len(retrieved_ids & relevant_set) / len(relevant_set)
    return round(recall, 4)


# ─────────────────────────────────────────────────────────
# Manual test-set runner
# ─────────────────────────────────────────────────────────

def run_test_set(
    test_questions: List[str],
    vector_store,
    top_k: int = 4,
    export: bool = True,
) -> List[EvalRecord]:
    """
    Run a batch of test questions through the full RAG pipeline
    and record evaluation metrics for each.

    Parameters
    ----------
    test_questions : list of question strings
    vector_store   : FAISSVectorStore (must already be built)
    top_k          : retrieval count
    export         : if True, auto-export results to CSV

    Returns
    -------
    List[EvalRecord]
    """
    from src.retriever import retrieve
    from src.generator import generate_answer

    tracker = EvaluationTracker()

    for q in test_questions:
        try:
            ret = retrieve(q, vector_store, top_k=top_k)
            gen = generate_answer(q, ret.chunks)
            tracker.log(ret, gen, notes="test_set")
        except Exception as exc:
            logger.error(f"Test set error for query '{q}': {exc}")

    if export:
        tracker.export_csv()

    return tracker.records
