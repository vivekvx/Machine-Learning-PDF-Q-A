"""
retriever.py
------------
Takes a user query, embeds it, and retrieves the top-k most relevant
chunks from the FAISS vector store.

v2 improvements:
  - Supports multi-query retrieval (query expansion)
  - Deduplicates chunks by chunk_index across expanded queries
  - Fallback expansion if initial retrieval has low confidence
  - No hard similarity cutoff for broad questions (let the LLM decide relevance)
  - Detailed debug logging for every retrieval step
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import logging

from src.embeddings import embed_query
from src.vector_store import FAISSVectorStore
from src.chunking import TextChunk
from src.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

# Similarity threshold below which we trigger fallback expansion
_LOW_CONFIDENCE_THRESHOLD = 0.25


@dataclass
class RetrievalResult:
    """
    Holds the output of one retrieval operation.

    Attributes:
        query           : The original user question.
        chunks          : Retrieved TextChunk objects (deduplicated, top-k).
        scores          : Cosine similarity score for each chunk.
        retrieval_time  : Wall-clock time in seconds for retrieval.
        embedding_time  : Time taken to embed the query.
        expanded_queries: All queries used (original + expanded).
        fallback_triggered : True if fallback expansion was used.
    """
    query: str
    chunks: List[TextChunk]
    scores: List[float]
    retrieval_time: float
    embedding_time: float
    expanded_queries: List[str] = field(default_factory=list)
    fallback_triggered: bool = False

    @property
    def avg_similarity(self) -> float:
        if not self.scores:
            return 0.0
        return round(sum(self.scores) / len(self.scores), 4)

    @property
    def top_score(self) -> float:
        return round(self.scores[0], 4) if self.scores else 0.0


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

def _search_single(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int,
) -> List[Tuple[TextChunk, float]]:
    """Embed one query and search FAISS. Returns (chunk, score) pairs."""
    q_emb = embed_query(query)
    return vector_store.search(q_emb, top_k=top_k)


def _multi_query_retrieve(
    queries: List[str],
    vector_store: FAISSVectorStore,
    top_k: int,
) -> Tuple[List[TextChunk], List[float]]:
    """
    Run FAISS search for each query in the list.
    Deduplicate results by chunk_index, keeping the highest score per chunk.
    Return chunks sorted by score descending, capped at top_k.
    """
    seen: dict[int, Tuple[TextChunk, float]] = {}  # chunk_index → (chunk, best_score)

    for q in queries:
        try:
            results = _search_single(q, vector_store, top_k)
            for chunk, score in results:
                idx = chunk.chunk_index
                if idx not in seen or score > seen[idx][1]:
                    seen[idx] = (chunk, score)
        except Exception as exc:
            logger.warning(f"[Retriever] Sub-query failed for '{q[:50]}': {exc}")

    if not seen:
        return [], []

    # Sort by score descending and cap at top_k
    sorted_items = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]
    chunks = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    return chunks, scores


def _section_aware_queries(intent: str) -> List[str]:
    """Return paper-structure queries that improve coverage for broad questions."""
    if intent == "broad_summary":
        return [
            "abstract research purpose overview",
            "introduction background research question",
            "conclusion discussion implications summary",
        ]
    if intent == "methodology":
        return ["methods methodology study design data analysis"]
    if intent in {"findings", "metrics"}:
        return ["results findings discussion conclusion"]
    return []


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = DEFAULT_TOP_K,
    expanded_queries: Optional[List[str]] = None,
    is_broad: bool = False,
    query_type: str = "unknown",
) -> RetrievalResult:
    """
    Embed a query and retrieve the top-k matching chunks.

    Parameters
    ----------
    query            : The user's question (plain text).
    vector_store     : A FAISSVectorStore that has been built / loaded.
    top_k            : Number of chunks to retrieve.
    expanded_queries : Pre-computed expanded queries from query_intelligence.
                       If None, only the original query is used.
    is_broad         : If True, adds paper-structure queries and never drops useful
                       chunks solely because their similarity is low.
    query_type       : Query taxonomy label from query_intelligence.

    Returns
    -------
    RetrievalResult with chunks, scores, and timing info.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    if not vector_store.is_ready:
        raise RuntimeError(
            "Vector store is not ready. Please upload and index PDFs first."
        )

    t0 = time.perf_counter()

    # Build the list of queries to run
    all_queries: List[str] = list(expanded_queries) if expanded_queries else [query]
    # Always ensure original query is first
    if query not in all_queries:
        all_queries = [query] + all_queries
    if is_broad:
        all_queries.extend(q for q in _section_aware_queries(query_type) if q not in all_queries)

    logger.info(
        f"[Retriever] query='{query[:70]}' | top_k={top_k} | "
        f"expanded_queries={len(all_queries)} | is_broad={is_broad}"
    )
    for i, q in enumerate(all_queries):
        logger.info(f"  [Q{i+1}] {q[:80]}")

    # ── Multi-query retrieval ──────────────────────────────
    embed_t0 = time.perf_counter()
    chunks, scores = _multi_query_retrieve(all_queries, vector_store, top_k)
    embed_time = time.perf_counter() - embed_t0

    fallback_triggered = False

    # ── Fallback: low confidence → expand further ──────────
    if is_broad and (not scores or scores[0] < _LOW_CONFIDENCE_THRESHOLD):
        logger.info(
            f"[Retriever] Low confidence (top={scores[0] if scores else 0:.3f}). "
            "Triggering fallback: broader retrieval with higher top_k."
        )
        fallback_triggered = True
        fallback_top_k = min(top_k + 6, vector_store.total_vectors)
        # Add generic academic expansion queries
        fallback_extras = [
            "introduction background overview",
            "conclusion discussion summary findings",
            "methodology approach methods",
            "results evidence outcomes",
            "abstract purpose aim objective",
        ]
        all_fallback = all_queries + [q for q in fallback_extras if q not in all_queries]
        chunks, scores = _multi_query_retrieve(all_fallback, vector_store, fallback_top_k)

    total_time = time.perf_counter() - t0

    # ── Debug logging ──────────────────────────────────────
    top_score = scores[0] if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(
        f"[Retriever] Retrieved {len(chunks)} chunks in {total_time:.3f}s "
        f"(top_score={top_score:.4f}, avg={avg_score:.4f})"
    )
    for i, (c, s) in enumerate(zip(chunks, scores)):
        logger.info(
            f"  [Chunk {i+1}] page={c.page_number}, score={s:.4f}, "
            f"section={c.metadata.get('section_heading', 'Unlabelled section')!r}, "
            f"text='{c.chunk_text[:300].replace(chr(10), ' ')}...'"
        )

    return RetrievalResult(
        query=query,
        chunks=chunks,
        scores=scores,
        retrieval_time=round(total_time, 4),
        embedding_time=round(embed_time, 4),
        expanded_queries=all_queries,
        fallback_triggered=fallback_triggered,
    )
