"""
retriever.py
------------
Takes a user query, embeds it, and retrieves the top-k most relevant
chunks from the FAISS vector store.

This module is the "R" in RAG — Retrieval-Augmented Generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import time
import logging

from src.embeddings import embed_query
from src.vector_store import FAISSVectorStore
from src.chunking import TextChunk
from src.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Holds the output of one retrieval operation.

    Attributes:
        query           : The original user question.
        chunks          : Retrieved TextChunk objects (top-k).
        scores          : Cosine similarity score for each chunk.
        retrieval_time  : Wall-clock time in seconds for retrieval.
        embedding_time  : Time taken to embed the query.
    """
    query: str
    chunks: List[TextChunk]
    scores: List[float]
    retrieval_time: float   # seconds (embedding + FAISS search)
    embedding_time: float   # seconds (just embedding)

    @property
    def avg_similarity(self) -> float:
        """Average cosine similarity of retrieved chunks."""
        if not self.scores:
            return 0.0
        return round(sum(self.scores) / len(self.scores), 4)

    @property
    def top_score(self) -> float:
        """Highest similarity score."""
        return round(self.scores[0], 4) if self.scores else 0.0


def retrieve(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    """
    Embed a query and retrieve the top-k matching chunks.

    Parameters
    ----------
    query        : The user's question (plain text).
    vector_store : A FAISSVectorStore that has been built / loaded.
    top_k        : Number of chunks to retrieve.

    Returns
    -------
    RetrievalResult with chunks, scores, and timing info.

    Raises
    ------
    RuntimeError : If the vector store is empty.
    ValueError   : If the query is empty.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    if not vector_store.is_ready:
        raise RuntimeError(
            "Vector store is not ready. Please upload and index PDFs first."
        )

    # ── 1. Embed the query ──────────────────────────────────
    t0 = time.perf_counter()
    query_embedding = embed_query(query)
    embed_time = time.perf_counter() - t0

    # ── 2. FAISS similarity search ─────────────────────────
    t1 = time.perf_counter()
    raw_results: List[Tuple[TextChunk, float]] = vector_store.search(
        query_embedding, top_k=top_k
    )
    search_time = time.perf_counter() - t1

    chunks = [r[0] for r in raw_results]
    scores = [r[1] for r in raw_results]

    total_time = embed_time + search_time

    logger.info(
        f"Retrieved {len(chunks)} chunks for query='{query[:60]}…' "
        f"in {total_time:.3f}s (embed={embed_time:.3f}s, search={search_time:.3f}s)"
    )

    return RetrievalResult(
        query=query,
        chunks=chunks,
        scores=scores,
        retrieval_time=round(total_time, 4),
        embedding_time=round(embed_time, 4),
    )
