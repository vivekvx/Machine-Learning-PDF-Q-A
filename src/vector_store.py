"""
vector_store.py
---------------
Manages the FAISS vector index: build, save, load, and search.

FAISS (Facebook AI Similarity Search):
  - In-memory vector database
  - IndexFlatIP = exact dot-product search (inner product ≡ cosine similarity
    when vectors are L2-normalised, which our embed_texts does by default)
  - Very fast for up to ~1M vectors on CPU
  - Deterministic results (no approximation)

Index structure stored on disk:
  vector_store/faiss_index.index   ← FAISS binary index
  vector_store/faiss_index.meta    ← chunk metadata (pickle)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import pickle

import numpy as np
import faiss

from src.chunking import TextChunk
from src.embeddings import embed_texts
from src.config import FAISS_INDEX_FILE, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Wraps a FAISS index together with the TextChunk metadata.

    Workflow:
      1. build_from_chunks()  → embed all chunks and add to index
      2. save()               → persist to disk
      3. load()               → restore from disk
      4. search()             → retrieve top-k similar chunks
    """

    def __init__(self):
        # FAISS IndexFlatIP: exact inner-product search
        # Cosine similarity = inner product when vectors are L2-normalised
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[TextChunk] = []          # parallel list of metadata
        self._index_path = FAISS_INDEX_FILE + ".index"
        self._meta_path  = FAISS_INDEX_FILE + ".meta"

    # ──────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────

    def build_from_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """
        Embed all chunks and insert them into the FAISS index.

        Parameters
        ----------
        chunks       : list of TextChunk from chunking.py
        batch_size   : embedding batch size
        show_progress: show tqdm progress bar during embedding
        """
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunk list.")

        texts = [c.chunk_text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")

        embeddings = embed_texts(texts, batch_size=batch_size, show_progress=show_progress)
        # embeddings shape: (N, EMBEDDING_DIM), already L2-normalised

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner-product index
        self.index.add(embeddings)
        self.chunks = list(chunks)

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, dim={dim}"
        )

    # ──────────────────────────────────────────────────────
    # Persist / restore
    # ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Save the FAISS index and chunk metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_from_chunks() first.")

        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Index saved → {self._index_path}")

    def load(self) -> bool:
        """
        Load a previously saved index from disk.

        Returns True if loaded successfully, False if no saved index found.
        """
        if not Path(self._index_path).exists():
            logger.info("No saved FAISS index found.")
            return False

        self.index = faiss.read_index(self._index_path)
        with open(self._meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(
            f"Index loaded: {self.index.ntotal} vectors, "
            f"{len(self.chunks)} chunks"
        )
        return True

    def clear(self) -> None:
        """Reset the index and remove files from disk."""
        self.index  = None
        self.chunks = []
        for p in [self._index_path, self._meta_path]:
            if Path(p).exists():
                Path(p).unlink()
        logger.info("Vector store cleared.")

    # ──────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
    ) -> List[Tuple[TextChunk, float]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray of shape (1, dim) or (dim,)
        top_k           : number of results to return

        Returns
        -------
        List of (TextChunk, similarity_score) tuples, sorted by score desc.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Index PDFs first.")

        # Ensure correct shape
        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)   # shapes: (1, k)

        results: List[Tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for padding
                continue
            results.append((self.chunks[idx], float(score)))

        return results   # already sorted descending by FAISS

    # ──────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the index has at least one vector."""
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def __repr__(self):
        return (
            f"FAISSVectorStore("
            f"vectors={self.total_vectors}, "
            f"chunks={len(self.chunks)})"
        )
