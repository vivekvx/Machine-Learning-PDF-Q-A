"""
embeddings.py
-------------
Wraps the sentence-transformers model to produce dense vector embeddings.

Why sentence-transformers?
  - Pretrained on semantic textual similarity tasks
  - Produces fixed-size vectors (384-dim for MiniLM-L6-v2)
  - Runs fully locally — no API key required
  - Highly cited in NLP literature

Model: sentence-transformers/all-MiniLM-L6-v2
  - ~80 MB download
  - 384-dimensional output
  - Mean pooling over token embeddings
  - Trained on 1B sentence pairs
"""

from __future__ import annotations
from typing import List
import logging
import numpy as np

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — load the model once, reuse everywhere
_embedding_model = None


def _get_model():
    """
    Lazily load the SentenceTransformer model (only on first call).
    The model is cached as a module-level singleton so Streamlit
    re-runs don't reload it from disk every time.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.

    Parameters
    ----------
    texts        : list of strings to embed
    batch_size   : how many texts to process at once (affects memory usage)
    show_progress: if True, shows a tqdm progress bar

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
        Each row is the embedding for the corresponding text.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    model = _get_model()
    logger.info(f"Embedding {len(texts)} text(s) with batch_size={batch_size}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize → cosine similarity = dot product
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Parameters
    ----------
    query : the user question

    Returns
    -------
    np.ndarray of shape (1, embedding_dim)
    """
    return embed_texts([query])


def get_model_info() -> dict:
    """Return metadata about the loaded embedding model.

    Compatible with sentence-transformers 2.x through 6.x.
    In 6.x, get_sentence_embedding_dimension() was renamed to get_embedding_dimension().
    """
    model = _get_model()

    # Try new name first (sentence-transformers >= 6.x), then fall back to old
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    elif hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
    else:
        dim = 384  # hard fallback for all-MiniLM-L6-v2

    max_seq = getattr(model, "max_seq_length", 256)

    return {
        "model_name":     EMBEDDING_MODEL,
        "embedding_dim":  dim,
        "max_seq_length": max_seq,
    }
