"""
chunking.py
-----------
Splits PageDocument objects into smaller, overlapping text chunks.

Why chunking?
  Embedding models have a fixed token limit (~512 tokens for most sentence-
  transformers models). Chunking breaks long pages into manageable pieces
  so the embedding captures focused semantics rather than diluted page-level
  meaning.

Strategy: Recursive Character Text Splitter
  Tries to split on paragraphs → sentences → words → characters.
  This preserves linguistic structure as much as possible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import logging

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.pdf_loader import PageDocument

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    A single text chunk ready for embedding.

    Attributes:
        chunk_text  : The actual text of this chunk.
        page_number : Source page number (1-indexed).
        source      : Name of the source PDF file.
        chunk_index : Sequential index of this chunk (across all docs).
        metadata    : Passthrough metadata from the original page.
    """
    chunk_text: str
    page_number: int
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Splitter factory
# ─────────────────────────────────────────────────────────

def _make_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Build a RecursiveCharacterTextSplitter.

    The splitter tries these separators in order until the chunk is small enough:
      1. Double newline (paragraph break)
      2. Single newline
      3. Period + space (sentence boundary)
      4. Comma + space
      5. Single space
      6. Any character (last resort)
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def chunk_documents(
    page_docs: List[PageDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Split a list of PageDocuments into smaller TextChunk objects.

    Each chunk retains:
    - the source PDF filename
    - the page number it came from
    - its sequential index

    Parameters
    ----------
    page_docs    : list of PageDocument from pdf_loader
    chunk_size   : maximum characters per chunk
    chunk_overlap: number of characters shared between consecutive chunks

    Returns
    -------
    List[TextChunk]
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    all_chunks: List[TextChunk] = []
    global_idx = 0

    for doc in page_docs:
        # Split the page text into raw string pieces
        raw_pieces = splitter.split_text(doc.page_content)

        for piece in raw_pieces:
            piece = piece.strip()
            if not piece:           # skip empty strings
                continue
            all_chunks.append(TextChunk(
                chunk_text=piece,
                page_number=doc.page_number,
                source=doc.source,
                chunk_index=global_idx,
                metadata=doc.metadata,
            ))
            global_idx += 1

    logger.info(
        f"Chunking complete: {len(page_docs)} pages → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


def get_chunk_stats(chunks: List[TextChunk]) -> dict:
    """Return basic statistics about the chunks for display / debugging."""
    if not chunks:
        return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}

    lengths = [len(c.chunk_text) for c in chunks]
    return {
        "count":   len(chunks),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }
