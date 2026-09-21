"""
pdf_loader.py
-------------
Handles loading one or more PDF files and extracting their text.

We use PyMuPDF (fitz) as the primary loader because it:
- is faster than pypdf
- preserves page numbers reliably
- handles complex layouts better

Falls back to pypdf if PyMuPDF is not installed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageDocument:
    """
    Represents a single page extracted from a PDF.

    Attributes:
        page_content : The raw text of the page.
        page_number  : 1-indexed page number inside the PDF.
        source       : Name of the source PDF file.
        metadata     : Extra metadata dict (e.g., title, author).
    """
    page_content: str
    page_number: int
    source: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Low-level extraction helpers
# ─────────────────────────────────────────────────────────

def _load_with_pymupdf(pdf_path: Path) -> List[PageDocument]:
    """Extract text page-by-page using PyMuPDF (pymupdf)."""
    try:
        import pymupdf as fitz  # modern import (PyMuPDF >= 1.24)
    except ImportError:
        import fitz              # older PyMuPDF fallback

    docs: List[PageDocument] = []
    with fitz.open(str(pdf_path)) as pdf:
        meta = pdf.metadata or {}
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            text = page.get_text("text")  # plain text extraction
            if text.strip():              # skip blank pages
                docs.append(PageDocument(
                    page_content=text,
                    page_number=page_idx + 1,   # 1-indexed
                    source=pdf_path.name,
                    metadata={
                        "title":  meta.get("title", ""),
                        "author": meta.get("author", ""),
                        "total_pages": len(pdf),
                    },
                ))
    return docs


def _load_with_pypdf(pdf_path: Path) -> List[PageDocument]:
    """Fallback extractor using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    docs: List[PageDocument] = []
    total = len(reader.pages)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(PageDocument(
                page_content=text,
                page_number=idx + 1,
                source=pdf_path.name,
                metadata={"total_pages": total},
            ))
    return docs


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def load_pdf(pdf_path: str | Path) -> List[PageDocument]:
    """
    Load a single PDF file and return a list of PageDocument objects.

    Tries PyMuPDF first; falls back to pypdf automatically.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    List[PageDocument]
        One PageDocument per non-blank page.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no text could be extracted (e.g., scanned / image-only PDF).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Loading PDF: {pdf_path.name}")

    # Try PyMuPDF first (preferred)
    try:
        docs = _load_with_pymupdf(pdf_path)
        logger.info(f"  Loaded {len(docs)} pages via PyMuPDF")
    except ImportError:
        logger.warning("PyMuPDF not installed — falling back to pypdf")
        docs = _load_with_pypdf(pdf_path)
        logger.info(f"  Loaded {len(docs)} pages via pypdf")

    if not docs:
        raise ValueError(
            f"No text extracted from '{pdf_path.name}'. "
            "The PDF may be scanned or image-only."
        )

    return docs


def load_pdfs(pdf_paths: List[str | Path]) -> List[PageDocument]:
    """
    Load multiple PDF files and combine their pages into one list.

    Parameters
    ----------
    pdf_paths : list of str | Path

    Returns
    -------
    List[PageDocument]
    """
    all_docs: List[PageDocument] = []
    for path in pdf_paths:
        try:
            all_docs.extend(load_pdf(path))
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"Skipping '{path}': {exc}")
    return all_docs
