"""
config.py
---------
Central configuration for the RAG-PDF-QA system.
All tunable parameters live here so you only need to change one file.
"""

import os
from pathlib import Path

# Load .env from the project root — must happen before any os.getenv() call.
# This works whether the module is imported first by app.py or test_pipeline.py.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data" / "sample_pdfs"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"

# ─────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────
# Options:
#   "sentence-transformers/all-MiniLM-L6-v2"  (small, fast, 384-dim)
#   "BAAI/bge-small-en-v1.5"                  (slightly better quality)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # must match the chosen model

# ─────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
DEFAULT_TOP_K = 4   # number of chunks to retrieve per query

# ─────────────────────────────────────────────
# Generator / LLM
# ─────────────────────────────────────────────
# Primary: local FLAN-T5 (no API key needed)
# Fallback: Groq API (set GROQ_API_KEY in .env)
LOCAL_LLM_MODEL   = "google/flan-t5-small"  # stable on CPU; upgrade to flan-t5-base if RAM > 8 GB
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.1   # keep low for factual grounding

# Groq fallback (optional)
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "openai/gpt-oss-20b"

# Gemini fallback (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"

# ─────────────────────────────────────────────
# FAISS index file names
# ─────────────────────────────────────────────
FAISS_INDEX_FILE = str(VECTOR_STORE_DIR / "faiss_index")

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
EVAL_RESULTS_FILE = str(ROOT_DIR / "evaluation_results.csv")
