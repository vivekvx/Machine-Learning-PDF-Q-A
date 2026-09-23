"""
server.py
---------
FastAPI High-Performance Backend for RAG PDF Q&A System.
Production-grade REST API + HTML5/CSS3 Single Page Application (SPA).

Endpoints:
  GET  /              -> Serves the single-page application (SPA)
  POST /api/upload    -> Multipart PDF upload & vector indexing
  POST /api/query     -> Grounded RAG query execution
  GET  /api/status    -> Index & KB status
  POST /api/reset     -> Clear vector index & session state
  GET  /api/export-csv-> Export evaluation CSV log
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader        import load_pdfs
from src.chunking          import chunk_documents, get_chunk_stats
from src.vector_store      import FAISSVectorStore
from src.retriever         import retrieve
from src.generator         import generate_answer
from src.evaluation        import EvaluationTracker
from src.query_intelligence import classify_and_expand
from src.config            import (
    DEFAULT_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, LOCAL_LLM_MODEL, GROQ_MODEL,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI App Initialisation ───────────────────────────────────────────────
app = FastAPI(
    title="A Retrieval-Augmented Question Answering System for PDF Documents Using Machine Learning",
    description="Research-Grade Retrieval-Augmented Generation REST API",
    version="2.0.0",
)

# Enable CORS for cross-origin browser fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared System State
vector_store = FAISSVectorStore()
eval_tracker = EvaluationTracker()
indexed_files: List[str] = []
chunk_stats_data = {}

# Load existing FAISS index from disk if present
if not vector_store.is_ready:
    vector_store.load()


class QueryRequest(BaseModel):
    question: str


# ── REST API Endpoints ───────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Returns vector store status and indexed document metadata."""
    return {
        "is_ready": vector_store.is_ready,
        "total_vectors": vector_store.total_vectors if vector_store.is_ready else 0,
        "total_chunks": len(vector_store.chunks) if vector_store.is_ready else 0,
        "indexed_files": indexed_files,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": GROQ_MODEL,
    }


@app.post("/api/upload")
async def upload_and_index(files: List[UploadFile] = File(...)):
    """
    Accepts multipart PDF upload(s), extracts text, chunks, embeds,
    and updates the FAISS vector index. Zero timeout, streaming byte handler.
    """
    global indexed_files, chunk_stats_data

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(f"Receiving {len(pdf_files)} PDF file(s) for indexing...")

    try:
        tmp_dir = tempfile.mkdtemp()
        tmp_paths = []
        
        for file in pdf_files:
            path = Path(tmp_dir) / file.filename
            contents = await file.read()
            path.write_bytes(contents)
            tmp_paths.append(path)

        # Extract & Index
        page_docs = load_pdfs(tmp_paths)
        if not page_docs:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the PDF. The file may be scanned or image-only."
            )

        chunks = chunk_documents(page_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        stats = get_chunk_stats(chunks)

        vector_store.build_from_chunks(chunks, show_progress=False)
        vector_store.save()

        indexed_files = [f.filename for f in pdf_files]
        chunk_stats_data = stats

        logger.info(f"Successfully indexed {len(pdf_files)} files ({stats['count']} chunks).")

        return {
            "status": "success",
            "message": f"Successfully indexed {len(pdf_files)} file(s).",
            "indexed_files": indexed_files,
            "total_chunks": stats["count"],
            "total_vectors": vector_store.total_vectors,
            "avg_chunk_len": stats["avg_len"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload & Index failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(exc)}")


@app.post("/api/query")
def execute_query(req: QueryRequest):
    """
    Executes a RAG query with query intelligence:
    1. Classify intent & expand queries
    2. Multi-query FAISS retrieval with deduplication
    3. Generates grounded answer via Groq LLM
    4. Logs evaluation metrics
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not vector_store.is_ready:
        raise HTTPException(status_code=400, detail="No active document index found. Please upload a PDF first.")

    try:
        question = req.question.strip()

        # 1. Classify intent & expand queries
        intel = classify_and_expand(question, index_size=vector_store.total_vectors)
        logger.info(
            f"[Query] intent={intel.intent}, top_k={intel.recommended_top_k}, "
            f"expanded={len(intel.expanded_queries)}, is_broad={intel.is_broad}"
        )

        # 2. Multi-query retrieval
        ret_result = retrieve(
            question,
            vector_store,
            top_k=intel.recommended_top_k,
            expanded_queries=intel.expanded_queries,
            is_broad=intel.is_broad,
            query_type=intel.intent,
        )

        # 3. Generate Grounded Answer
        gen_result = generate_answer(
            question,
            ret_result.chunks,
            prefer_local=False,
            scores=ret_result.scores,
            intent=intel.intent,
            query_expanded=len(intel.expanded_queries) > 1,
            fallback_triggered=ret_result.fallback_triggered,
        )
        logger.info(
            f"[Query] question={question!r} query_type={intel.intent} "
            f"pages={[c.page_number for c in ret_result.chunks]} "
            f"scores={[round(s, 4) for s in ret_result.scores]} "
            f"model={gen_result.model_used} latency="
            f"{ret_result.retrieval_time + gen_result.generation_time:.3f}s"
        )

        # 4. Log Performance Evaluation
        eval_tracker.log(ret_result, gen_result)

        # 5. Format Chunk Snippets for JSON
        chunk_snippets = [
            {
                "source": c.source,
                "page_number": c.page_number,
                "score": round(score, 4),
                "section_heading": c.metadata.get("section_heading", "Unlabelled section"),
                "text": c.chunk_text,
            }
            for c, score in zip(ret_result.chunks, ret_result.scores)
        ]

        return {
            "question": question,
            "answer": gen_result.answer,
            "source_pages": gen_result.source_pages,
            "source_files": gen_result.source_files,
            "model_used": gen_result.model_used,
            "retrieval_time": ret_result.retrieval_time,
            "generation_time": gen_result.generation_time,
            "total_time": round(ret_result.retrieval_time + gen_result.generation_time, 3),
            "top_similarity": ret_result.top_score,
            "avg_similarity": ret_result.avg_similarity,
            "chunks": chunk_snippets,
            # Intelligence metadata
            "intent": intel.intent,
            "query_expanded": gen_result.query_expanded,
            "fallback_triggered": gen_result.fallback_triggered,
            "expanded_queries": intel.expanded_queries,
        }

    except Exception as exc:
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=f"Query error: {str(exc)}")


@app.post("/api/reset")
def reset_system():
    """Resets vector store and session state."""
    global indexed_files, chunk_stats_data
    vector_store.clear()
    eval_tracker.clear()
    indexed_files = []
    chunk_stats_data = {}
    return {"status": "success", "message": "Knowledge base reset."}


@app.get("/api/export-csv")
def export_csv():
    """Returns downloadable CSV file of evaluation logs."""
    path = eval_tracker.export_csv()
    if path and Path(path).exists():
        return FileResponse(path, filename="rag_evaluation_results.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="No evaluation logs to export.")


# Serve Single Page Application (SPA)
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Index file missing. Please build static/index.html</h1>"
