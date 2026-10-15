"""Separate offline FastAPI prototype; production server.py is untouched."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[4]
SCRATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pdf_loader import PageDocument  # noqa: E402
from research.slm.scratch.pdf_qa.pipeline import (MAX_PDF_BYTES, PDFIndex,  # noqa: E402
    evidence_decision, validate_citation)
from research.slm.scratch.pdf_qa import hybrid_service  # noqa: E402
from research.slm.scratch.pdf_qa.hybrid import plan  # noqa: E402

LABEL = "Scratch Neural QA (Experimental)"
LIMITATION = ("Four-choice extractive pointer over automatically marked source sentences; "
              "deterministic checks are not semantic entailment or validated abstention.")
POINTER_PYTHON = SCRATCH / ".venv" / "bin" / "python"
WORKER = Path(__file__).with_name("pointer_worker.py")
app = FastAPI(title="Offline Scratch Neural PDF QA (Experimental)")
DOCUMENTS: dict[str, PDFIndex] = {}


class Ask(BaseModel):
    document_id: str
    question: str


class HybridAsk(Ask):
    mode: str = "qwen"


def upload_bytes(data: bytes, filename: str):
    if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise ValueError("A readable PDF file is required")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds 12 MB local prototype limit")
    safe_name = Path(filename).name[:200] or "document.pdf"
    # The UI sends the PDF as a raw body so neither FastAPI's multipart spool
    # nor this service writes a copy to disk. Extraction matches src.pdf_loader.
    try:
        import fitz
        with fitz.open(stream=data, filetype="pdf") as pdf:
            metadata = pdf.metadata or {}
            pages = [PageDocument(page_content=text, page_number=i + 1, source=safe_name,
                      metadata={"title": metadata.get("title", ""),
                                "author": metadata.get("author", ""), "total_pages": len(pdf)})
                     for i, page in enumerate(pdf)
                     if (text := page.get_text("text")).strip()]
    except Exception as exc:
        raise ValueError("PDF could not be read or has no selectable text; OCR is not available") from exc
    if not pages:
        raise ValueError("PDF has no selectable text; OCR is not available")
    if len(pages) > 120 or sum(len(page.page_content) for page in pages) > 1_000_000:
        raise ValueError("PDF exceeds the local limit of 120 text pages or one million extracted characters")
    for page in pages:
        page.source = safe_name
    index = PDFIndex.from_pages(data, safe_name, pages)
    DOCUMENTS[index.document_id] = index
    return {"document_id": index.document_id, "filename": safe_name,
            "pages": len(pages), "chunks": len(index.chunks), "mode": LABEL}


def pointer_request(question: str, candidates: list[dict]):
    if not POINTER_PYTHON.exists():
        raise RuntimeError("Frozen MLX pointer environment missing")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "NO_PROXY": "*", "TOKENIZERS_PARALLELISM": "false"}
    result = subprocess.run([str(POINTER_PYTHON), str(WORKER)],
                            input=json.dumps({"question": question, "candidates": candidates}),
                            text=True, capture_output=True, timeout=90, env=env, cwd=str(ROOT))
    try:
        output = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Pointer worker returned no structured result") from exc
    if result.returncode or output.get("status") == "ERROR":
        raise RuntimeError(output.get("error", "Pointer worker failed"))
    return output


def ask(index: PDFIndex, question: str):
    question = question.strip()
    if not question or len(question) > 300:
        raise ValueError("Enter a question of at most 300 characters")
    started = time.monotonic()
    retrieval = index.retrieve_candidates(question)
    base = {"mode": LABEL, "document_id": index.document_id, "filename": index.filename,
            "question": question, "experimental_limitation": LIMITATION,
            "retrieved": retrieval["retrieved"], "candidate_pool_count": len(retrieval["candidates"]),
            "retrieval_latency_ms": retrieval["retrieval_latency_ms"],
            "neural_prediction": None, "deterministic_decision": None,
            "rule_fallback_used": False}
    if len(retrieval["candidates"]) < 4:
        return {**base, "status": "INSUFFICIENT_CANDIDATES", "answer": None, "citation": None,
                "reason": "fewer_than_four_extracted_source_spans", "four_candidates": [],
                "total_latency_ms": (time.monotonic() - started) * 1000}
    neural = pointer_request(question, retrieval["candidates"])
    if neural["status"] == "INSUFFICIENT_CANDIDATES":
        return {**base, "status": "INSUFFICIENT_CANDIDATES", "answer": None, "citation": None,
                "reason": neural["reason"], "four_candidates": neural.get("four_candidates", []),
                "tokenizer_rejections": neural["rejected"],
                "total_latency_ms": (time.monotonic() - started) * 1000}
    four = neural["four_candidates"]
    selected = four[neural["candidate_index"]]
    if selected["value"] != neural["prediction"] or not validate_citation(index, selected):
        raise ValueError("Neural answer failed exact PDF source-span validation")
    status, reason = evidence_decision(index, question, selected, four)
    citation = {"filename": selected["source"], "document_id": index.document_id,
                "page_number": selected["page_number"], "sentence": selected["sentence"],
                "sentence_start": selected["sentence_start"], "sentence_end": selected["sentence_end"],
                "value_start": selected["value_start"], "value_end": selected["value_end"]}
    return {**base, "status": status, "answer": neural["prediction"] if status == "ANSWERED" else None,
            "neural_prediction": neural["prediction"], "neural_candidate_index": neural["candidate_index"],
            "neural_confidence_uncalibrated": neural["neural_confidence_uncalibrated"],
            "deterministic_decision": reason, "citation": citation if status == "ANSWERED" else None,
            "selected_source": citation, "four_candidates": four,
            "tokenizer_rejections": neural["rejected"], "model": neural["model"],
            "pointer_latency_ms": neural["pointer_latency_ms"], "peak_mlx_bytes": neural["peak_mlx_bytes"],
            "pointer_resource": neural["resource"], "total_latency_ms": (time.monotonic() - started) * 1000}


@app.get("/api/status")
def status():
    return {"mode": LABEL, "offline": True, "hybrid_modes": ["qwen", "hybrid"],
            "qwen_model_installed": (hybrid_service.SLM / ".cache" / hybrid_service.CONFIG["local_dir"] / "model.safetensors").is_file(),
            "documents": [
        {"document_id": d.document_id, "filename": d.filename, "pages": len(d.pages)}
        for d in DOCUMENTS.values()]}


@app.get("/api/model-readiness")
def model_readiness():
    """Read-only resource snapshot; full artifact checks still run on request."""
    current = hybrid_service.snapshot()
    problems = hybrid_service.reasons(current, current, loading=True)
    return {"resource_ready_now": not problems, "reasons": problems,
            "available_bytes": current["available_bytes"],
            "required_available_bytes": hybrid_service.CONFIG["min_preflight_available_bytes"],
            "pressure_free_percent": current["pressure_free_percent"],
            "swap_used_bytes": current["swap_used_bytes"],
            "disk_free_bytes": current["disk_free_bytes"],
            "note": "Snapshot only; full preflight and live guards run when asking."}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Legacy multipart API. The browser uses /api/upload-memory instead."""
    try:
        data = await file.read(MAX_PDF_BYTES + 1)
        return upload_bytes(data, file.filename or "upload.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/upload-memory")
async def upload_memory(request: Request):
    """Read a bounded raw PDF stream without writing or spooling a PDF copy."""
    if request.headers.get("content-type", "").split(";")[0].lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="Send a PDF with application/pdf content type")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_PDF_BYTES:
            raise HTTPException(status_code=422, detail="PDF exceeds 12 MB local prototype limit")
        data.extend(chunk)
    try:
        filename = unquote(request.headers.get("x-pdf-filename", "document.pdf"))
        return upload_bytes(bytes(data), filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str):
    if DOCUMENTS.pop(document_id, None) is None:
        raise HTTPException(status_code=404, detail="Document is not in this local session")
    return {"removed": True, "document_id": document_id}


@app.post("/api/ask")
def answer(req: Ask):
    index = DOCUMENTS.get(req.document_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Upload this PDF in the current local session first")
    try:
        return ask(index, req.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/hybrid/ask")
def hybrid_answer(req: HybridAsk):
    index = DOCUMENTS.get(req.document_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Upload this PDF in the current local session first")
    try:
        return hybrid_service.answer(index, req.question, req.mode, pointer_request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/evidence-preview")
def evidence_preview(req: Ask):
    """Expose exact selected passages when model inference cannot run."""
    index = DOCUMENTS.get(req.document_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Upload this PDF in the current local session first")
    if not req.question.strip() or len(req.question) > 300:
        raise HTTPException(status_code=422, detail="Enter a question of at most 300 characters")
    chosen = plan(index, req.question, max_sources=4)
    return {"label": "source_passages_only_not_a_model_answer", "intent": chosen["intent"],
            "sources": chosen["sources"], "coverage": chosen["coverage"]}


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).with_name("index.html")).read_text()
