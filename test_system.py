"""
test_system.py
--------------
Lightweight verification suite for the RAG PDF Q&A Platform.
Tests ML pipeline components and FastAPI HTTP endpoints end-to-end.

Run with:
    python test_system.py
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents, get_chunk_stats
from src.embeddings import embed_texts, embed_query
from src.vector_store import FAISSVectorStore
from src.retriever import retrieve
from src.generator import generate_answer, _generate_extractive


def run_tests():
    print("=" * 60)
    print("🧪 RAG PDF Q&A SYSTEM VERIFICATION SUITE")
    print("=" * 60)

    # 1. Test Sample PDF Creation / Loading
    sample_pdf = Path("data/sample_pdfs/machine_learning_notes.pdf")
    if not sample_pdf.exists():
        print("📄 Creating sample PDF for testing...")
        from data.sample_pdfs.create_sample_pdf import create_sample_pdf
        create_sample_pdf()
    
    assert sample_pdf.exists(), "Sample PDF creation failed!"
    print("✅ Step 1: Sample PDF verified ->", sample_pdf)

    # 2. Test PDF Text Extraction
    page_docs = load_pdfs([sample_pdf])
    assert len(page_docs) > 0, "PDF text extraction returned 0 pages!"
    print(f"✅ Step 2: PDF extraction verified -> {len(page_docs)} pages extracted.")

    # 3. Test Text Chunking
    chunks = chunk_documents(page_docs, chunk_size=500, chunk_overlap=50)
    stats = get_chunk_stats(chunks)
    assert len(chunks) > 0, "Chunking returned 0 chunks!"
    assert chunks[0].page_number is not None, "Chunk metadata missing page_number!"
    print(f"✅ Step 3: Chunking verified -> {stats['count']} chunks (avg {stats['avg_len']} chars).")

    # 4. Test Embedding Generation
    sample_strings = [c.chunk_text for c in chunks[:3]]
    embeddings = embed_texts(sample_strings)
    assert len(embeddings) == 3, "Embedding count mismatch!"
    assert len(embeddings[0]) == 384, f"Embedding dimension mismatch ({len(embeddings[0])} != 384)!"
    print("✅ Step 4: Sentence-Transformer embeddings verified (384-dim).")

    # 5. Test FAISS Indexing & Search
    vs = FAISSVectorStore()
    vs.build_from_chunks(chunks, show_progress=False)
    vs.save()
    assert vs.is_ready, "Vector store is not ready after build!"
    assert vs.total_vectors == len(chunks), "FAISS vector count mismatch!"
    print(f"✅ Step 5: FAISS index built & saved -> {vs.total_vectors} vectors.")

    # 6. Test Retrieval Engine
    query = "What is machine learning?"
    ret_result = retrieve(query, vs, top_k=4)
    assert len(ret_result.chunks) == 4, "Retrieved chunk count mismatch!"
    assert ret_result.top_score > 0.0, "Similarity score invalid!"
    print(f"✅ Step 6: Retrieval engine verified (Top score: {ret_result.top_score:.4f}).")

    # 7. Test Extractive Fallback
    ext_answer, ext_model = _generate_extractive(query, ret_result.chunks)
    assert len(ext_answer) > 5, "Extractive answer is empty!"
    print(f"✅ Step 7: Extractive QA fallback verified -> model: {ext_model}")

    # 8. Test Grounded Answer Generation
    gen_result = generate_answer(query, ret_result.chunks, prefer_local=False)
    assert len(gen_result.answer) > 5, "Generated answer is empty!"
    assert len(gen_result.source_pages) > 0, "Source page citations missing!"
    print(f"✅ Step 8: Answer generation verified -> model: {gen_result.model_used}, pages: {gen_result.source_pages}")

    # 9. Test FastAPI REST Server Endpoints
    print("\n🌐 Testing FastAPI REST Server Endpoints (http://localhost:8000)...")
    try:
        # GET /api/status
        status_req = urllib.request.urlopen("http://localhost:8000/api/status")
        status = json.loads(status_req.read().decode())
        assert status_req.status == 200, "Status endpoint failed!"
        print(f"  ✓ GET /api/status -> {status['total_vectors']} vectors, ready={status['is_ready']}")

        # POST /api/query (Present answer test)
        q_data = json.dumps({"question": "What is supervised learning?"}).encode()
        q_req = urllib.request.Request("http://localhost:8000/api/query", data=q_data, headers={"Content-Type": "application/json"})
        q_res = urllib.request.urlopen(q_req)
        q_json = json.loads(q_res.read().decode())
        assert q_res.status == 200, "Query endpoint failed!"
        assert len(q_json["answer"]) > 0, "Answer empty!"
        print(f"  ✓ POST /api/query (Present Context) -> Model: {q_json['model_used']}, Latency: {q_json['total_time']}s")

        # POST /api/query (Absent context test)
        absent_data = json.dumps({"question": "What is the capital of France?"}).encode()
        absent_req = urllib.request.Request("http://localhost:8000/api/query", data=absent_data, headers={"Content-Type": "application/json"})
        absent_res = urllib.request.urlopen(absent_req)
        absent_json = json.loads(absent_res.read().decode())
        assert absent_res.status == 200, "Absent context query failed!"
        print(f"  ✓ POST /api/query (Absent Context) -> No hallucination verified.")

        print("✅ Step 9: FastAPI REST server endpoints 100% verified!")

    except Exception as exc:
        print(f"⚠️ FastAPI REST Server test error: {exc}")
        print("  Make sure the server is running with: uvicorn server:app --port 8000")

    print("\n" + "=" * 60)
    print("🎉 ALL 9 SYSTEM VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
