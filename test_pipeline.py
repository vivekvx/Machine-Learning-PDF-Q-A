#!/usr/bin/env python3
"""
test_pipeline.py  — End-to-end integration test for the RAG PDF Q&A pipeline.

Uses a shared State object so all tests can access previously computed objects
(page_docs, chunks, vector_store) even if an earlier test fails partially.

Run with:  venv/bin/python3 test_pipeline.py
"""

import sys
import time
import traceback
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Shared state (populated by each test section) ─────────────────────────
class S:
    page_docs = []
    chunks    = []
    vs        = None   # FAISSVectorStore instance

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name: str, fn):
    """Run fn(), print result, record pass/fail. Returns True on success."""
    try:
        info = fn()
        print(f"{PASS}  {name}")
        if info:
            print(f"       → {info}")
        results.append((name, True, ""))
        return True
    except Exception as exc:
        print(f"{FAIL}  {name}")
        print(f"       → {exc}")
        traceback.print_exc()
        results.append((name, False, str(exc)))
        return False

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  RAG PDF Q&A — End-to-End Integration Test Suite")
print("="*62 + "\n")

# ── 0. Config ──────────────────────────────────────────────────────────────
print("── Section 0: Config ──")
def t0_config():
    from src.config import (EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
                            DEFAULT_TOP_K, LOCAL_LLM_MODEL)
    return (f"embed={EMBEDDING_MODEL}, chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, "
            f"k={DEFAULT_TOP_K}, llm={LOCAL_LLM_MODEL}")
check("Config values load correctly", t0_config)

# ── 1. Sample PDF ──────────────────────────────────────────────────────────
print("\n── Section 1: Sample PDF ──")
SAMPLE_PDF = Path("data/sample_pdfs/machine_learning_notes.pdf")

def t1_pdf_exists():
    if not SAMPLE_PDF.exists():
        raise FileNotFoundError(
            f"{SAMPLE_PDF} missing. Run: "
            "venv/bin/pip install reportlab && "
            "venv/bin/python3 data/sample_pdfs/create_sample_pdf.py"
        )
    return f"size={SAMPLE_PDF.stat().st_size // 1024} KB"
check("Sample PDF file exists", t1_pdf_exists)

# ── 2. PDF Loader ──────────────────────────────────────────────────────────
print("\n── Section 2: PDF Loader ──")

def t2_load_pdf():
    from src.pdf_loader import load_pdf
    S.page_docs = load_pdf(SAMPLE_PDF)
    assert len(S.page_docs) >= 1, "No pages extracted"
    return f"{len(S.page_docs)} pages extracted"
check("PDF extraction succeeds", t2_load_pdf)

def t2_page_numbers():
    assert S.page_docs, "No pages loaded"
    pages = [d.page_number for d in S.page_docs]
    assert pages[0] == 1, f"First page should be 1, got {pages[0]}"
    assert pages == sorted(pages), "Pages not ordered"
    return f"pages={pages}"
check("Page numbers are 1-indexed and sequential", t2_page_numbers)

def t2_metadata():
    assert S.page_docs, "No pages loaded"
    for d in S.page_docs:
        assert d.source == SAMPLE_PDF.name
        assert d.page_content.strip()
    return f"source='{S.page_docs[0].source}'"
check("PageDocument has source and non-empty content", t2_metadata)

def t2_missing_file():
    from src.pdf_loader import load_pdf
    try:
        load_pdf("this_does_not_exist.pdf")
        raise AssertionError("Should raise FileNotFoundError")
    except FileNotFoundError:
        return "FileNotFoundError raised correctly"
check("Missing PDF raises FileNotFoundError", t2_missing_file)

# ── 3. Chunking ────────────────────────────────────────────────────────────
print("\n── Section 3: Chunking ──")

def t3_chunk():
    from src.chunking import chunk_documents, get_chunk_stats
    # Use a fallback if page_docs empty
    docs = S.page_docs
    if not docs:
        raise RuntimeError("page_docs is empty — PDF loader must have failed")
    S.chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(S.chunks) > 0, "No chunks produced"
    stats = get_chunk_stats(S.chunks)
    return (f"{stats['count']} chunks, avg={stats['avg_len']} chars, "
            f"min={stats['min_len']}, max={stats['max_len']}")
check("Chunking produces chunks from page_docs", t3_chunk)

def t3_chunk_meta():
    if not S.chunks:
        raise RuntimeError("No chunks available")
    for c in S.chunks:
        assert c.page_number >= 1, f"Bad page_number: {c.page_number}"
        assert c.source,           "Missing source"
        assert c.chunk_text.strip(),"Empty chunk_text"
        assert c.chunk_index >= 0,  "Bad chunk_index"
    return f"All {len(S.chunks)} chunks have valid metadata"
check("Chunks carry page_number, source, chunk_index", t3_chunk_meta)

# ── 4. Embeddings ──────────────────────────────────────────────────────────
print("\n── Section 4: Embeddings ──")

def t4_embed_texts():
    from src.embeddings import embed_texts, get_model_info
    import numpy as np
    texts = [c.chunk_text for c in S.chunks[:3]] if S.chunks else [
        "machine learning test", "neural network", "FAISS index"]
    embs = embed_texts(texts)
    assert embs.shape == (len(texts), 384), f"Wrong shape: {embs.shape}"
    norms = np.linalg.norm(embs, axis=1)
    assert all(abs(n - 1.0) < 0.01 for n in norms), f"Not normalised: {norms}"
    info = get_model_info()
    return f"shape={embs.shape}, model={info['model_name']}, dim={info['embedding_dim']}"
check("Embeddings: correct shape (N,384), L2-normalised", t4_embed_texts)

def t4_embed_query():
    from src.embeddings import embed_query
    q_emb = embed_query("What is machine learning?")
    assert q_emb.shape == (1, 384), f"Wrong shape: {q_emb.shape}"
    return f"query embedding shape={q_emb.shape}"
check("Query embedding returns shape (1, 384)", t4_embed_query)

# ── 5. Vector Store ────────────────────────────────────────────────────────
print("\n── Section 5: FAISS Vector Store ──")

def t5_build():
    from src.vector_store import FAISSVectorStore
    if not S.chunks:
        raise RuntimeError("chunks empty — chunking must have failed")
    S.vs = FAISSVectorStore()
    t0 = time.perf_counter()
    S.vs.build_from_chunks(S.chunks, show_progress=False)
    elapsed = time.perf_counter() - t0
    assert S.vs.is_ready
    assert S.vs.total_vectors == len(S.chunks)
    return f"{S.vs.total_vectors} vectors in {elapsed:.2f}s"
check("FAISS index builds from all chunks", t5_build)

def t5_save():
    from src.config import FAISS_INDEX_FILE
    S.vs.save()
    assert Path(FAISS_INDEX_FILE + ".index").exists(), ".index missing"
    assert Path(FAISS_INDEX_FILE + ".meta").exists(),  ".meta missing"
    kb = Path(FAISS_INDEX_FILE + ".index").stat().st_size // 1024
    return f"faiss_index.index={kb} KB, .meta exists"
check("FAISS index saves to disk", t5_save)

def t5_load():
    from src.vector_store import FAISSVectorStore
    vs2 = FAISSVectorStore()
    ok = vs2.load()
    assert ok and vs2.is_ready
    assert vs2.total_vectors == S.vs.total_vectors
    return f"Loaded {vs2.total_vectors} vectors from disk"
check("FAISS index loads from disk", t5_load)

def t5_clear():
    from src.config import FAISS_INDEX_FILE
    from src.vector_store import FAISSVectorStore
    tmp_vs = FAISSVectorStore()
    tmp_vs.build_from_chunks(S.chunks, show_progress=False)
    tmp_vs.save()
    tmp_vs.clear()
    assert not tmp_vs.is_ready
    assert not Path(FAISS_INDEX_FILE + ".index").exists()
    # Restore S.vs and save it again for downstream tests
    S.vs.save()
    return "Cleared state and files correctly; restored for downstream tests"
check("FAISS index clear removes state and files", t5_clear)

# ── 6. Retrieval ───────────────────────────────────────────────────────────
print("\n── Section 6: Retrieval ──")

def t6_basic():
    from src.retriever import retrieve
    ret = retrieve("What is machine learning?", S.vs, top_k=4)
    assert len(ret.chunks) > 0
    assert len(ret.scores) == len(ret.chunks)
    assert ret.retrieval_time > 0
    assert ret.embedding_time > 0
    assert -1.0 <= ret.avg_similarity <= 1.0
    return (f"chunks={len(ret.chunks)}, top_score={ret.top_score:.4f}, "
            f"avg_sim={ret.avg_similarity:.4f}, time={ret.retrieval_time:.4f}s")
check("Retrieval returns chunks, scores, and timing", t6_basic)

def t6_page_numbers():
    from src.retriever import retrieve
    ret = retrieve("What is overfitting?", S.vs, top_k=4)
    pages = [c.page_number for c in ret.chunks]
    assert all(p >= 1 for p in pages), f"Bad page numbers: {pages}"
    return f"retrieved from pages: {pages}"
check("Retrieved chunks have valid page numbers", t6_page_numbers)

def t6_top_k():
    from src.retriever import retrieve
    r3 = retrieve("neural networks", S.vs, top_k=3)
    r1 = retrieve("neural networks", S.vs, top_k=1)
    assert len(r3.chunks) == 3
    assert len(r1.chunks) == 1
    return "top_k=3→3 chunks, top_k=1→1 chunk"
check("top_k parameter controls retrieved count", t6_top_k)

def t6_empty_query():
    from src.retriever import retrieve
    try:
        retrieve("", S.vs, top_k=4)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        return "ValueError raised for empty query"
check("Empty query raises ValueError", t6_empty_query)

def t6_similarity_sorted():
    from src.retriever import retrieve
    ret = retrieve("decision tree algorithm", S.vs, top_k=4)
    scores = ret.scores
    assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"
    return f"scores (desc): {[round(s,4) for s in scores]}"
check("Similarity scores returned in descending order", t6_similarity_sorted)

# ── 7. Generator ───────────────────────────────────────────────────────────
print("\n── Section 7: Generator (FLAN-T5) ──")
print("   ⚠️  First run downloads FLAN-T5 model (~300 MB). Please wait...\n")

def t7_q1_present_in_doc():
    """Q1: answer IS in the document."""
    from src.retriever import retrieve
    from src.generator import generate_answer
    q = "What is overfitting in machine learning?"
    ret = retrieve(q, S.vs, top_k=4)
    gen = generate_answer(q, ret.chunks, prefer_local=True)
    assert gen.answer and len(gen.answer) > 5, "Empty answer"
    assert gen.generation_time > 0
    assert gen.model_used
    assert len(gen.source_pages) > 0, "No source pages"
    print(f"\n   Q: {q}")
    print(f"   A: {gen.answer}")
    print(f"   Pages cited: {gen.source_pages}")
    return (f"model={gen.model_used}, time={gen.generation_time:.2f}s, "
            f"pages={gen.source_pages}")
check("Q1 (answer present): generates answer with page citation", t7_q1_present_in_doc)

def t7_q2_not_in_doc():
    """Q2: answer is NOT in the document."""
    from src.retriever import retrieve
    from src.generator import generate_answer
    q = "What is the current price of Bitcoin in USD?"
    ret = retrieve(q, S.vs, top_k=4)
    gen = generate_answer(q, ret.chunks, prefer_local=True)
    assert gen.answer, "Answer should not be empty"
    print(f"\n   Q: {q}")
    print(f"   A: {gen.answer}")
    return f"answer='{gen.answer[:120]}'"
check("Q2 (not in doc): gracefully handles out-of-scope query", t7_q2_not_in_doc)

def t7_q3_page_citation():
    """Q3: specifically tests page reference attribution."""
    from src.retriever import retrieve
    from src.generator import generate_answer
    q = "Which paper introduced the Transformer architecture?"
    ret = retrieve(q, S.vs, top_k=4)
    gen = generate_answer(q, ret.chunks, prefer_local=True)
    assert len(gen.source_pages) > 0, "No source pages returned"
    assert len(gen.source_files) > 0, "No source files returned"
    pages_str = ", ".join(f"p.{p}" for p in gen.source_pages)
    print(f"\n   Q: {q}")
    print(f"   A: {gen.answer}")
    print(f"   Cited pages: {pages_str} | Files: {gen.source_files}")
    return f"pages={pages_str}, files={gen.source_files}"
check("Q3 (page citation): answer includes source pages", t7_q3_page_citation)

def t7_no_chunks():
    from src.generator import generate_answer
    gen = generate_answer("anything", [], prefer_local=True)
    assert "No relevant context" in gen.answer, f"Unexpected: {gen.answer}"
    return f"Graceful: '{gen.answer[:80]}'"
check("Empty chunk list → graceful 'no context' message", t7_no_chunks)

# ── 8. Evaluation ──────────────────────────────────────────────────────────
print("\n── Section 8: Evaluation ──")

def t8_log_records():
    from src.retriever import retrieve
    from src.generator import generate_answer
    from src.evaluation import EvaluationTracker
    tracker = EvaluationTracker()
    questions = [
        "What is supervised learning?",
        "Explain the confusion matrix",
        "What is BERT?",
    ]
    for q in questions:
        ret = retrieve(q, S.vs, top_k=4)
        gen = generate_answer(q, ret.chunks, prefer_local=True)
        tracker.log(ret, gen)
    assert len(tracker.records) == 3
    return f"{len(tracker.records)} records logged"
check("EvaluationTracker logs all records", t8_log_records)

def t8_summary():
    from src.retriever import retrieve
    from src.generator import generate_answer
    from src.evaluation import EvaluationTracker
    tracker = EvaluationTracker()
    ret = retrieve("What is regularisation?", S.vs, top_k=4)
    gen = generate_answer("What is regularisation?", ret.chunks, prefer_local=True)
    tracker.log(ret, gen)
    s = tracker.summary()
    assert s["total_queries"] == 1
    assert s["avg_retrieval_latency"]  > 0
    assert s["avg_generation_latency"] > 0
    assert s["avg_total_latency"]      > 0
    assert s["avg_num_chunks"]         > 0
    assert s["avg_similarity"]         > 0
    return (f"total={s['total_queries']}, ret={s['avg_retrieval_latency']:.3f}s, "
            f"gen={s['avg_generation_latency']:.3f}s, sim={s['avg_similarity']:.3f}")
check("Evaluation summary contains all metric keys", t8_summary)

def t8_csv_export():
    from src.retriever import retrieve
    from src.generator import generate_answer
    from src.evaluation import EvaluationTracker
    tracker = EvaluationTracker()
    ret = retrieve("What is cross-validation?", S.vs, top_k=4)
    gen = generate_answer("What is cross-validation?", ret.chunks, prefer_local=True)
    tracker.log(ret, gen)
    path = "test_eval_export.csv"
    tracker.export_csv(path)
    assert Path(path).exists(), "CSV not created"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    required_cols = {"query","answer","retrieval_latency","generation_latency",
                     "avg_similarity","num_chunks","source_pages"}
    missing = required_cols - set(rows[0].keys())
    assert not missing, f"Missing columns: {missing}"
    Path(path).unlink()
    return f"CSV has all required columns: {sorted(rows[0].keys())}"
check("CSV export creates valid file with required columns", t8_csv_export)

# ── 9. App restart: load saved index ──────────────────────────────────────
print("\n── Section 9: App Restart Simulation ──")

def t9_cold_reload():
    """Simulate app restart: fresh FAISSVectorStore loads saved index."""
    from src.vector_store import FAISSVectorStore
    cold_vs = FAISSVectorStore()    # brand-new instance
    ok = cold_vs.load()
    assert ok,               "load() returned False on restart"
    assert cold_vs.is_ready, "Not ready after cold reload"
    # Run a full query on the reloaded store
    from src.retriever import retrieve
    from src.generator import generate_answer
    ret = retrieve("What is backpropagation?", cold_vs, top_k=4)
    gen = generate_answer("What is backpropagation?", ret.chunks, prefer_local=True)
    assert gen.answer
    return (f"Cold reload OK: {cold_vs.total_vectors} vectors, "
            f"query answered in {ret.retrieval_time + gen.generation_time:.2f}s")
check("Cold reload: fresh instance loads saved index and answers correctly", t9_cold_reload)

# ── 10. Full 3-question smoke test ─────────────────────────────────────────
print("\n── Section 10: Full Pipeline Smoke Test ──")

def t10_full():
    from src.vector_store import FAISSVectorStore
    from src.retriever import retrieve
    from src.generator import generate_answer
    from src.evaluation import EvaluationTracker

    vs_f = FAISSVectorStore()
    vs_f.load()
    tracker = EvaluationTracker()

    test_qs = [
        ("Q-Present",  "What is supervised learning?"),
        ("Q-NotInDoc", "What is the boiling point of nitrogen?"),
        ("Q-Citation", "Who introduced the Transformer architecture and in which year?"),
    ]

    print()
    for label, q in test_qs:
        ret = retrieve(q, vs_f, top_k=4)
        gen = generate_answer(q, ret.chunks, prefer_local=True)
        rec = tracker.log(ret, gen)
        print(f"   [{label}] {q}")
        print(f"   Answer : {gen.answer[:130]}")
        print(f"   Pages  : {gen.source_pages} | "
              f"Latency: {rec.total_latency:.2f}s | "
              f"AvgSim: {rec.avg_similarity:.3f} | "
              f"Chunks: {rec.num_chunks}")
        print()

    s = tracker.summary()
    assert s["total_queries"] == 3
    return f"3/3 answered. Avg latency={s['avg_total_latency']:.2f}s, Avg sim={s['avg_similarity']:.3f}"
check("Full 3-question pipeline smoke test", t10_full)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  FINAL RESULTS")
print("="*62)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  ✅ Passed : {passed}")
print(f"  ❌ Failed : {failed}")
print(f"  Total    : {len(results)}")

if failed:
    print("\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    ❌ {name}: {err[:100]}")
else:
    print("\n  🎉 All tests passed! Project is verified and demo-ready.")

print("="*62 + "\n")
sys.exit(0 if failed == 0 else 1)
