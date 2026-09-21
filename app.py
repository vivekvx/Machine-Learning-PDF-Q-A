"""
app.py
------
Streamlit front-end for the RAG-PDF-QA system.

Run with:
    streamlit run app.py

Pipeline steps shown to the user:
  1. Upload PDF(s)
  2. Extract text (PyMuPDF / pypdf)
  3. Split into chunks (RecursiveCharacterTextSplitter)
  4. Embed chunks (sentence-transformers)
  5. Store in FAISS
  6. User asks question → embed query → FAISS search → FLAN-T5 answer
  7. Show answer + source pages + retrieved chunks + eval metrics
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Ensure src/ is on the Python path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader   import load_pdfs
from src.chunking     import chunk_documents, get_chunk_stats
from src.vector_store import FAISSVectorStore
from src.retriever    import retrieve
from src.generator    import generate_answer
from src.evaluation   import EvaluationTracker
from src.config       import (
    DEFAULT_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, LOCAL_LLM_MODEL,
    VECTOR_STORE_DIR,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════════
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()

if "eval_tracker" not in st.session_state:
    st.session_state.eval_tracker = EvaluationTracker()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_stats" not in st.session_state:
    st.session_state.chunk_stats = {}

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []   # list of (question, GenerationResult, RetrievalResult)

# ── Try loading a previously saved index on app start ─────────────────────
if not st.session_state.vector_store.is_ready:
    st.session_state.vector_store.load()


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — configuration & pipeline info
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("### Retrieval Settings")
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1, max_value=10, value=DEFAULT_TOP_K,
        help="How many document chunks are retrieved per query.",
    )

    st.markdown("### Chunk Settings")
    chunk_size = st.slider(
        "Chunk size (chars)", 200, 1000, CHUNK_SIZE, step=50,
        help="Maximum characters per text chunk.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (chars)", 0, 200, CHUNK_OVERLAP, step=10,
        help="Overlap between consecutive chunks for context continuity.",
    )

    st.markdown("### LLM Backend")
    use_local = st.checkbox("Use local FLAN-T5 (no API key)", value=True)
    if not use_local:
        st.info("Set GROQ_API_KEY or GEMINI_API_KEY in a .env file for remote LLMs.")

    st.divider()

    # Pipeline architecture overview
    st.markdown("### 🔬 Pipeline Architecture")
    st.markdown("""
```
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Recursive Chunking
    ↓
Sentence-Transformer Embeddings
    ↓
FAISS Vector Index
    ↓  (query time)
Query Embedding
    ↓
Cosine Similarity Search
    ↓
Context Retrieval (top-k)
    ↓
FLAN-T5 Answer Generation
    ↓
Answer + Page Citations
```
""")

    st.markdown("### 📊 Session Evaluation")
    summary = st.session_state.eval_tracker.summary()
    if summary.get("total_queries", 0) > 0:
        st.metric("Queries answered",    summary["total_queries"])
        st.metric("Avg total latency",   f"{summary['avg_total_latency']:.2f}s")
        st.metric("Avg similarity",      f"{summary['avg_similarity']:.3f}")
        st.metric("Avg chunks retrieved",f"{summary['avg_num_chunks']:.1f}")
    else:
        st.info("No queries yet. Ask a question to see metrics.")

    st.divider()

    # Clear index button
    if st.button("🗑️ Clear Vector Index", type="secondary"):
        st.session_state.vector_store.clear()
        st.session_state.indexed_files = []
        st.session_state.chunk_stats   = {}
        st.session_state.eval_tracker.clear()
        st.session_state.qa_history    = []
        st.success("Index cleared!")
        st.rerun()

    # Export evaluation CSV
    if st.button("📥 Export Evaluation CSV"):
        path = st.session_state.eval_tracker.export_csv()
        if path:
            st.success(f"Saved to {path}")
        else:
            st.warning("No evaluation data to export yet.")

    st.markdown("---")
    st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{LOCAL_LLM_MODEL}`")


# ═══════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════
st.title("📄 RAG-Based PDF Question Answering System")
st.markdown(
    "**Upload PDF documents → Index them → Ask questions grounded in the document.**  \n"
    "Answers include page citations and retrieved context snippets."
)

# ── Tab layout ─────────────────────────────────────────────────────────────
tab_upload, tab_qa, tab_eval, tab_about = st.tabs([
    "📤 Upload & Index", "💬 Ask Questions", "📊 Evaluation", "ℹ️ About"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Index
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Step 1 — Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="The system will extract text, chunk it, embed it, and store it in FAISS.",
    )

    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"{len(uploaded_files)} file(s) selected.")
        with col2:
            index_btn = st.button("🚀 Index Documents", type="primary")

        if index_btn:
            with st.status("📥 Indexing documents...", expanded=True) as status:

                # ── Save uploads to temp dir ──────────────────────────────
                st.write("💾 Saving uploaded files...")
                tmp_dir = tempfile.mkdtemp()
                tmp_paths = []
                for uf in uploaded_files:
                    tmp_path = Path(tmp_dir) / uf.name
                    tmp_path.write_bytes(uf.read())
                    tmp_paths.append(tmp_path)

                # ── Extract text ───────────────────────────────────────────
                st.write("📖 Extracting text from PDFs...")
                try:
                    page_docs = load_pdfs(tmp_paths)
                    st.write(f"   ✅ Extracted {len(page_docs)} pages total.")
                except Exception as e:
                    st.error(f"❌ PDF extraction failed: {e}")
                    st.stop()

                # ── Chunk ─────────────────────────────────────────────────
                st.write("✂️ Splitting into chunks...")
                chunks = chunk_documents(page_docs, chunk_size, chunk_overlap)
                stats  = get_chunk_stats(chunks)
                st.write(
                    f"   ✅ Created {stats['count']} chunks "
                    f"(avg {stats['avg_len']} chars each)."
                )

                # ── Embed + FAISS ─────────────────────────────────────────
                st.write("🧠 Generating embeddings and building FAISS index...")
                t_start = time.perf_counter()
                vs = st.session_state.vector_store
                vs.build_from_chunks(chunks, show_progress=False)
                vs.save()
                elapsed = time.perf_counter() - t_start
                st.write(
                    f"   ✅ FAISS index built with {vs.total_vectors} vectors "
                    f"in {elapsed:.1f}s."
                )

                # ── Update session state ───────────────────────────────────
                st.session_state.indexed_files = [uf.name for uf in uploaded_files]
                st.session_state.chunk_stats   = stats
                status.update(label="✅ Indexing complete!", state="complete")

    # ── Index status panel ────────────────────────────────────────────────
    st.divider()
    st.subheader("📦 Index Status")

    vs = st.session_state.vector_store
    if vs.is_ready:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Vectors in index", vs.total_vectors)
        col_b.metric("Chunks stored",    len(vs.chunks))
        col_c.metric("Files indexed",    len(st.session_state.indexed_files))

        if st.session_state.indexed_files:
            st.markdown("**Indexed files:**")
            for f in st.session_state.indexed_files:
                st.markdown(f"- 📄 `{f}`")

        if st.session_state.chunk_stats:
            with st.expander("📊 Chunk Statistics"):
                s = st.session_state.chunk_stats
                st.json(s)
    else:
        st.warning("No documents indexed yet. Upload and index PDFs above.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask Questions
# ═══════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Step 2 — Ask a Question")

    if not st.session_state.vector_store.is_ready:
        st.warning("⚠️ Please upload and index PDFs first (Tab 1).")
    else:
        question = st.text_input(
            "Enter your question about the document:",
            placeholder="e.g. What is regularization and how does it prevent overfitting?",
        )
        ask_btn = st.button("🔍 Get Answer", type="primary", disabled=not question)

        if ask_btn and question:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    # ── Retrieval ─────────────────────────────────────────
                    ret_result = retrieve(
                        question,
                        st.session_state.vector_store,
                        top_k=top_k,
                    )

                    # ── Generation ────────────────────────────────────────
                    gen_result = generate_answer(
                        question,
                        ret_result.chunks,
                        prefer_local=use_local,
                    )

                    # ── Log evaluation ────────────────────────────────────
                    eval_record = st.session_state.eval_tracker.log(ret_result, gen_result)

                    # ── Store in history ──────────────────────────────────
                    st.session_state.qa_history.insert(0, (question, gen_result, ret_result))

                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    logger.exception("QA error")
                    st.stop()

        # ── Display Q&A history ───────────────────────────────────────────
        for i, (q, gen, ret) in enumerate(st.session_state.qa_history):
            with st.container():
                st.markdown(f"### 🔹 Q: {q}")

                # Answer box
                st.markdown("#### 💡 Answer")
                st.success(gen.answer)

                # Page citations
                if gen.source_pages:
                    pages_str = ", ".join(f"p.{p}" for p in gen.source_pages)
                    files_str = ", ".join(gen.source_files)
                    st.markdown(
                        f"📌 **Source:** {files_str} &nbsp;|&nbsp; "
                        f"**Pages:** {pages_str} &nbsp;|&nbsp; "
                        f"**Model:** `{gen.model_used}`"
                    )

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval time", f"{ret.retrieval_time:.3f}s")
                col2.metric("Generation time",f"{gen.generation_time:.3f}s")
                col3.metric("Top similarity", f"{ret.top_score:.3f}")
                col4.metric("Avg similarity", f"{ret.avg_similarity:.3f}")

                # Retrieved chunks
                with st.expander(f"📚 Retrieved Context Chunks ({len(ret.chunks)})"):
                    for j, (chunk, score) in enumerate(
                        zip(ret.chunks, ret.scores), start=1
                    ):
                        st.markdown(
                            f"**Chunk {j}** | "
                            f"Source: `{chunk.source}` | "
                            f"Page **{chunk.page_number}** | "
                            f"Similarity: `{score:.4f}`"
                        )
                        st.info(chunk.chunk_text)

                st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 Evaluation Dashboard")
    tracker = st.session_state.eval_tracker

    summary = tracker.summary()
    if summary.get("total_queries", 0) == 0:
        st.info("No queries answered yet. Ask questions in the Q&A tab.")
    else:
        # ── Aggregate metrics ─────────────────────────────────────────────
        st.markdown("#### Aggregate Metrics (Session)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries",         summary["total_queries"])
        c2.metric("Avg Retrieval Latency", f"{summary['avg_retrieval_latency']:.3f}s")
        c3.metric("Avg Generation Latency",f"{summary['avg_generation_latency']:.3f}s")
        c4.metric("Avg Total Latency",     f"{summary['avg_total_latency']:.3f}s")

        c5, c6, c7 = st.columns(3)
        c5.metric("Avg Similarity Score",  f"{summary['avg_similarity']:.3f}")
        c6.metric("Avg Top Similarity",    f"{summary['avg_top_similarity']:.3f}")
        c7.metric("Avg Chunks Retrieved",  f"{summary['avg_num_chunks']:.1f}")

        # ── Per-query table ────────────────────────────────────────────────
        st.markdown("#### Per-Query Results")
        import pandas as pd
        rows = []
        for r in tracker.records:
            rows.append({
                "Query":            r.query[:60] + "…" if len(r.query) > 60 else r.query,
                "Retrieval (s)":    r.retrieval_latency,
                "Generation (s)":   r.generation_latency,
                "Total (s)":        r.total_latency,
                "Chunks":           r.num_chunks,
                "Avg Sim":          r.avg_similarity,
                "Top Sim":          r.top_similarity,
                "Pages":            r.source_pages,
                "Model":            r.model_used,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # ── Export ────────────────────────────────────────────────────────
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name="rag_evaluation_results.csv",
            mime="text/csv",
        )

        # ── RAG metric stubs ──────────────────────────────────────────────
        st.markdown("#### 🔬 Advanced RAG Metrics (Research Paper)")
        st.info(
            "The following metrics require ground-truth labels or an LLM judge. "
            "They are implemented as stubs and discussed in the research paper methodology."
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Faithfulness",       "Stub — see paper")
        mc2.metric("Answer Relevancy",   "Stub — see paper")
        mc3.metric("Context Precision",  "Stub — see paper")
        mc4.metric("Context Recall",     "Stub — see paper")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — About / Architecture
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About This System")
    st.markdown("""
## Retrieval-Augmented Generation (RAG) for PDF Q&A

This system implements a complete **RAG pipeline** as an academic project demonstrating
key machine learning concepts in information retrieval and natural language processing.

### System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PHASE                        │
│                                                         │
│  PDF Files → Text Extraction → Chunking                 │
│      → Embedding (sentence-transformers)                │
│      → FAISS Vector Index (stored to disk)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL PHASE                        │
│                                                         │
│  User Query → Query Embedding                           │
│      → Cosine Similarity Search in FAISS               │
│      → Top-K Relevant Chunks                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  GENERATION PHASE                        │
│                                                         │
│  Retrieved Chunks + Query → Prompt Engineering          │
│      → FLAN-T5 (local) / Groq / Gemini                 │
│      → Grounded Answer with Page Citations             │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack
| Component          | Technology                                    |
|--------------------|-----------------------------------------------|
| PDF Extraction     | PyMuPDF (fitz) + pypdf fallback               |
| Text Chunking      | RecursiveCharacterTextSplitter (LangChain)    |
| Embeddings         | sentence-transformers/all-MiniLM-L6-v2        |
| Vector DB          | FAISS (IndexFlatIP — cosine similarity)       |
| LLM (local)        | google/flan-t5-base (HuggingFace Transformers)|
| LLM (fallback)     | Groq API (llama3) / Google Gemini             |
| UI                 | Streamlit                                     |
| Evaluation         | Custom latency + similarity metrics           |

### Key ML Concepts Demonstrated
- **Dense retrieval** using bi-encoder sentence embeddings
- **Approximate nearest neighbour search** via FAISS
- **Seq2Seq text generation** with encoder-decoder (FLAN-T5)
- **Retrieval-augmented generation** to reduce hallucination
- **Grounded answering** with source page citation
    """)
