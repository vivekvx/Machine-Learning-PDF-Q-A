"""
app.py
------
Streamlined, Impeccable RAG PDF Q&A System.
Axios Timeout Fix & Rock-Solid Document Pipeline.

Key Design & UX Rules Applied:
  - Instant File Upload (Zero Axios Timeout)
  - Dark Slate Design System (#0B0F19 background, #111827 surfaces, #38BDF8 cyan highlights)
  - Instant One-Click Sample Prompts with immediate execution
  - Factual Grounding with page citations & match confidence metrics
  - Groq LLM Acceleration (Primary) -> Gemini -> Extractive Fallback
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Prevent fork-safety issues on macOS Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader   import load_pdfs
from src.chunking     import chunk_documents, get_chunk_stats
from src.vector_store import FAISSVectorStore
from src.retriever    import retrieve
from src.generator    import generate_answer
from src.evaluation   import EvaluationTracker
from src.config       import (
    DEFAULT_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, LOCAL_LLM_MODEL, GROQ_MODEL,
)

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Intelligence Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Modern Dark Slate Design System CSS ────────────────────────────────
CSS_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global Dark Theme Background */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }

    /* Header Banner */
    .hero-container {
        padding: 1.2rem 0rem 1rem 0rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #1F2937;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #9CA3AF;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-primary {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-success {
        background-color: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-warning {
        background-color: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Answer Card */
    .answer-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .answer-card:hover {
        border-color: #374151;
    }

    /* Input Field Overrides */
    .stTextInput > div > div > input {
        background-color: #111827;
        color: #F9FAFB;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38BDF8;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)

# ── Session State Initialisation ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()

if "eval_tracker" not in st.session_state:
    st.session_state.eval_tracker = EvaluationTracker()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_stats" not in st.session_state:
    st.session_state.chunk_stats = {}

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "query_preset" not in st.session_state:
    st.session_state.query_preset = ""

# Try loading existing FAISS index from disk on app start
if not st.session_state.vector_store.is_ready:
    st.session_state.vector_store.load()


# ── Helper Function: Index Documents ─────────────────────────────────────────
def process_and_index_files(uploaded_files):
    with st.spinner("⚡ Processing & Embedding PDF Document(s)..."):
        try:
            tmp_dir = tempfile.mkdtemp()
            tmp_paths = []
            for uf in uploaded_files:
                path = Path(tmp_dir) / uf.name
                path.write_bytes(uf.getvalue())
                tmp_paths.append(path)

            page_docs = load_pdfs(tmp_paths)
            if not page_docs:
                st.error("⚠️ Could not extract text from the PDF. The file may be scanned or image-only.")
                return

            chunks = chunk_documents(page_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            stats  = get_chunk_stats(chunks)

            vs = st.session_state.vector_store
            vs.build_from_chunks(chunks, show_progress=False)
            vs.save()

            st.session_state.indexed_files = [uf.name for uf in uploaded_files]
            st.session_state.chunk_stats   = stats
            st.session_state.last_indexed_sig = sorted([f"{f.name}_{f.size}" for f in uploaded_files])
            
            st.toast(f"✅ Successfully indexed {len(uploaded_files)} PDF(s) ({stats['count']} chunks)", icon="🚀")
            logger.info(f"Indexed {len(uploaded_files)} files successfully.")

        except Exception as e:
            st.error(f"❌ Document indexing failed: {e}")
            logger.exception("Indexing failure")


# ── Helper Function: Execute Q&A Query ────────────────────────────────────────
def execute_qa(query_text):
    if not query_text or not query_text.strip():
        return

    with st.spinner("🧠 Retrieving semantic context & generating answer..."):
        try:
            # 1. Dense Retrieval
            ret_result = retrieve(
                query_text.strip(),
                st.session_state.vector_store,
                top_k=DEFAULT_TOP_K,
            )

            # 2. Answer Generation (Groq -> Gemini -> Extractive)
            gen_result = generate_answer(
                query_text.strip(),
                ret_result.chunks,
                prefer_local=False,
            )

            # 3. Log Performance Evaluation
            st.session_state.eval_tracker.log(ret_result, gen_result)

            # 4. Save to History
            st.session_state.qa_history.insert(0, {
                "question": query_text.strip(),
                "answer": gen_result.answer,
                "source_pages": gen_result.source_pages,
                "source_files": gen_result.source_files,
                "model": gen_result.model_used,
                "retrieval_time": ret_result.retrieval_time,
                "gen_time": gen_result.generation_time,
                "top_sim": ret_result.top_score,
                "avg_sim": ret_result.avg_similarity,
                "chunks": ret_result.chunks,
                "scores": ret_result.scores,
            })

        except Exception as e:
            st.error(f"❌ Error executing Q&A: {e}")
            logger.exception("Execution Error")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Document Engine & Knowledge Base
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Document Engine")
    st.markdown("Select PDF document(s) below:")

    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        current_sig = sorted([f"{f.name}_{f.size}" for f in uploaded_files])
        is_already_indexed = (st.session_state.get("last_indexed_sig") == current_sig)
        
        if not is_already_indexed:
            st.info(f"📄 {len(uploaded_files)} PDF(s) ready to index.")
            if st.button("🚀 Process & Index PDF(s)", width="stretch", type="primary"):
                process_and_index_files(uploaded_files)
                st.rerun()

    st.divider()

    # Knowledge Base Status Card
    st.markdown("### 📊 Knowledge Base")
    vs = st.session_state.vector_store

    if vs.is_ready:
        st.markdown(
            f"""
            <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">Status</div>
                <div style="font-weight: 600; color: #34D399; font-size: 0.95rem;">● Active & Ready</div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1;">
                    • <b>{vs.total_vectors}</b> Vector Embeddings<br>
                    • <b>{len(vs.chunks)}</b> Document Chunks<br>
                    • <b>{len(st.session_state.indexed_files)}</b> Active File(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.indexed_files:
            st.markdown("**Indexed Files:**")
            for fname in st.session_state.indexed_files:
                st.markdown(f"- 📄 `{fname}`")
    else:
        st.markdown(
            """
            <div style="background-color: #1E293B; border: 1px dashed #475569; border-radius: 8px; padding: 0.8rem; color: #94A3B8; font-size: 0.85rem;">
                ⚠️ No active document index.<br>Upload a PDF above to begin.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # System Control Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Reset KB", width="stretch", type="secondary"):
            st.session_state.vector_store.clear()
            st.session_state.indexed_files = []
            st.session_state.chunk_stats   = {}
            st.session_state.last_indexed_sig = None
            st.session_state.eval_tracker.clear()
            st.session_state.qa_history    = []
            st.toast("Knowledge base reset!", icon="🗑️")
            st.rerun()

    with col_b:
        if st.button("📥 Export CSV", width="stretch", type="secondary"):
            path = st.session_state.eval_tracker.export_csv()
            if path:
                st.toast(f"Exported to {path}", icon="📊")
            else:
                st.toast("No query history to export.", icon="⚠️")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"**Embedding**: `{EMBEDDING_MODEL}`")
    st.caption(f"**LLM Engine**: `{GROQ_MODEL}`")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WORKSPACE — Q&A Intelligence Interface
# ═════════════════════════════════════════════════════════════════════════════

# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">RAG PDF Intelligence Platform</div>
        <div class="hero-subtitle">
            Grounded Neural Retrieval & Question Answering over PDF Documents
        </div>
        <div style="margin-top: 0.6rem;">
            <span class="badge badge-primary">FAISS Vector Search</span>
            <span class="badge badge-success">Dense Embeddings</span>
            <span class="badge badge-warning">Groq LLM Accelerated</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

vs = st.session_state.vector_store

if not vs.is_ready:
    st.info("💡 **Getting Started**: Upload one or more PDF documents in the sidebar on the left and click **Process & Index PDF(s)**.")
else:
    # Quick Sample Prompts (Instant Execution)
    st.markdown("**Sample Prompts:**")
    sp_col1, sp_col2, sp_col3 = st.columns(3)
    
    selected_prompt = None
    with sp_col1:
        if st.button("💡 Summarize key findings", width="stretch"):
            selected_prompt = "Summarize the key findings and main conclusions of this document."
    with sp_col2:
        if st.button("📊 Explain core methodology", width="stretch"):
            selected_prompt = "What is the core methodology or approach described in this paper?"
    with sp_col3:
        if st.button("🎯 List primary data & results", width="stretch"):
            selected_prompt = "What are the primary data points, results, and evaluation metrics presented?"

    if selected_prompt:
        execute_qa(selected_prompt)

    # Question Input Form
    with st.form("qa_form", clear_on_submit=False):
        query_input = st.text_input(
            "Ask a Question",
            placeholder="e.g. What are the key findings or methodologies described in the document?",
            key="user_query",
        )
        submit_btn = st.form_submit_button("🔍 Get Grounded Answer", type="primary")

    if submit_btn and query_input:
        execute_qa(query_input)

    # Display Session Q&A History
    if st.session_state.qa_history:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, item in enumerate(st.session_state.qa_history):
            pages_text = ", ".join(f"p.{p}" for p in item["source_pages"]) if item["source_pages"] else "N/A"
            files_text = ", ".join(item["source_files"]) if item["source_files"] else "Document"
            
            with st.container():
                st.markdown(f"### 💬 Q: {item['question']}")
                st.markdown(item['answer'])
                
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid #1F2937;">
                        📌 <b>Sources:</b> {files_text} &nbsp;|&nbsp; 
                        📄 <b>Pages:</b> <code>{pages_text}</code> &nbsp;|&nbsp; 
                        ⚡ <b>Latency:</b> {item['gen_time']:.2f}s (retrieval: {item['retrieval_time']:.3f}s) &nbsp;|&nbsp; 
                        🎯 <b>Similarity:</b> {item['top_sim']*100:.1f}% Match &nbsp;|&nbsp; 
                        🤖 <b>Engine:</b> <code>{item['model']}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Grounded Context Chunks Inspector
                with st.expander(f"📚 View {len(item['chunks'])} Retrieved Grounded Context Snippets"):
                    for idx, (c, score) in enumerate(zip(item['chunks'], item['scores']), start=1):
                        st.markdown(f"**Chunk {idx}** | Source: `{c.source}` | Page **{c.page_number}** | Relevance Score: `{score:.4f}`")
                        st.info(c.chunk_text)
                st.divider()

    # Performance Metrics Drawer
    if st.session_state.eval_tracker.records:
        with st.expander("📊 Live System Metrics & Benchmark Summary"):
            summary = st.session_state.eval_tracker.summary()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Queries", summary["total_queries"])
            col2.metric("Avg Total Latency", f"{summary['avg_total_latency']:.2f}s")
            col3.metric("Avg Similarity Score", f"{summary['avg_similarity']:.3f}")
            col4.metric("Avg Chunks Retrieved", f"{summary['avg_num_chunks']:.1f}")
