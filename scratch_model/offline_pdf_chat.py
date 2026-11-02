import re
import sys
import time
from pathlib import Path

import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from scratch_model.gpt2_inference import GPT2QAModel
    from scratch_model.retriever import BM25Retriever
except ImportError:
    from gpt2_inference import GPT2QAModel
    from retriever import BM25Retriever

# ==========================================
# 1. STREAMLIT PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="Offline PDF AI Assistant | Fine-Tuned LM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }
    .custom-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.84rem;
        font-weight: 600;
        background-color: #0284c7;
        color: #f0f9ff;
        margin-bottom: 8px;
    }
    .retrieval-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.92rem;
        color: #e2e8f0;
    }
    .score-badge {
        background-color: #334155;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.78rem;
        color: #38bdf8;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD FINE-TUNED LANGUAGE MODEL
# ==========================================
@st.cache_resource(show_spinner="Loading Fine-Tuned Language Model...")
def load_llm():
    checkpoint_dir = BASE_DIR / "checkpoint" / "gpt2_finetuned"
    if not checkpoint_dir.exists():
        return None, None

    model = GPT2QAModel(checkpoint_dir)
    model.load()
    info = model.get_model_info()
    return model, info


qa_engine, model_meta = load_llm()


# ==========================================
# 3. SIDEBAR SPECIFICATIONS
# ==========================================
with st.sidebar:
    st.title("⚙️ Model Specifications")

    if model_meta:
        st.success("🟢 Fine-Tuned LM Loaded")
        st.markdown(f"""
        **Model Architecture:** `Causal Transformer (GPT-2 Small)`
        - **Parameters:** `{model_meta['total_params']:,}`
        - **Tokenizer:** BPE Subwords (`{model_meta['vocab_size']:,}` vocab)
        - **Context Window:** `{model_meta['max_seq_len']}` tokens
        - **Execution:** `{model_meta['device'].upper()} (CPU Optimized)`
        - **Pretrained Weights:** `{model_meta.get('pretrained_weights', 'GPT-2 Small')}`
        - **Training Approach:** `{model_meta.get('training_approach', 'Fine-tuned on QA')}`
        - **Training Examples:** `{model_meta.get('training_examples', '176')}`
        - **External APIs:** `0 (100% Offline)`
        """)
    else:
        st.error("🔴 Fine-Tuned Checkpoint Missing")
        st.info("Run `python scratch_model/train_gpt2.py` to train and save the model.")

    st.markdown("---")
    temperature = st.slider("Temperature (Creativity)", 0.05, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max Answer Tokens", 100, 400, 280, 20)
    st.markdown("---")
    st.caption("College Capstone Project | Offline PDF Question Answering System")


# ==========================================
# 4. MAIN CHAT APPLICATION
# ==========================================
st.title("📚 Offline PDF Question Answering System")
st.markdown("**Powered by Fine-Tuned Causal Language Model (124M Parameters) + BM25 Semantic Indexing**")

# Session State initialization
if "retriever" not in st.session_state:
    st.session_state.retriever = BM25Retriever()
if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_pdf = st.file_uploader("Upload any technical or research PDF", type=["pdf"])

if uploaded_pdf:
    if not st.session_state.doc_indexed or st.session_state.doc_name != uploaded_pdf.name:
        with st.spinner("Extracting text and indexing document into semantic BM25 chunks..."):
            retriever = BM25Retriever()
            num_chunks = retriever.index_pdf(uploaded_pdf.getvalue())
            st.session_state.retriever = retriever
            st.session_state.doc_indexed = True
            st.session_state.doc_name = uploaded_pdf.name
            st.success(f"Indexed **{uploaded_pdf.name}**: {num_chunks} semantic chunks created across {len(retriever.chunks)} passages.")

if st.session_state.doc_indexed:
    st.markdown("**💡 Quick Demonstration Inquiries:**")
    c1, c2, c3 = st.columns(3)
    clicked_prompt = None
    if c1.button("🚅 Railway & Monte Carlo", use_container_width=True):
        clicked_prompt = "Tell me about the railway vehicle model and Monte Carlo simulations."
    if c2.button("📋 Summarize Document", use_container_width=True):
        clicked_prompt = "Give me a summary of this PDF."
    if c3.button("🎯 Problem Statement", use_container_width=True):
        clicked_prompt = "What is the precise problem statement?"

    c4, c5, c6 = st.columns(3)
    if c4.button("🔬 Methodology", use_container_width=True):
        clicked_prompt = "Explain the methodology."
    if c5.button("⚠️ Limitations", use_container_width=True):
        clicked_prompt = "What are the limitations?"
    if c6.button("📄 About This Research", use_container_width=True):
        clicked_prompt = "Tell me about this PDF."

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    user_query = st.chat_input("Ask about technical concepts, summaries, methodology, or problem statements...")
    active_query = user_query or clicked_prompt

    if active_query:
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving document context and generating answer..."):
                t_start = time.time()

                top_chunks, intent, scores = st.session_state.retriever.retrieve(active_query, top_k=3)

                if not top_chunks:
                    response_html = """
<div class="custom-badge" style="background-color: #854d0e;">⚠️ Topic Not Found</div>
<br>
The document does not contain information matching this topic.
"""
                elif qa_engine is None:
                    response_html = """
<div class="custom-badge" style="background-color: #dc2626;">❌ Model Not Found</div>
<br>
Please run training to build the fine-tuned checkpoint.
"""
                else:
                    context_text = "\n\n".join([c.text for c in top_chunks])
                    generated_answer, gen_time = qa_engine.generate_answer(
                        question=active_query,
                        context=context_text,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )

                    pages_list = sorted(list(set(c.page for c in top_chunks)))
                    page_str = ", ".join(map(str, pages_list))

                    snippets_html = ""
                    for idx, c in enumerate(top_chunks):
                        score_label = f"<span class='score-badge'>BM25: {scores[idx]}</span>" if idx < len(scores) else ""
                        snippets_html += f"<div style='margin-bottom: 8px;'><b>Page {c.page}</b> {score_label}<br><em>\"{c.text[:220]}...\"</em></div>"

                    # Format formatted bullet points/newlines nicely
                    formatted_ans = generated_answer.replace("\n", "<br>")

                    response_html = f"""
<div class="custom-badge">🧠 Fine-Tuned LM Output (124M Params)</div>
<br>
<b>Generated Answer:</b><br>
<div style="line-height: 1.6; font-size: 0.98rem; margin: 8px 0;">{formatted_ans}</div>

<div class="retrieval-box">
<b>📄 Retrieved Document Evidence (Page {page_str}) [Intent: {intent}]:</b><br><br>
{snippets_html}
</div>
<br>
<small style="color: #94a3b8;">⚡ End-to-end latency: {time.time() - t_start:.2f}s | Retrieval: BM25 Intent Routing | Pretrained base: GPT-2 Small</small>
"""
                st.markdown(response_html, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_html})

else:
    st.info("👆 Please upload a PDF above to begin question answering.")
