# 📄 RAG PDF Question Answering System

> **Academic Project | 7th Semester B.Sc. Computer Science**  
> *"A Retrieval-Augmented Question Answering System for PDF Documents Using Machine Learning"*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-brightgreen)]()

---

## 🧩 Problem Statement

Large Language Models hallucinate facts they were not trained on and cannot reason
over new documents. Keyword search fails on semantic queries (vocabulary mismatch).

**Solution:** Retrieval-Augmented Generation (RAG) retrieves the most semantically
relevant document passages at query time and feeds them as context to the LLM — so
every answer is grounded in your actual PDF, not in the model's parametric memory.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📤 Multi-PDF upload | Upload one or more PDFs through the Streamlit UI |
| 📖 Smart text extraction | PyMuPDF with 1-indexed page-number preservation |
| ✂️ Semantic chunking | Recursive character splitter (size/overlap configurable) |
| 🧠 Local embeddings | `sentence-transformers/all-MiniLM-L6-v2` — no API key |
| 🔍 FAISS vector search | Exact cosine similarity via L2-normalised IndexFlatIP |
| 💬 Grounded answers | FLAN-T5-base (local) with anti-hallucination prompt |
| 📌 Page citations | Answers display exact source pages and filenames |
| 📊 Evaluation metrics | Retrieval/generation latency, similarity score, CSV export |
| ⚙️ Configurable UI | Top-K, chunk size, chunk overlap sliders in sidebar |
| 🗑️ Index management | Clear / reload index without restarting the app |

---

## 🏗️ System Architecture

```
┌────────────────────────── INDEXING (one-time) ───────────────────────────┐
│                                                                          │
│  PDF Files ──► PyMuPDF Extraction ──► Recursive Chunking                │
│                                            │                             │
│                              SentenceTransformer Embeddings (384-dim)   │
│                                            │                             │
│                                   FAISS IndexFlatIP ──► saved to disk   │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
┌────────────────────────── RETRIEVAL (per query) ─────────────────────────┐
│                                                                          │
│  User Query ──► Embed Query ──► Cosine Similarity Search ──► Top-K Chunks│
└──────────────────────────────────────────────────────────────────────────┘
                                        │
┌────────────────────────── GENERATION ────────────────────────────────────┐
│                                                                          │
│  Context Chunks + Question ──► Prompt Engineering ──► FLAN-T5-base      │
│                                                      ──► Answer + Pages  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Notes |
|---|---|---|
| PDF Extraction | PyMuPDF (`pymupdf`) + pypdf fallback | Page-number aware |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` | Respects sentence boundaries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, L2-normalised |
| Vector DB | FAISS `IndexFlatIP` | Exact cosine similarity |
| LLM (primary) | `google/flan-t5-base` — runs fully locally | No API key needed |
| LLM (fallback) | Groq API / Google Gemini | Optional, set in `.env` |
| UI | Streamlit 1.32+ | 4-tab interface |
| Evaluation | Custom tracker + pandas + CSV export | Latency + similarity |

---

## 📁 Project Structure

```
Machine-Learning-PDF-Q-A/
├── app.py                        ← Streamlit main application
├── requirements.txt              ← All Python dependencies
├── test_pipeline.py              ← End-to-end integration tests (28 tests)
├── .env.example                  ← API key template
├── README.md
│
├── src/                          ← Modular ML pipeline
│   ├── config.py                 ← Hyperparameters (edit here to tune)
│   ├── pdf_loader.py             ← PDF text extraction with page numbers
│   ├── chunking.py               ← Recursive text chunking
│   ├── embeddings.py             ← Sentence-transformer embedding wrapper
│   ├── vector_store.py           ← FAISS index: build/save/load/search/clear
│   ├── retriever.py              ← Query embedding + FAISS similarity search
│   ├── generator.py              ← FLAN-T5 answer generation (direct API)
│   └── evaluation.py             ← Metrics tracking + CSV export
│
├── data/sample_pdfs/             ← Place your PDF files here
│   └── create_sample_pdf.py      ← Script to generate a test PDF
│
├── vector_store/                 ← Auto-created: FAISS index files (binary)
├── screenshots/                  ← Add UI screenshots here for the report
│
└── research/                     ← Research paper support files
    ├── paper_outline.md          ← Full 16-section paper outline
    ├── methodology.md            ← Detailed technical methodology
    └── references.md             ← 20 academic references (APA format)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python **3.10 or higher** (tested on 3.14)
- `pip`
- ~3 GB free disk space (for model downloads)
- Internet connection (first run only — models are cached locally)

---

### Step 1 — Clone the repository
```bash
git clone https://github.com/Youssefelsayed148/Machine-Learning-PDF-Q-A.git
cd Machine-Learning-PDF-Q-A
```

### Step 2 — Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows CMD
# venv\Scripts\Activate.ps1       # Windows PowerShell
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **First run note:** The initial startup downloads two models:
> - `all-MiniLM-L6-v2` (~80 MB) — embedding model
> - `google/flan-t5-base` (~1 GB) — answer generation model
>
> These are cached by HuggingFace Hub (in `~/.cache/huggingface/`).
> Subsequent runs start instantly.

### Step 4 — (Optional) Configure API keys for remote LLM fallbacks
```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY and/or GEMINI_API_KEY if desired
```

### Step 5 — Run the application
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧪 Verify the Installation (Run Tests)

To confirm every component works end-to-end before the demo:

```bash
# First create the sample test PDF
pip install reportlab
python3 data/sample_pdfs/create_sample_pdf.py

# Run all 28 integration tests
python3 test_pipeline.py
```

Expected output:
```
✅ Passed : 28
❌ Failed : 0
🎉 All tests passed! Project is verified and demo-ready.
```

---

## 📖 How to Use the App

### 1. Upload & Index
1. Open the **"📤 Upload & Index"** tab.
2. Click **"Upload PDF files"** and select your PDF(s).
3. Optionally adjust *chunk size* and *chunk overlap* in the left sidebar.
4. Click **"🚀 Index Documents"** — watch the 5-step pipeline status.

### 2. Ask Questions
1. Switch to the **"💬 Ask Questions"** tab.
2. Type your question in the text box.
3. Click **"🔍 Get Answer"**.
4. See: answer, page citations, similarity scores, retrieved chunks.

### 3. View Evaluation
1. Switch to the **"📊 Evaluation"** tab.
2. View per-query latency and similarity metrics.
3. Click **"📥 Download CSV"** to export results for your report.

---

## 🔬 How the RAG Pipeline Works

```
Step 1  User uploads PDF
         ↓
Step 2  PyMuPDF extracts text + page numbers (1-indexed)
         ↓
Step 3  RecursiveCharacterTextSplitter creates overlapping ~500-char chunks
        (tries: paragraph → sentence → word boundaries)
         ↓
Step 4  SentenceTransformer encodes each chunk → 384-dim L2-normalised vector
         ↓
Step 5  FAISS.IndexFlatIP stores all vectors → saved to disk as faiss_index.*
         ↓
Step 6  User types a question
         ↓
Step 7  Same SentenceTransformer encodes question → query vector
         ↓
Step 8  FAISS.search(query_vec, k) → top-k chunks by cosine similarity
         ↓
Step 9  Retrieved chunks + question → instruction prompt for FLAN-T5
        (prompt instructs: answer ONLY from context, cite page numbers,
         do not hallucinate)
         ↓
Step 10 FLAN-T5.generate() produces a grounded answer
         ↓
Step 11 Answer + page citations + similarity scores displayed in UI
```

---

## 📊 Evaluation Metrics

| Metric | Measured? | Description |
|---|---|---|
| Retrieval latency | ✅ | Time to embed query + run FAISS search |
| Generation latency | ✅ | Time for FLAN-T5 to generate answer |
| Total latency | ✅ | Full end-to-end response time |
| Avg cosine similarity | ✅ | Mean similarity of top-k retrieved chunks |
| Top cosine similarity | ✅ | Highest similarity score in results |
| Num chunks retrieved | ✅ | Equal to top-k setting |
| Faithfulness | Stub | Requires NLI model or LLM judge (RAGAS) |
| Answer relevancy | Stub | Requires cosine(query_emb, answer_emb) |
| Context precision | Stub | Requires ground-truth annotations |
| Context recall | Stub | Requires ground-truth annotations |

---

## ⚠️ Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pymupdf'` | PyMuPDF not installed | `pip install PyMuPDF` |
| `ModuleNotFoundError: No module named 'langchain_text_splitters'` | Wrong langchain package | `pip install langchain-text-splitters` |
| `RuntimeError: No FAISS index found` | PDFs not indexed yet | Upload PDFs in the Upload & Index tab first |
| `ValueError: No text extracted from PDF` | Scanned/image-only PDF | Use a text-based PDF; OCR not currently supported |
| `FileNotFoundError: PDF not found` | Wrong path | Check file path; use the UI uploader |
| App is slow on first run | Downloading models (~1 GB) | Wait ~5 min; cached on disk afterwards |
| Poor answer quality | Using flan-t5-small | Change `LOCAL_LLM_MODEL` in `src/config.py` to `google/flan-t5-base` |
| Out-of-memory on generation | Model too large for RAM | Use `flan-t5-small` or switch to Groq API |
| `KeyError: Unknown task text2text-generation` | Transformers ≥ 5.x | Already fixed — we use `model.generate()` directly |

---

## 🖼️ Screenshots

*Add screenshots to `screenshots/` after running the app.*

| File | Description |
|---|---|
| `screenshots/01_upload.png` | PDF upload + 5-step indexing status |
| `screenshots/02_question.png` | Question input and answer with citations |
| `screenshots/03_chunks.png` | Retrieved context chunks display |
| `screenshots/04_eval.png` | Evaluation metrics dashboard |

---

## 🎓 Research Paper Relevance

| ML Concept | Where Demonstrated |
|---|---|
| Dense sentence embeddings | `src/embeddings.py` + `all-MiniLM-L6-v2` |
| Cosine similarity via inner product | `src/vector_store.py` + FAISS `IndexFlatIP` |
| Approximate nearest-neighbour search | FAISS vector store |
| Encoder-decoder seq2seq generation | `src/generator.py` + FLAN-T5 |
| Retrieval-Augmented Generation | Full pipeline `app.py` |
| Anti-hallucination prompt engineering | `generator._build_prompt()` |
| Grounded QA with source attribution | Page citations in `GenerationResult` |
| Evaluation metrics for RAG | `src/evaluation.py` + Evaluation tab |
| Modular ML system design | `src/` package (8 specialised modules) |

Supporting research documents:
- [`research/paper_outline.md`](research/paper_outline.md) — Full 16-section paper
- [`research/methodology.md`](research/methodology.md) — Technical deep-dive
- [`research/references.md`](research/references.md) — 20 academic references

---

## 🔮 Future Improvements

1. OCR support for scanned PDFs (Tesseract / EasyOCR)
2. Hybrid retrieval — BM25 (lexical) + dense (semantic) fusion
3. Cross-encoder re-ranking for improved precision
4. Conversation memory for multi-turn Q&A sessions
5. Automated RAGAS faithfulness and relevancy scoring
6. Multi-language support via multilingual sentence-transformers
7. Table and figure extraction (pdfplumber / Camelot)
8. Streaming token output for faster perceived responses

---

## 📄 License

MIT License — free to use, modify, and distribute for academic and personal projects.
