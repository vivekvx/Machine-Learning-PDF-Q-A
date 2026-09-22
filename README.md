# 📄 RAG PDF Question Answering System

> **Academic Research Project | 7th Semester B.Sc. Computer Science**  
> *"A Retrieval-Augmented Question Answering System for PDF Documents Using Machine Learning"*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen)]()

---

## 🧩 Problem Statement

Large Language Models (LLMs) hallucinate facts when queried on non-training documents. Keyword search fails on complex semantic queries due to vocabulary mismatch.

**Solution:** Retrieval-Augmented Generation (RAG) extracts text from PDF documents, splits it into semantic chunks, generates 384-dimensional dense vector embeddings (`all-MiniLM-L6-v2`), and indexes them using **FAISS**. At query time, cosine similarity search retrieves the top-K relevant passages to ground the LLM's response with exact source page citations (`[Page N]`).

---

## ✨ System Architecture

```
┌────────────────────────── INDEXING PHASE ───────────────────────────┐
│                                                                     │
│  PDF Files ──► PyMuPDF Extraction ──► Recursive Chunking            │
│                                           │                         │
│                             SentenceTransformer Embeddings (384-dim)│
│                                           │                         │
│                                  FAISS IndexFlatIP ──► Saved to Disk│
└─────────────────────────────────────────────────────────────────────┘
                                        │
┌────────────────────────── RETRIEVAL PHASE ──────────────────────────┐
│                                                                     │
│  User Query ──► Query Embedding ──► Cosine Similarity ──► Top-K Chunks│
└─────────────────────────────────────────────────────────────────────┘
                                        │
┌────────────────────────── GENERATION PHASE ─────────────────────────┐
│                                                                     │
│  Context Chunks + Query ──► Prompt Engineering ──► Groq LLM / Gemini │
│                                                  ──► Answer + Pages │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack & Dependencies

| Component | Technology | Notes |
|---|---|---|
| **Backend REST API** | FastAPI + Uvicorn | High-performance asynchronous REST endpoints |
| **Frontend UI** | HTML5 / CSS3 / JavaScript (SPA) | Impeccable dark-slate responsive interface |
| **PDF Extraction** | PyMuPDF (`fitz`) + `pypdf` fallback | Preserves 1-indexed page numbers |
| **Text Chunking** | LangChain `RecursiveCharacterTextSplitter` | Respects paragraph and sentence boundaries |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional dense vectors, L2-normalised |
| **Vector DB** | FAISS `IndexFlatIP` | Exact cosine similarity via inner product |
| **LLM Engine** | Groq API (`openai/gpt-oss-20b`) / Gemini / Extractive | Grounded Q&A with page citations |
| **Evaluation** | Custom tracker + CSV exporter | Retrieval & generation latency + similarity |

---

## 📁 Project Structure

```
Machine-Learning-PDF-Q-A/
├── server.py                     ← FastAPI REST Server & Entrypoint
├── static/
│   └── index.html                ← Modern SPA Frontend (Dark Slate Theme)
├── test_system.py                ← Lightweight end-to-end verification suite
├── requirements.txt              ← Python dependencies
├── .env.example                  ← Environment variable template
├── .env                          ← Active environment variables (gitignored)
│
├── src/                          ← Machine Learning Pipeline
│   ├── config.py                 ← Central Hyperparameters & Configuration
│   ├── pdf_loader.py             ← Page-number preserving PDF text extractor
│   ├── chunking.py               ← Recursive text splitter & chunk stats
│   ├── embeddings.py             ← SentenceTransformer bi-encoder wrapper
│   ├── vector_store.py           ← FAISS vector index build/save/load/search
│   ├── retriever.py              ← Query embedding & cosine similarity search
│   ├── generator.py              ← Groq / Gemini / Local FLAN-T5 / Extractive QA
│   └── evaluation.py             ← Latency & similarity evaluation logger
│
├── data/sample_pdfs/             ← Reference test documents
│   └── create_sample_pdf.py      ← Generates synthetic ML reference PDF
│
└── research/                     ← Research Paper Assets
    ├── paper_outline.md          ← 16-section academic paper outline
    ├── methodology.md            ← Detailed RAG pipeline methodology
    └── references.md             ← 20 APA-formatted academic citations
```

---

## 🚀 Installation & Localhost Setup

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/vivekvx/Machine-Learning-PDF-Q-A.git
cd Machine-Learning-PDF-Q-A

python3 -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=
```

### 4. Start the Application Server
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### 5. Open in Browser
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## 🧪 System Verification Test Suite

To run all 9 pipeline and API verification tests:

```bash
python3 test_system.py
```

Expected output:
```
============================================================
🧪 RAG PDF Q&A SYSTEM VERIFICATION SUITE
============================================================
✅ Step 1: Sample PDF verified -> data/sample_pdfs/machine_learning_notes.pdf
✅ Step 2: PDF extraction verified -> 3 pages extracted.
✅ Step 3: Chunking verified -> 17 chunks (avg 443.1 chars).
✅ Step 4: Sentence-Transformer embeddings verified (384-dim).
✅ Step 5: FAISS index built & saved -> 17 vectors.
✅ Step 6: Retrieval engine verified (Top score: 0.7908).
✅ Step 7: Extractive QA fallback verified -> model: extractive-tfidf
✅ Step 8: Answer generation verified -> model: groq/openai/gpt-oss-20b, pages: [1]
🌐 Testing FastAPI REST Server Endpoints (http://localhost:8000)...
  ✓ GET /api/status -> 64 vectors, ready=True
  ✓ POST /api/query (Present Context) -> Model: groq/openai/gpt-oss-20b, Latency: 1.25s
  ✓ POST /api/query (Absent Context) -> No hallucination verified.
✅ Step 9: FastAPI REST server endpoints 100% verified!
============================================================
🎉 ALL 9 SYSTEM VERIFICATION STEPS PASSED SUCCESSFULLY!
============================================================
```

---

## 🎓 Demo Flow for Professor Presentation

1. **Open Localhost Interface**:
   Navigate to `http://localhost:8000`. Show the active knowledge base indicator.
2. **Upload PDF Document**:
   Drag & drop a reference PDF (e.g. `data/sample_pdfs/machine_learning_notes.pdf`). Click **Process & Index PDF(s)**. Show the toast notification confirming text extraction, chunking, and FAISS vector generation.
3. **Ask Test Question 1 (Fact Present in Document)**:
   - *Query*: `"What is supervised learning?"`
   - *Demonstrate*: Grounded answer generated by LLM with source page citation `[Page 1]`, top similarity score (`~80% Match`), and latency metrics.
4. **Ask Test Question 2 (Fact Absent from Document)**:
   - *Query*: `"What is the capital of France?"`
   - *Demonstrate*: System correctly replies that context is missing without hallucinating.
5. **Inspect Grounded Context Chunks**:
   Expand **View Retrieved Grounded Context Snippets** to inspect the raw FAISS chunks, page numbers, and cosine match scores.
6. **Export Evaluation Log**:
   Click **Export CSV** in the sidebar to download `rag_evaluation_results.csv` for your research report.

---

## ⚠️ Troubleshooting & Error Matrix

| Issue | Cause | Solution |
|---|---|---|
| `Failed to fetch` / Connection Error | Server not running or missing CORS headers | Run `python3 -m uvicorn server:app --port 8000`. CORS middleware is enabled. |
| `Axios 504 Upload Timeout` | Old Streamlit ASGI upload stream flaw | Upgraded to FastAPI native multipart upload endpoint (`/api/upload`). |
| `Model llama3-8b-8192 decommissioned` | Deprecated Groq model ID | Updated `GROQ_MODEL` in `src/config.py` to `openai/gpt-oss-20b`. |
| `No text extracted from PDF` | Scanned or image-only PDF | Upload a PDF with readable text layers. |
| `PyTorch OOM on model.generate()` | High memory usage of FLAN-T5 on CPU | The system automatically uses Groq API / Extractive QA fallback. |

---

## 📄 License

MIT License — Academic Research Project for Computer Science.
