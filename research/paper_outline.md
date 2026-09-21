# Research Paper Outline
## "A Retrieval-Augmented Question Answering System for PDF Documents Using Machine Learning"

**Author:** [Your Name]  
**Institution:** [Your University / Department]  
**Semester:** 7th Semester, B.Sc. Computer Science  
**Date:** [Month, Year]

---

## Abstract

The exponential growth of digital documents has created a pressing need for systems capable of
answering natural-language questions directly from unstructured text. This paper presents a
**Retrieval-Augmented Generation (RAG)** system for PDF question answering that combines
dense vector retrieval with a local sequence-to-sequence language model. The system extracts
text from uploaded PDF documents, splits it into semantically coherent chunks, encodes them
using the `sentence-transformers/all-MiniLM-L6-v2` model, and stores the resulting embeddings
in a FAISS index. At query time, the user's question is embedded and the top-k most similar
chunks are retrieved via cosine similarity. These chunks form the context for a FLAN-T5-base
model that generates a grounded, citation-aware answer. Experimental evaluation on a
document corpus demonstrates retrieval latencies under X ms and answer generation under Y s,
with average cosine similarity scores of Z across retrieved chunks. The system is deployed as
an interactive Streamlit application and runs entirely locally without requiring external APIs.

*Keywords:* Retrieval-Augmented Generation, PDF Question Answering, FAISS, Sentence
Transformers, FLAN-T5, Natural Language Processing, Information Retrieval.

---

## 1. Introduction

The digital transformation of academic, legal, medical, and business domains has led to an
unprecedented volume of information locked inside PDF documents. Manually searching through
long documents for specific answers is time-consuming and error-prone. Large Language Models
(LLMs) offer the potential for automated question answering, but they suffer from two critical
limitations: (1) they cannot access documents beyond their training data, and (2) they
hallucinate plausible but factually incorrect information.

Retrieval-Augmented Generation (RAG) addresses both limitations by grounding the LLM's
response in documents retrieved at query time. This paper proposes and implements a complete
RAG pipeline that:

- Extracts and preserves page-level structure from PDFs.
- Constructs a dense vector index using state-of-the-art sentence embeddings.
- Retrieves context using approximate nearest-neighbour search.
- Generates answers that cite specific pages in the source document.

The rest of this paper is structured as follows: Section 2 states the problem formally.
Section 3 defines the objectives. Section 4 surveys related work. Section 5 describes the
existing system. Section 6 presents the proposed system. Section 7 details the architecture.
Section 8 describes the methodology. Section 9 explains the algorithm. Section 10 covers
the implementation. Section 11 describes the experimental setup. Section 12 presents
evaluation metrics. Section 13 discusses results. Section 14 discusses limitations.
Section 15 outlines future work. Section 16 concludes.

---

## 2. Problem Statement

Traditional keyword-based document search systems (e.g., Ctrl+F or Elasticsearch) match
queries to documents based on lexical overlap rather than semantic meaning. They fail when:

- The user uses different words than the document (vocabulary mismatch).
- The answer spans multiple sentences or paragraphs.
- The document is too long to scan manually.

Closed-book LLMs (like GPT-2 or BERT) can answer questions about their training data but
cannot reason about new, unseen documents.

**Research Problem:** How can we build a system that answers natural-language questions
about arbitrary PDF documents accurately, with source attribution, and without requiring
external cloud APIs?

---

## 3. Objectives

1. Implement a PDF text extraction pipeline that preserves page-number metadata.
2. Design a chunking strategy that balances semantic coherence with embedding-model token limits.
3. Build a dense vector index using sentence-transformer embeddings stored in FAISS.
4. Implement semantic similarity retrieval (top-k chunks per query).
5. Integrate a local LLM (FLAN-T5) for grounded answer generation from retrieved context.
6. Develop a Streamlit-based user interface with real-time evaluation metrics.
7. Evaluate the system using retrieval latency, generation latency, and similarity score.

---

## 4. Literature Review

### 4.1 Retrieval-Augmented Generation
Lewis et al. (2020) introduced RAG as a paradigm that combines a parametric memory (the LLM)
with a non-parametric memory (a dense retrieval corpus). They showed that RAG models
outperform closed-book LLMs on knowledge-intensive tasks.

### 4.2 Dense Passage Retrieval
Karpukhin et al. (2020) proposed DPR — using bi-encoder neural networks to embed both
questions and passages into a shared dense space, enabling retrieval by maximum inner product
search. Our system uses a similar bi-encoder approach via sentence-transformers.

### 4.3 Sentence Transformers
Reimers & Gurevych (2019) introduced SBERT, adapting BERT with siamese network structures
and mean pooling to produce semantically meaningful sentence embeddings. The
`all-MiniLM-L6-v2` model distills this knowledge into a 6-layer architecture with 384-dim
embeddings, achieving strong performance with low inference cost.

### 4.4 FAISS
Johnson et al. (2017) developed FAISS as a library for efficient similarity search over dense
vectors. Its IndexFlatIP (exact inner-product search) provides deterministic results and is
well-suited for moderate-scale corpora (up to millions of vectors on CPU).

### 4.5 FLAN-T5
Chung et al. (2022) introduced FLAN-T5, a T5 model fine-tuned on a large collection of
instruction-following datasets. It demonstrates strong zero-shot and few-shot performance on
question-answering tasks, making it ideal for controlled, context-grounded generation.

### 4.6 RAG Evaluation (RAGAS)
Es et al. (2023) proposed RAGAS, a framework for evaluating RAG systems on faithfulness,
answer relevancy, context precision, and context recall. Our evaluation adopts these metric
definitions while implementing them at a simplified level appropriate for the project scope.

---

## 5. Existing System

The original system (reference implementation: Youssefelsayed148/Machine-Learning-PDF-Q-A)
was a Jupyter Notebook that:
- Hard-coded a single PDF file path.
- Used LangChain's PyPDFLoader without preserving page numbers.
- Relied on LangChain abstractions (ConversationalRetrievalChain) that hide the ML pipeline.
- Provided a basic Gradio UI with no source attribution or evaluation.
- Had no modular code structure.

**Limitations:**
- No support for multiple PDFs.
- No page citation in answers.
- No evaluation metrics.
- Not runnable as a standalone application.
- Code not suitable for academic presentation.

---

## 6. Proposed System

Our proposed system improves over the existing baseline in every dimension:

| Feature                  | Existing System        | Proposed System               |
|--------------------------|------------------------|-------------------------------|
| PDF support              | Single, hard-coded     | Multiple, user-uploaded       |
| Page preservation        | None                   | 1-indexed per page            |
| UI                       | Basic Gradio           | Full Streamlit with tabs      |
| Source citation          | None                   | Page numbers + filenames      |
| Evaluation               | None                   | Latency, similarity, CSV      |
| Code structure           | Monolithic notebook    | Modular src/ package          |
| LLM                      | FLAN-T5-small          | FLAN-T5-base + API fallbacks  |
| Vector store persistence | None                   | Saved to disk (faiss_index)   |
| Chunk settings           | Fixed                  | User-configurable via UI      |

---

## 7. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INDEXING PIPELINE                        │
│                                                              │
│  [PDF Files]                                                 │
│      │                                                       │
│      ▼                                                       │
│  [pdf_loader.py] ── PyMuPDF extraction ── PageDocument[]     │
│      │                                                       │
│      ▼                                                       │
│  [chunking.py] ─── RecursiveCharacterTextSplitter ─ Chunk[]  │
│      │                                                       │
│      ▼                                                       │
│  [embeddings.py] ── SentenceTransformer ── float32 vectors   │
│      │                                                       │
│      ▼                                                       │
│  [vector_store.py] ── FAISS IndexFlatIP ── saved to disk     │
└──────────────────────────────────────────────────────────────┘
                            │
                     [FAISS INDEX]
                            │
┌──────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE                       │
│                                                              │
│  [User Query]                                                │
│      │                                                       │
│      ▼                                                       │
│  [embeddings.py] ── embed_query() ── query vector            │
│      │                                                       │
│      ▼                                                       │
│  [vector_store.py] ── faiss.search() ── top-k chunks         │
│      │                                                       │
│      ▼                                                       │
│  [retriever.py] ── RetrievalResult (chunks + scores + time)  │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                      GENERATION PIPELINE                      │
│                                                              │
│  [Context Chunks + Question]                                 │
│      │                                                       │
│      ▼                                                       │
│  [generator.py] ── _build_prompt() ── instruction prompt     │
│      │                                                       │
│      ▼                                                       │
│  [FLAN-T5 / Groq / Gemini] ── GenerationResult              │
│      │                                                       │
│      ▼                                                       │
│  [Answer + Page Citations + Similarity Scores]               │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│                      EVALUATION                              │
│  [evaluation.py] ── EvaluationTracker ── CSV export          │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Methodology
*(See methodology.md for full detail)*

- PDF text extraction using PyMuPDF
- Recursive character text splitting (chunk_size=500, overlap=50)
- Sentence embedding using all-MiniLM-L6-v2
- FAISS IndexFlatIP (exact cosine similarity via L2-normalised inner product)
- Top-k semantic retrieval
- Prompt-engineered answer generation with FLAN-T5
- Evaluation via latency measurement and similarity scoring

---

## 9. Algorithm

```
ALGORITHM: RAG-PDF-QA

INPUT: User query Q, FAISS index I, chunk metadata M, integer k
OUTPUT: Grounded answer A with page citations

INDEXING (offline):
  For each PDF file P:
    1. Extract pages D = extract_pages(P)             # PyMuPDF
    2. Split into chunks C = recursive_split(D)       # chunk_size=500
    3. Embed chunks E = embed(C)                      # all-MiniLM-L6-v2
    4. Add E to FAISS index I
    5. Store metadata M = {chunk_text, page, source}

RETRIEVAL:
  6. Embed query: q_vec = embed(Q)
  7. Search: indices, scores = I.search(q_vec, k)
  8. Fetch: chunks = [M[i] for i in indices]

GENERATION:
  9. Build prompt: prompt = SYSTEM_PROMPT + context(chunks) + Q
  10. Generate: A = FLAN-T5(prompt)
  11. Cite: citations = {chunk.page for chunk in chunks}

RETURN: A, citations
```

---

## 10. Implementation

The system is implemented in Python 3.10+ with the following modular structure:

- `src/config.py`       — All tunable hyperparameters
- `src/pdf_loader.py`   — PDF text extraction
- `src/chunking.py`     — Recursive text chunking
- `src/embeddings.py`   — Sentence-transformer wrapper
- `src/vector_store.py` — FAISS index management
- `src/retriever.py`    — Query embedding + search
- `src/generator.py`    — LLM answer generation
- `src/evaluation.py`   — Metrics tracking and export
- `app.py`              — Streamlit UI

---

## 11. Experimental Setup

- **Hardware:** [Your CPU/GPU specs, e.g., Intel Core i5, 16 GB RAM, no GPU]
- **OS:** [Your OS, e.g., macOS / Ubuntu]
- **Python:** 3.10.x
- **Test corpus:** [Describe the PDFs you tested with]
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **LLM:** `google/flan-t5-base` (250M params, local)
- **Chunk size:** 500 characters, 50 overlap
- **Top-K:** 4 chunks per query
- **Test questions:** [Number] manually crafted questions across [N] topics

---

## 12. Evaluation Metrics

| Metric                  | Definition                                      | Implementation |
|-------------------------|-------------------------------------------------|----------------|
| Retrieval latency       | Time to embed query + FAISS search              | Measured        |
| Generation latency      | Time for LLM to produce answer                  | Measured        |
| Total latency           | Retrieval + generation time                     | Measured        |
| Avg cosine similarity   | Mean similarity of top-k retrieved chunks       | Measured        |
| Top cosine similarity   | Highest similarity score in results             | Measured        |
| Num chunks retrieved    | Count of retrieved chunks (= k)                 | Measured        |
| Faithfulness            | Answer stays within context                     | Stub (manual)   |
| Answer relevancy        | Answer addresses the question                   | Stub            |
| Context precision       | Fraction of retrieved chunks that are useful    | Stub            |
| Context recall          | Fraction of relevant chunks retrieved           | Stub            |

---

## 13. Results and Discussion

*(Fill in your actual results after running experiments)*

- **Average retrieval latency:** X ms
- **Average generation latency:** Y s
- **Average cosine similarity:** Z
- **Observation 1:** [Your observation]
- **Observation 2:** [Your observation]

Discussion should address:
- Does higher top-k improve answer quality or just add noise?
- Does chunk size affect similarity scores?
- How does FLAN-T5-base compare to larger models?

---

## 14. Limitations

1. **FLAN-T5 context window:** FLAN-T5-base has a 512-token input limit, which constrains
   how many chunks can be included in the prompt.
2. **Scanned PDFs:** The system cannot process image-only PDFs without OCR.
3. **Faithfulness evaluation:** Automated faithfulness measurement requires an LLM judge,
   which adds cost. Manual evaluation was used as a proxy.
4. **Single-language support:** Currently optimised for English documents only.
5. **No conversation memory:** Each question is answered independently.

---

## 15. Future Scope

1. **OCR support** using Tesseract or EasyOCR for scanned documents.
2. **Conversation history** using a sliding-window memory buffer.
3. **Hybrid retrieval** combining BM25 (lexical) + dense (semantic) search.
4. **Re-ranking** using a cross-encoder model after initial retrieval.
5. **RAGAS evaluation** with automated faithfulness and relevancy scoring.
6. **Multi-language support** using multilingual sentence-transformers.
7. **Document hierarchy** preserving section titles and headings as metadata.
8. **Table and figure extraction** for non-textual content.

---

## 16. Conclusion

This paper presented a complete Retrieval-Augmented Generation system for PDF question
answering. The system successfully addresses the hallucination problem of standalone LLMs
by grounding answers in retrieved document context. By combining sentence-transformer
embeddings, FAISS vector search, and FLAN-T5 generation, the system achieves efficient and
accurate question answering with full page citations, running entirely locally without
external API dependencies. The modular architecture and evaluation framework make it suitable
for academic demonstration and future extension.

---

## References
*(See references.md)*
