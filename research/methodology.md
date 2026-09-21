# Methodology

## "A Retrieval-Augmented Question Answering System for PDF Documents Using Machine Learning"

---

## Overview

The RAG pipeline consists of two major phases:

1. **Offline Indexing Phase** — Process PDFs and build the vector index (done once).
2. **Online Query Phase** — Answer user questions in real time using the index.

---

## Phase 1: Offline Indexing

### 1.1 PDF Text Extraction

**Tool:** PyMuPDF (`fitz`) with pypdf as fallback.

**Why PyMuPDF?**
- Faster than pypdf on large documents.
- Better layout preservation (respects multi-column PDFs better).
- Provides access to per-page metadata (author, title, total pages).

**Process:**
1. Open the PDF file.
2. Iterate over each page (0-indexed internally, stored as 1-indexed).
3. Extract plain text using `page.get_text("text")`.
4. Skip blank pages (pages with no extractable text).
5. Store each page as a `PageDocument` with: `page_content`, `page_number`, `source`, `metadata`.

**Handling edge cases:**
- Empty PDFs → raise `ValueError` with a clear message.
- Missing file → raise `FileNotFoundError`.
- Scanned PDFs → inform user that OCR is needed (not currently supported).

---

### 1.2 Text Chunking Strategy

**Tool:** `RecursiveCharacterTextSplitter` from LangChain.

**Why chunking?**
Sentence-transformer embedding models have a maximum token length (typically 256–512 tokens).
Entire pages may exceed this limit and, even when they don't, page-level embeddings are
semantically diluted — one vector must represent many different topics. Smaller, focused
chunks produce more semantically precise embeddings, improving retrieval precision.

**Why recursive character splitting?**
The recursive splitter respects natural linguistic boundaries by trying separators in order:
1. `\n\n` — paragraph boundaries (strongest boundary)
2. `\n` — line boundaries
3. `. ` — sentence boundaries
4. `, ` — clause boundaries
5. ` ` — word boundaries
6. `""` — character-level split (last resort)

This ensures chunks are grammatically coherent rather than cut mid-word.

**Hyperparameters (configurable in `src/config.py` and UI):**
- `chunk_size` = 500 characters (≈ 100–150 words)
- `chunk_overlap` = 50 characters (10% overlap to preserve cross-chunk context)

**Overlap justification:**
A 10% overlap ensures that sentences split across two chunks are partially present in both,
reducing information loss at chunk boundaries.

**Output:** `TextChunk` objects with `chunk_text`, `page_number`, `source`, `chunk_index`.

---

### 1.3 Embedding Generation

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Architecture:**
- 6-layer BERT variant (MiniLM)
- Mean pooling over all token embeddings
- Followed by L2 normalisation
- Output: 384-dimensional dense vector per text input

**Why this model?**
- Pretrained on 1 billion sentence pairs (NLI + semantic similarity tasks)
- Optimised for semantic textual similarity (STS benchmarks)
- Only ~80 MB download; fast inference on CPU
- Widely cited in NLP literature (Reimers & Gurevych, 2019)

**Embedding process:**
1. Extract `chunk_text` from all `TextChunk` objects.
2. Encode in batches of 32 to manage memory.
3. L2-normalise each vector (enabled via `normalize_embeddings=True`).
4. Output: float32 numpy array of shape `(N_chunks, 384)`.

**Normalisation significance:**
When vectors are L2-normalised, inner product equals cosine similarity:
```
cosine_similarity(a, b) = a · b / (|a| × |b|) = a · b   [when |a| = |b| = 1]
```
This allows us to use FAISS's IndexFlatIP (inner product) as a cosine similarity index.

---

### 1.4 FAISS Vector Indexing

**Tool:** FAISS `IndexFlatIP` (Facebook AI Similarity Search)

**Index type: IndexFlatIP**
- Exact (brute-force) inner-product search — no approximation.
- Deterministic: same query always returns the same results.
- Suitable for up to ~1 million vectors on CPU with acceptable speed.
- For larger corpora, `IndexIVFFlat` (approximate) would be preferred.

**Indexing process:**
1. Create `faiss.IndexFlatIP(dim=384)`.
2. Add all chunk embeddings in a single `index.add(embeddings)` call.
3. Store chunk metadata (text, page, source) as a parallel Python list.
4. Persist index to disk: `faiss.write_index(index, path)`.
5. Persist metadata using `pickle`.

**Why save to disk?**
Users should not have to re-index PDFs on every app restart. The saved index loads in
milliseconds vs. minutes for re-embedding large documents.

---

## Phase 2: Online Query (Retrieval + Generation)

### 2.1 Query Embedding

The user's question is passed through the same `SentenceTransformer` model:
```
q_vec = model.encode([question], normalize_embeddings=True)
q_vec = q_vec.astype(np.float32)
```
This places the query in the same embedding space as the document chunks, enabling
meaningful similarity comparison.

---

### 2.2 Similarity Search (Retrieval)

FAISS computes the inner product between the query vector and all indexed chunk vectors:
```
scores, indices = faiss_index.search(q_vec, k=4)
```
Results are returned sorted by descending similarity score. The top-k chunk indices are
used to look up the corresponding `TextChunk` objects from the metadata list.

**Similarity interpretation:**
Since all vectors are L2-normalised, scores are cosine similarities in [-1, 1]:
- 1.0 → identical meaning
- 0.7+ → highly relevant
- 0.5–0.7 → moderately relevant
- < 0.5 → possibly irrelevant

---

### 2.3 Retrieval-Augmented Prompt Engineering

The retrieved chunks are assembled into an instructional prompt:

```
You are a precise document question-answering assistant.
Use ONLY the context passages provided below to answer the question.
- If the answer is present, quote or paraphrase the relevant part and include the page number.
- If the context does not contain enough information, say so explicitly.
- Do NOT use any outside knowledge. Do NOT hallucinate.

=== CONTEXT ===
[Context 1 | Source: doc.pdf | Page 3]
<chunk text>

[Context 2 | Source: doc.pdf | Page 5]
<chunk text>
...

=== QUESTION ===
<user question>

=== ANSWER ===
```

**Prompt design principles:**
- Explicit instruction to stay within context (anti-hallucination).
- Each context block labelled with source file and page number.
- Clear output delimiter (`=== ANSWER ===`) guides the model's output.
- No few-shot examples (zero-shot) to keep latency low with FLAN-T5.

---

### 2.4 Answer Generation

**Model:** `google/flan-t5-base` (250 million parameters)

**Model type:** Encoder-Decoder (T5 architecture)
- Encoder processes the full prompt (question + context).
- Decoder generates the answer auto-regressively.

**Inference settings:**
- `max_new_tokens = 512`
- `temperature = 0.1` (near-deterministic for factual grounding)
- `do_sample = False` (greedy decoding for reproducibility)

**Answer grounding:**
The model is constrained by the prompt to cite only information from the context.
The page numbers embedded in the context labels allow the model to include inline citations.
Post-processing extracts unique page numbers from the retrieved chunks to display as
structured citations regardless of whether the model references them explicitly.

---

### 2.5 Source Citation

Two citation mechanisms are used:

1. **Implicit:** The LLM may naturally reference "Page 3" or "Source: doc.pdf" based on
   the context labels embedded in the prompt.
2. **Explicit:** The `GenerationResult` always includes `source_pages` and `source_files`
   extracted directly from the retrieved `TextChunk` objects, displayed in the UI.

---

## Phase 3: Evaluation

### 3.1 Measured Metrics

| Metric               | Measurement Method                              |
|----------------------|-------------------------------------------------|
| Retrieval latency    | `time.perf_counter()` around embed + search     |
| Generation latency   | `time.perf_counter()` around LLM inference      |
| Avg cosine similarity| Mean of FAISS scores for top-k results          |
| Top cosine similarity| Max FAISS score                                 |
| Num chunks           | Count of retrieved chunks                       |

### 3.2 Advanced Metrics (Stubs)

Per the RAGAS framework (Es et al., 2023):

- **Faithfulness:** Fraction of answer sentences supported by context.
  - Full implementation: NLI model (e.g., `cross-encoder/nli-deberta-v3-small`)
- **Answer Relevancy:** Semantic similarity between question and answer.
  - Full implementation: `cosine(embed(Q), embed(A))`
- **Context Precision:** Fraction of retrieved chunks that contribute to the answer.
  - Full implementation: LLM judge or human annotation
- **Context Recall:** Fraction of ground-truth relevant chunks that were retrieved.
  - Full implementation: Requires annotated ground-truth relevance labels

These are implemented as stub functions in `src/evaluation.py` that return `-1.0`,
clearly signalling "not computed", without breaking the evaluation interface.
