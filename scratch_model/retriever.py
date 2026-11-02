import math
import re
from pathlib import Path
import fitz  # PyMuPDF


class DocumentChunk:
    def __init__(self, text: str, page: int, chunk_id: int, is_abstract: bool = False, is_conclusion: bool = False):
        self.text = text.strip()
        self.page = page
        self.chunk_id = chunk_id
        self.is_abstract = is_abstract
        self.is_conclusion = is_conclusion
        self.words = [w.lower() for w in re.findall(r"\w+", self.text)]

    def __len__(self):
        return len(self.words)


class BM25Retriever:
    """
    Okapi BM25 Document Chunk Retrieval Engine implemented from scratch in pure Python.
    Preserves page citations and ranks passages by probabilistic relevance.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_title = ""

    def index_pdf(self, pdf_input: str | bytes | Path, chunk_size: int = 180, overlap: int = 40):
        """Extract and semantically index an uploaded PDF into overlapping chunks."""
        if isinstance(pdf_input, (str, Path)):
            doc = fitz.open(pdf_input)
        else:
            doc = fitz.open(stream=pdf_input, filetype="pdf")

        self.chunks = []
        self.doc_len = []
        self.doc_freqs = {}
        self.idf = {}

        total_pages = len(doc)
        chunk_counter = 0

        # Extract title from first page
        first_page_text = doc[0].get_text() if total_pages > 0 else ""
        first_lines = [l.strip() for l in first_page_text.split("\n") if len(l.strip()) > 5]
        self.doc_title = first_lines[0] if first_lines else "Uploaded PDF Document"

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page_text = doc[page_idx].get_text()
            cleaned_text = re.sub(r"\s+", " ", page_text).strip()
            if not cleaned_text:
                continue

            words = cleaned_text.split()
            if not words:
                continue

            is_first_page = (page_num == 1)
            is_last_page = (page_num >= total_pages - 1)

            # Slide window across words with overlap
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 25 and len(words) > 50:
                    continue  # Skip tiny trailing slivers

                chunk_str = " ".join(chunk_words)
                is_abstract = is_first_page and ("abstract" in chunk_str.lower() or i == 0)
                is_conclusion = is_last_page and ("conclusion" in chunk_str.lower() or "concluding" in chunk_str.lower())

                chunk = DocumentChunk(
                    text=chunk_str,
                    page=page_num,
                    chunk_id=chunk_counter,
                    is_abstract=is_abstract,
                    is_conclusion=is_conclusion
                )
                self.chunks.append(chunk)
                chunk_counter += 1

        # Precompute BM25 statistics
        num_docs = len(self.chunks)
        if num_docs == 0:
            return 0

        self.doc_len = [len(c.words) for c in self.chunks]
        self.avg_doc_len = sum(self.doc_len) / num_docs

        for chunk in self.chunks:
            seen_words = set(chunk.words)
            for w in seen_words:
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1

        for w, df in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[w] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

        return num_docs

    def score_chunk(self, query_terms: list[str], chunk_idx: int) -> float:
        """Compute BM25 score for a specific chunk given query terms."""
        chunk = self.chunks[chunk_idx]
        d_len = self.doc_len[chunk_idx]
        score = 0.0

        term_counts = {}
        for w in chunk.words:
            term_counts[w] = term_counts.get(w, 0) + 1

        for q in query_terms:
            if q not in term_counts:
                continue
            tf = term_counts[q]
            idf = self.idf.get(q, 0.0)

            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[DocumentChunk], str, list[float]]:
        """
        Intent-aware retrieval:
        1. Checks for Document Summary / Overview Intent.
        2. Checks for Section-Specific Intent (methodology, limitations, problem statement).
        3. Runs BM25 for technical concept search.
        """
        if not self.chunks:
            return [], "EMPTY_INDEX", []

        q_lower = query.lower().strip()

        # --- 1. Detect Document Summary / Overview Intent ---
        summary_triggers = [
            "summary", "summarize", "tell me about this pdf", "tell me about the document",
            "what is this pdf about", "what is this document about", "overview", "main topic",
            "give me a summary", "brief summary", "explain this paper", "what does this paper discuss"
        ]

        if any(trigger in q_lower for trigger in summary_triggers):
            # Gather abstract chunks, first page chunks, and conclusion chunks
            abstract_chunks = [c for c in self.chunks if c.is_abstract or c.page == 1]
            conclusion_chunks = [c for c in self.chunks if c.is_conclusion or c.page == self.chunks[-1].page]

            selected = []
            if abstract_chunks:
                selected.append(abstract_chunks[0])
            if len(abstract_chunks) > 1:
                selected.append(abstract_chunks[1])
            if conclusion_chunks and conclusion_chunks[-1] not in selected:
                selected.append(conclusion_chunks[-1])

            if selected:
                return selected[:top_k], "DOCUMENT_SUMMARY", [1.0] * len(selected)

        # --- 2. Query Preprocessing & Expansion for Section Search ---
        query_words = [w.lower() for w in re.findall(r"\w+", q_lower)]
        stop_words = {"tell", "me", "about", "what", "is", "the", "are", "and", "in", "of", "to", "for", "a", "an", "this", "that"}
        filtered_q = [w for w in query_words if w not in stop_words]

        # Semantic term expansion
        expanded_q = list(filtered_q)
        if "advantages" in filtered_q or "advantage" in filtered_q or "benefits" in filtered_q:
            expanded_q.extend(["accuracy", "performance", "improvement", "novel", "benefit", "advantages", "effective", "robustness", "capability"])
        if "disadvantages" in filtered_q or "disadvantage" in filtered_q or "drawback" in filtered_q or "drawbacks" in filtered_q:
            expanded_q.extend(["limitation", "limitations", "compromise", "challenge", "difficult", "sensitivity", "deteriorates", "worn"])
        if "findings" in filtered_q or "finding" in filtered_q or "results" in filtered_q:
            expanded_q.extend(["results", "findings", "accuracy", "demonstrated", "achieved", "improvement", "conclusions", "table", "performance"])
        if "methodology" in filtered_q or "method" in filtered_q:
            expanded_q.extend(["method", "methodology", "framework", "simulation", "simulations", "approach", "algorithm"])
        if "limitations" in filtered_q or "limitation" in filtered_q:
            expanded_q.extend(["limitations", "limitation", "weakness", "restricted", "uncertainty", "further"])
        if "problem" in filtered_q and "statement" in filtered_q:
            expanded_q.extend(["problem", "statement", "objective", "aim", "formulation", "hunting"])
        if "conclusion" in filtered_q or "conclusions" in filtered_q:
            expanded_q.extend(["conclusion", "conclusions", "concluding", "remarks", "demonstrated", "results"])
        if "railway" in filtered_q or "vehicle" in filtered_q:
            expanded_q.extend(["railway", "vehicle", "model", "hunting", "bogie", "wheelset"])
        if "monte" in filtered_q or "carlo" in filtered_q:
            expanded_q.extend(["monte", "carlo", "simulation", "simulations", "stochastic", "uncertainty"])

        # --- 3. Compute BM25 Scores Across All Chunks ---
        scores = []
        for idx in range(len(self.chunks)):
            sc = self.score_chunk(expanded_q, idx)
            # Boost section headers
            chunk_text_lower = self.chunks[idx].text.lower()
            if "problem statement" in q_lower and "problem statement" in chunk_text_lower:
                sc += 8.0
            if "methodology" in q_lower and ("method" in chunk_text_lower or "simulation" in chunk_text_lower):
                sc += 4.0
            if "limitations" in q_lower and "limitation" in chunk_text_lower:
                sc += 6.0

            scores.append((sc, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        top_results = []
        top_scores = []
        for sc, idx in scores[:top_k]:
            if sc > 0.05:  # Valid non-zero BM25 match
                top_results.append(self.chunks[idx])
                top_scores.append(round(sc, 3))

        if top_results:
            return top_results, "BM25_MATCH", top_scores

        # Fallback to first page if query has no matching terms
        fallback_chunks = [self.chunks[0]] if self.chunks else []
        return fallback_chunks, "FALLBACK_FIRST_PAGE", [0.0]
