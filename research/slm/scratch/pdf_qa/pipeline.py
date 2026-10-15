"""Offline PDF extraction, TF-IDF+FAISS retrieval, and inspectable candidates."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pdf_loader import PageDocument
from src.chunking import chunk_documents

MAX_PDF_BYTES = 12_000_000
TOP_K = 6
WORD = re.compile(r"[a-z0-9_]+", re.I)
STOP = {"what", "which", "is", "the", "a", "an", "for", "of", "in", "on", "does", "do",
        "how", "many", "name", "current", "value", "result", "recorded", "please"}
PATTERNS = (
    ("method", re.compile(r"\b(?:method|algorithm|procedure|technique)\s+(?:is|was|:|=)\s+(?P<value>[A-Za-z][\w+-]*(?:\s+[A-Za-z][\w+-]*){0,2})(?=[.;,!?]|\s*$)", re.I)),
    ("number", re.compile(r"\b(?:metric|latency|duration|capacity|quantity|precision|port|count|rate)\s+(?:is|was|:|=)\s+(?P<value>\d+(?:\.\d+)?\s?(?:ms|s|MB|GB|%|bits|units|records)?)\b", re.I)),
    ("number", re.compile(r"(?<!\w)(?P<value>\d+(?:\.\d+)?(?:-bit|%|\s?(?:ms|MB|GB|bits|units|records)))(?!\w)", re.I)),
    ("identifier", re.compile(r"\b(?:function|class|module|identifier|owner|team)\s+(?:is|named|called|:|=)\s+(?P<value>[A-Za-z_][A-Za-z0-9_+-]*)\b", re.I)),
)


def source_sentences(text: str):
    """Yield exact source slices; never normalize or manufacture PDF offsets."""
    start = 0
    for i, ch in enumerate(text):
        if ch not in ".!?":
            continue
        after = text[i + 1:]
        if after and not (after[0].isspace() and (not after.lstrip() or after.lstrip()[0].isupper())):
            continue
        a, b = start, i + 1
        while a < b and text[a].isspace():
            a += 1
        if a < b:
            yield a, b, text[a:b]
        start = b
    if start < len(text):
        a, b = start, len(text)
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            yield a, b, text[a:b]


def question_terms(question: str):
    return {w.lower() for w in WORD.findall(question) if w.lower() not in STOP and len(w) > 1}


def question_entity(question: str):
    match = re.search(r"\b(?:for|of)\s+([\w -]+?)\s*\?\s*$", question, re.I)
    return match.group(1).strip().lower() if match else None


def relevance(question: str, sentence: str, kind: str, retrieval_score: float):
    q = question_terms(question)
    words = {x.lower() for x in WORD.findall(sentence)}
    overlap = len(q & words)
    entity = question_entity(question)
    entity_bonus = 5 if entity and entity in sentence.lower() else 0
    question_kind = "number" if re.search(r"\b(?:how many|what (?:metric|latency|capacity|port|rate))\b", question, re.I) else \
                    "method" if re.search(r"\b(?:method|algorithm|procedure|technique)\b", question, re.I) else None
    kind_bonus = 2 if question_kind == kind else 0
    return round(2 * overlap + entity_bonus + kind_bonus + max(retrieval_score, 0.0), 6)


def extract_candidates(page: PageDocument, chunk_index: int, score: float, bounds: tuple[int, int], question: str, document_id: str):
    text = page.page_content
    found = []
    for sentence_start, sentence_end, sentence in source_sentences(text):
        if sentence_end <= bounds[0] or sentence_start >= bounds[1]:
            continue
        for kind, pattern in PATTERNS:
            for match in pattern.finditer(sentence):
                value = match.group("value")
                a, b = sentence_start + match.start("value"), sentence_start + match.end("value")
                if not value.strip() or len(value) > 50 or text[a:b] != value:
                    continue
                found.append({"value": value, "kind": kind, "source": page.source,
                              "document_id": document_id, "page_number": page.page_number,
                              "sentence": sentence, "sentence_start": sentence_start,
                              "sentence_end": sentence_end, "value_start": a, "value_end": b,
                              "chunk_index": chunk_index, "retrieval_score": score,
                              "ranking_score": relevance(question, sentence, kind, score)})
    return found


@dataclass
class PDFIndex:
    document_id: str
    filename: str
    pages: list[PageDocument]
    chunks: list
    vectorizer: TfidfVectorizer
    index: faiss.IndexFlatIP

    @classmethod
    def from_pages(cls, pdf_bytes: bytes, filename: str, pages: list[PageDocument]):
        if not pages:
            raise ValueError("PDF has no extractable pages")
        chunks = chunk_documents(pages)
        if not chunks:
            raise ValueError("PDF has no indexable text")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), norm="l2", max_features=8192)
        vectors = vectorizer.fit_transform([c.chunk_text for c in chunks]).toarray().astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        return cls(hashlib.sha256(pdf_bytes).hexdigest(), filename, pages, chunks, vectorizer, index)

    def retrieve_candidates(self, question: str, top_k: int = TOP_K):
        started = time.monotonic()
        query = self.vectorizer.transform([question]).toarray().astype("float32")
        scores, indices = self.index.search(np.ascontiguousarray(query), min(top_k, len(self.chunks)))
        page_lookup = {(p.source, p.page_number): p for p in self.pages}
        candidates = {}
        retrieved = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            page = page_lookup[(chunk.source, chunk.page_number)]
            hint = chunk.metadata.get("start_index", 0)
            position = page.page_content.find(chunk.chunk_text, max(0, hint - 4))
            if position < 0:
                position = page.page_content.find(chunk.chunk_text)
            if position < 0:
                raise ValueError("Chunk cannot be mapped to exact PDF page text")
            bounds = (position, position + len(chunk.chunk_text))
            retrieved.append({"chunk_index": chunk.chunk_index, "page_number": page.page_number,
                              "score": float(score), "page_start": bounds[0], "page_end": bounds[1],
                              "text": chunk.chunk_text})
            for item in extract_candidates(page, chunk.chunk_index, float(score), bounds, question, self.document_id):
                key = (item["page_number"], item["value_start"], item["value_end"])
                if key not in candidates or item["ranking_score"] > candidates[key]["ranking_score"]:
                    candidates[key] = item
        ranked = sorted(candidates.values(), key=lambda x: (-x["ranking_score"],
                        -x["retrieval_score"], x["page_number"], x["value_start"]))
        unique = []
        seen_values = set()
        for item in ranked:
            key = item["value"].strip().casefold()
            if key not in seen_values:
                unique.append(item)
                seen_values.add(key)
        return {"retrieved": retrieved, "candidates": unique[:24],
                "retrieval_latency_ms": (time.monotonic() - started) * 1000}


def validate_citation(index: PDFIndex, candidate: dict):
    page = next((p for p in index.pages if p.source == candidate["source"] and
                 p.page_number == candidate["page_number"]), None)
    if page is None or candidate["document_id"] != index.document_id:
        return False
    text = page.page_content
    return (text[candidate["sentence_start"]:candidate["sentence_end"]] == candidate["sentence"] and
            text[candidate["value_start"]:candidate["value_end"]] == candidate["value"] and
            candidate["sentence_start"] <= candidate["value_start"] < candidate["value_end"] <= candidate["sentence_end"])


def evidence_decision(index: PDFIndex, question: str, selected: dict, four: list[dict]):
    """Conservative deterministic veto; never substitutes another answer."""
    if not validate_citation(index, selected):
        return "INSUFFICIENT_EVIDENCE", "citation_span_not_in_extracted_page"
    sentence = selected["sentence"].lower()
    entity = question_entity(question)
    if entity and entity not in sentence:
        return "INSUFFICIENT_EVIDENCE", "question_entity_absent_from_selected_sentence"
    asked_field = next((field for field in ("method", "metric", "latency", "capacity", "owner", "port", "precision")
                        if re.search(rf"\b{field}\b", question, re.I)), None)
    if asked_field and not re.search(rf"\b{asked_field}\b", sentence, re.I):
        return "INSUFFICIENT_EVIDENCE", "question_field_absent_from_selected_sentence"
    relevant = question_terms(question) & {w.lower() for w in WORD.findall(sentence)}
    if not relevant:
        return "INSUFFICIENT_EVIDENCE", "question_terms_absent_from_selected_sentence"
    if re.search(r"\b(?:obsolete|superseded|previously|formerly|replaced)\b", sentence):
        return "INSUFFICIENT_EVIDENCE", "selected_sentence_marks_value_obsolete"
    if re.search(r"\bwas\b", sentence) and any(
        c is not selected and c["value"] != selected["value"] and
        (not entity or entity in c["sentence"].lower()) and "current" in c["sentence"].lower()
        for c in four
    ):
        return "INSUFFICIENT_EVIDENCE", "competing_current_value"
    return "ANSWERED", "passed_surface_and_provenance_checks_not_semantic_entailment"
