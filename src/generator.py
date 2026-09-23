"""
generator.py
------------
Generates a grounded answer from retrieved context using LLM backends.

Priority:
  1. Groq API       (if GROQ_API_KEY is available) — ultra-fast GPT-OSS
  2. Gemini API     (if GEMINI_API_KEY is available) — Google Gemini Flash
  3. Local FLAN-T5  (if prefer_local=True) — PyTorch local inference
  4. Extractive QA  (always available) — pure Python TF-IDF fallback (zero-crash)

Key improvements over v1:
  - Academic-grade system + user prompt with explicit synthesis instructions
  - Intent-aware prompting (review articles vs experimental papers)
  - Richer context format with similarity scores
  - Instructed to explain what IS missing rather than refuse entirely
  - MAX_NEW_TOKENS raised to 1024 for adequate academic synthesis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time
import logging
import os

from src.chunking import TextChunk
from src.config import (
    LOCAL_LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE,
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

# Singletons for local FLAN-T5
_local_model     = None
_local_tokenizer = None


def _get_local_model():
    """Lazily load local FLAN-T5 model and tokenizer."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(2)

        logger.info(f"Loading local LLM: {LOCAL_LLM_MODEL}...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL, use_fast=True)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL, low_cpu_mem_usage=True)
        _local_model.config.tie_word_embeddings = False
        _local_model.eval()
        logger.info("Local LLM loaded successfully.")
    return _local_model, _local_tokenizer


# ─────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────

# System prompt for Groq / Gemini chat completions
_SYSTEM_PROMPT = """You are an academic PDF question-answering assistant. Answer only using the retrieved document context. You may synthesize across multiple chunks. If the document is a review article, explain themes, mechanisms, findings, and conclusions instead of expecting experimental metrics. If something is missing, say exactly what is missing, but still answer the parts supported by the document. Do not hallucinate. Always cite source pages."""

_USER_PROMPT_TEMPLATE = """Question:
{question}

Document type:
{document_type}

Query type:
{query_type}

Retrieved context:
{context}

Instructions:
- Use only the retrieved context.
- For broad questions, synthesize across chunks.
- Do not say “not enough information” if a partial grounded answer is possible.
- If the document does not report metrics or experiments, explain that clearly.
- Include source page references like (p. 1), (p. 3).
- Use bullets for clarity.
"""


def _format_context(chunks: List[TextChunk], scores: Optional[List[float]] = None) -> str:
    """Format retrieved chunks into a rich context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        score_str = ""
        if scores and i - 1 < len(scores):
            score_str = f" | Similarity {scores[i-1]:.3f}"
        heading = chunk.metadata.get("section_heading", "Unlabelled section")
        header = f"[Source {i} | {chunk.source} | Page {chunk.page_number} | Section: {heading}{score_str}]"
        parts.append(f"{header}\n{chunk.chunk_text.strip()}")
    return "\n\n".join(parts)


def infer_document_type(chunks: List[TextChunk]) -> str:
    """Conservatively distinguish review-like documents from empirical papers."""
    sample = " ".join(c.chunk_text.lower() for c in chunks[:4])
    if any(term in sample for term in ("systematic review", "literature review", "review article")):
        return "review article"
    if any(term in sample for term in ("methods", "participants", "experiment", "dataset")):
        return "empirical research paper"
    return "academic document (type not established from retrieved context)"


def _build_prompt(
    question: str,
    chunks: List[TextChunk],
    scores: Optional[List[float]] = None,
    intent: str = "unknown",
    document_type: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) pair for chat-style APIs (Groq/Gemini).
    Also returns a single merged prompt for single-turn APIs (local / extractive).
    """
    context_str = _format_context(chunks, scores)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        document_type=document_type or infer_document_type(chunks),
        query_type=intent,
        context=context_str,
    )
    return _SYSTEM_PROMPT, user_prompt


def _build_single_prompt(
    question: str,
    chunks: List[TextChunk],
    scores: Optional[List[float]] = None,
    intent: str = "unknown",
    document_type: Optional[str] = None,
) -> str:
    """Single merged prompt for non-chat APIs (local FLAN-T5)."""
    sys_p, usr_p = _build_prompt(question, chunks, scores, intent, document_type)
    return f"{sys_p}\n\n{usr_p}"


# ─────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Output of an answer generation execution."""
    answer: str
    source_pages: List[int]
    source_files: List[str]
    generation_time: float
    model_used: str
    prompt: str
    intent: str = "unknown"
    query_expanded: bool = False
    fallback_triggered: bool = False


# ─────────────────────────────────────────────────────────
# Backend generators
# ─────────────────────────────────────────────────────────

def _generate_groq(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> tuple[str, str]:
    """Generate answer using Groq API (chat completion)."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()
    return answer, f"groq/{GROQ_MODEL}"


def _generate_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> tuple[str, str]:
    """Generate answer using Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_prompt)
    answer = response.text.strip()
    return answer, f"gemini/{GEMINI_MODEL}"


def _generate_local(prompt: str) -> tuple[str, str]:
    """Generate answer using local FLAN-T5."""
    import torch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(2)

    model, tokenizer = _get_local_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=False,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    del output_ids
    if not answer:
        answer = "The local model could not produce an answer from the retrieved context."
    return answer, LOCAL_LLM_MODEL


def _generate_extractive(question: str, chunks: List[TextChunk]) -> tuple[str, str]:
    """Pure Python extractive fallback (zero crash, zero dependencies)."""
    import re

    stopwords = {
        "a","an","the","is","are","was","were","what","which","who","how","why",
        "when","where","does","do","did","can","could","will","would","should",
        "in","of","to","for","and","or","but","with","on","at","by","from",
        "that","this","these","those","it","its","be","been","being","have",
        "has","had","not","no","so","if","as","about","into","than","more",
    }
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower())) - stopwords

    if not q_words:
        best = chunks[0]
        return f"{best.chunk_text[:500].strip()}... (p. {best.page_number})", "extractive-tfidf"

    def score_chunk(chunk_text: str) -> float:
        words = set(re.findall(r'\b[a-z]{3,}\b', chunk_text.lower()))
        return len(q_words & words) / max(len(q_words), 1)

    scored = sorted(chunks, key=lambda c: score_chunk(c.chunk_text), reverse=True)

    # For extractive, concatenate top 2 chunks for richer answer
    top_chunks = scored[:2]
    answer_parts = []
    for c in top_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', c.chunk_text.strip())
        best_sents = sorted(sentences, key=score_chunk, reverse=True)[:3]
        answer_parts.append(f"{' '.join(best_sents)} (p. {c.page_number})")

    answer = "\n\n".join(answer_parts).strip()
    if not answer or len(answer) < 10:
        answer = f"{chunks[0].chunk_text[:500].strip()} (p. {chunks[0].page_number})"

    return answer, "extractive-tfidf"


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: List[TextChunk],
    prefer_local: bool = False,
    scores: Optional[List[float]] = None,
    intent: str = "unknown",
    document_type: Optional[str] = None,
    query_expanded: bool = False,
    fallback_triggered: bool = False,
) -> GenerationResult:
    """
    Generate a grounded answer from retrieved context chunks.

    Parameters
    ----------
    question          : Raw user question.
    chunks            : Retrieved TextChunk objects (already ordered by score).
    prefer_local      : If True, try local FLAN-T5 before extractive fallback.
    scores            : Corresponding similarity scores for each chunk.
    intent            : Classified query intent from query_intelligence.
    query_expanded    : Whether query expansion was used in retrieval.
    fallback_triggered: Whether fallback retrieval was triggered.
    """
    if not chunks:
        return GenerationResult(
            answer="No relevant document context was found. Please ensure the PDF is indexed properly.",
            source_pages=[],
            source_files=[],
            generation_time=0.0,
            model_used="none",
            prompt="",
            intent=intent,
            query_expanded=query_expanded,
            fallback_triggered=fallback_triggered,
        )

    doc_type = document_type or infer_document_type(chunks)
    sys_prompt, user_prompt = _build_prompt(question, chunks, scores, intent, doc_type)
    single_prompt = _build_single_prompt(question, chunks, scores, intent, doc_type)

    source_pages = sorted({c.page_number for c in chunks})
    source_files = sorted({c.source for c in chunks})

    # Log context being sent
    context_len = len(user_prompt)
    logger.info(
        f"[Generator] query_type={intent}, document_type={doc_type}, chunks={len(chunks)}, "
        f"context_chars={context_len}"
    )
    logger.debug(f"[Generator] First 300 chars of context: {user_prompt[:300]}")

    t0 = time.perf_counter()
    answer = ""
    model_used = "unknown"

    # Dynamic API key lookup (reads live .env overrides)
    active_groq_key   = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    active_gemini_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY

    # 1. Groq LLM (Primary)
    if active_groq_key:
        try:
            answer, model_used = _generate_groq(sys_prompt, user_prompt, active_groq_key)
            logger.info(f"Answer generated via Groq ({model_used}), len={len(answer)}")
        except Exception as exc:
            logger.warning(f"Groq API call failed: {exc}")

    # 2. Gemini API (Secondary)
    if not answer and active_gemini_key:
        try:
            answer, model_used = _generate_gemini(sys_prompt, user_prompt, active_gemini_key)
            logger.info(f"Answer generated via Gemini ({model_used})")
        except Exception as exc:
            logger.warning(f"Gemini API call failed: {exc}")

    # 3. Local FLAN-T5
    if not answer and prefer_local:
        try:
            answer, model_used = _generate_local(single_prompt)
            logger.info(f"Answer generated via local LLM ({model_used})")
        except Exception as exc:
            logger.warning(f"Local LLM failed: {exc}")

    # 4. Extractive QA (Guaranteed Fallback)
    if not answer:
        logger.info("Using extractive QA fallback engine.")
        answer, model_used = _generate_extractive(question, chunks)

    gen_time = round(time.perf_counter() - t0, 4)
    logger.info(f"[Generator] model={model_used}, latency={gen_time:.3f}s")
    return GenerationResult(
        answer=answer,
        source_pages=source_pages,
        source_files=source_files,
        generation_time=gen_time,
        model_used=model_used,
        prompt=single_prompt,
        intent=intent,
        query_expanded=query_expanded,
        fallback_triggered=fallback_triggered,
    )
