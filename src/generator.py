"""
generator.py
------------
Generates a grounded answer from retrieved context using LLM backends.

Priority:
  1. Groq API       (if GROQ_API_KEY is available) — ultra-fast Llama-3 / GPT-OSS
  2. Gemini API     (if GEMINI_API_KEY is available) — Google Gemini Flash
  3. Local FLAN-T5  (if prefer_local=True) — PyTorch local inference
  4. Extractive QA  (always available) — pure Python TF-IDF fallback (zero-crash)

The prompt explicitly instructs the model to:
  - Answer ONLY from the provided context.
  - Cite page numbers wherever possible ([Page N]).
  - Summarize accurately if a summary is requested.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
        import os
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


def _build_prompt(question: str, chunks: List[TextChunk]) -> str:
    """Assemble the grounded RAG prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Context Passage {i} | File: {chunk.source} | Page {chunk.page_number}]\n"
            f"{chunk.chunk_text.strip()}"
        )
    context_str = "\n\n".join(context_parts)

    prompt = f"""You are an expert academic research assistant.
Answer the user's question using ONLY the context passages provided below.

INSTRUCTIONS:
- Base your answer strictly on the provided context.
- Cite the relevant page numbers whenever referring to facts from the text (e.g. [Page 3]).
- If the question asks for a summary or overview, provide a concise synthesis of the key points present in the context.
- If the context does not contain sufficient information to answer the question, state clearly: "The document context provided does not contain enough information to answer this question."
- Do NOT hallucinate or infer facts outside the context.

=== CONTEXT PASSAGES ===
{context_str}

=== QUESTION ===
{question}

=== GROUNDED ANSWER ==="""
    return prompt


@dataclass
class GenerationResult:
    """Output of an answer generation execution."""
    answer: str
    source_pages: List[int]
    source_files: List[str]
    generation_time: float
    model_used: str
    prompt: str


def _generate_local(prompt: str) -> tuple[str, str]:
    """Generate answer using local FLAN-T5."""
    import os, torch
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
        answer = "The document context provided does not contain enough information to answer this question."
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
        return best.chunk_text[:350].strip() + "...", "extractive-tfidf"

    def score_chunk(chunk_text: str) -> float:
        words = set(re.findall(r'\b[a-z]{3,}\b', chunk_text.lower()))
        return len(q_words & words) / max(len(q_words), 1)

    scored = sorted(chunks, key=lambda c: score_chunk(c.chunk_text), reverse=True)
    best_chunk = scored[0]

    sentences = re.split(r'(?<=[.!?])\s+', best_chunk.chunk_text.strip())
    if len(sentences) == 1:
        answer = sentences[0]
    else:
        best_sent = max(sentences, key=lambda s: score_chunk(s))
        idx = sentences.index(best_sent)
        context = sentences[max(0, idx-1):idx+2]
        answer = " ".join(context).strip()

    if not answer or len(answer) < 10:
        answer = best_chunk.chunk_text[:400].strip()

    return answer, "extractive-tfidf"


def _generate_groq(prompt: str, api_key: str) -> tuple[str, str]:
    """Generate answer using Groq API."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()
    return answer, f"groq/{GROQ_MODEL}"


def _generate_gemini(prompt: str, api_key: str) -> tuple[str, str]:
    """Generate answer using Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return answer, f"gemini/{GEMINI_MODEL}"


def generate_answer(
    question: str,
    chunks: List[TextChunk],
    prefer_local: bool = False,
) -> GenerationResult:
    """Generate a grounded answer from retrieved context chunks."""
    if not chunks:
        return GenerationResult(
            answer="No relevant document context was found. Please ensure the PDF is indexed properly.",
            source_pages=[],
            source_files=[],
            generation_time=0.0,
            model_used="none",
            prompt="",
        )

    prompt = _build_prompt(question, chunks)
    source_pages = sorted({c.page_number for c in chunks})
    source_files = sorted({c.source for c in chunks})

    t0 = time.perf_counter()
    answer = ""
    model_used = "unknown"

    # Dynamic API key lookup (reads live .env overrides)
    active_groq_key   = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    active_gemini_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY

    # 1. Groq LLM (Primary)
    if active_groq_key:
        try:
            answer, model_used = _generate_groq(prompt, active_groq_key)
            logger.info(f"Answer generated via Groq ({model_used})")
        except Exception as exc:
            logger.warning(f"Groq API call failed: {exc}")

    # 2. Gemini API (Secondary)
    if not answer and active_gemini_key:
        try:
            answer, model_used = _generate_gemini(prompt, active_gemini_key)
            logger.info(f"Answer generated via Gemini ({model_used})")
        except Exception as exc:
            logger.warning(f"Gemini API call failed: {exc}")

    # 3. Local FLAN-T5
    if not answer and prefer_local:
        try:
            answer, model_used = _generate_local(prompt)
            logger.info(f"Answer generated via local LLM ({model_used})")
        except Exception as exc:
            logger.warning(f"Local LLM failed: {exc}")

    # 4. Extractive QA (Guaranteed Fallback)
    if not answer:
        logger.info("Using extractive QA fallback engine.")
        answer, model_used = _generate_extractive(question, chunks)

    gen_time = round(time.perf_counter() - t0, 4)
    return GenerationResult(
        answer=answer,
        source_pages=source_pages,
        source_files=source_files,
        generation_time=gen_time,
        model_used=model_used,
        prompt=prompt,
    )
