"""
generator.py
------------
Generates a grounded answer from retrieved context using a local LLM.

Primary : FLAN-T5-base (google/flan-t5-base) — runs fully locally, no API key.
Fallback : Groq API (llama3-8b-8192) — set GROQ_API_KEY in .env file.
Fallback2: Google Gemini API       — set GEMINI_API_KEY in .env file.

The prompt explicitly instructs the model to:
  - Answer ONLY from the provided context.
  - Cite page numbers wherever possible.
  - Say "not found" if the context lacks the answer.
  - Never hallucinate or use outside knowledge.
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

# ─────────────────────────────────────────────────────────
# Module-level singletons for local FLAN-T5
# ─────────────────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None


def _get_local_model():
    """
    Lazily load FLAN-T5 tokenizer and model (cached after first call).

    We call model.generate() directly instead of using the HuggingFace
    pipeline() API, because Transformers 5.x removed 'text2text-generation'
    from the pipeline task registry. Direct generate() works on all versions.
    """
    global _local_model, _local_tokenizer
    if _local_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(f"Loading local LLM: {LOCAL_LLM_MODEL} ...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
        _local_model     = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL)
        # Suppress tie_word_embeddings mismatch warning (cosmetic only)
        _local_model.config.tie_word_embeddings = False
        _local_model.eval()   # inference mode
        logger.info("Local LLM loaded.")
    return _local_model, _local_tokenizer


# ─────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: List[TextChunk]) -> str:
    """
    Assemble the RAG prompt.

    The context section lists each retrieved chunk with its source and page
    number so the model can include citations in its answer.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Context {i} | Source: {chunk.source} | Page {chunk.page_number}]\n"
            f"{chunk.chunk_text}"
        )
    context_str = "\n\n".join(context_parts)

    prompt = f"""You are a precise document question-answering assistant.
Use ONLY the context passages provided below to answer the question.
- If the answer is present, quote or paraphrase the relevant part and include the page number.
- If the context does not contain enough information to answer the question, respond with:
  "The document does not contain enough information to answer this question."
- Do NOT use any outside knowledge. Do NOT hallucinate.

=== CONTEXT ===
{context_str}

=== QUESTION ===
{question}

=== ANSWER ==="""
    return prompt


# ─────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Output of one answer-generation call.

    Attributes:
        answer          : The generated answer string.
        source_pages    : Unique page numbers of retrieved chunks.
        source_files    : Unique source filenames.
        generation_time : Wall-clock seconds for LLM inference.
        model_used      : Which model produced the answer.
        prompt          : The full prompt that was sent (for debugging).
    """
    answer: str
    source_pages: List[int]
    source_files: List[str]
    generation_time: float
    model_used: str
    prompt: str


# ─────────────────────────────────────────────────────────
# Generators (local + remote fallbacks)
# ─────────────────────────────────────────────────────────

def _generate_local(prompt: str) -> tuple[str, str]:
    """
    Generate answer using FLAN-T5 locally via direct model.generate().

    We use model.generate() + tokenizer.decode() instead of the pipeline API
    because Transformers 5.x removed the 'text2text-generation' pipeline task.
    This approach works on Transformers 2.x through 5.x.
    """
    import torch
    model, tokenizer = _get_local_model()

    # Tokenise the prompt; FLAN-T5 has a 512-token input limit
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,      # silently truncate if prompt is too long
        padding=False,
    )

    with torch.no_grad():     # disable gradient computation for inference
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,              # greedy decoding (deterministic)
        )

    # Decode only the generated tokens (skip special tokens)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if not answer:
        answer = "The document does not contain enough information to answer this question."
    return answer, LOCAL_LLM_MODEL


def _generate_groq(prompt: str) -> tuple[str, str]:
    """Generate answer using Groq API. Returns (answer, model_name)."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()
    return answer, f"groq/{GROQ_MODEL}"


def _generate_gemini(prompt: str) -> tuple[str, str]:
    """Generate answer using Google Gemini API. Returns (answer, model_name)."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    answer = response.text.strip()
    return answer, f"gemini/{GEMINI_MODEL}"


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: List[TextChunk],
    prefer_local: bool = True,
) -> GenerationResult:
    """
    Generate a grounded answer from retrieved context chunks.

    Strategy (in order):
      1. Local FLAN-T5 (if prefer_local=True)
      2. Groq API       (if GROQ_API_KEY is set)
      3. Gemini API     (if GEMINI_API_KEY is set)

    Parameters
    ----------
    question     : The user's question.
    chunks       : Retrieved TextChunk objects from the retriever.
    prefer_local : If True, try local FLAN-T5 first.

    Returns
    -------
    GenerationResult
    """
    if not chunks:
        return GenerationResult(
            answer="No relevant context was found in the document. "
                   "Please ensure the PDF is indexed and try rephrasing your question.",
            source_pages=[],
            source_files=[],
            generation_time=0.0,
            model_used="none",
            prompt="",
        )

    prompt = _build_prompt(question, chunks)

    # Collect unique page numbers and filenames for citations
    source_pages = sorted({c.page_number for c in chunks})
    source_files = sorted({c.source for c in chunks})

    # ── Try generators in priority order ───────────────────
    t0 = time.perf_counter()
    answer = ""
    model_used = "unknown"

    if prefer_local:
        try:
            answer, model_used = _generate_local(prompt)
        except Exception as exc:
            logger.warning(f"Local LLM failed: {exc}. Trying Groq...")
            prefer_local = False  # fall through

    if not answer and GROQ_API_KEY:
        try:
            answer, model_used = _generate_groq(prompt)
        except Exception as exc:
            logger.warning(f"Groq API failed: {exc}. Trying Gemini...")

    if not answer and GEMINI_API_KEY:
        try:
            answer, model_used = _generate_gemini(prompt)
        except Exception as exc:
            logger.error(f"Gemini API also failed: {exc}")
            answer = "All LLM backends failed. Please check your setup."
            model_used = "error"

    if not answer:
        answer = (
            "No LLM backend is available. "
            "Please install transformers for local FLAN-T5 or set GROQ_API_KEY."
        )

    gen_time = round(time.perf_counter() - t0, 4)
    logger.info(f"Answer generated in {gen_time}s using {model_used}")

    return GenerationResult(
        answer=answer,
        source_pages=source_pages,
        source_files=source_files,
        generation_time=gen_time,
        model_used=model_used,
        prompt=prompt,
    )
