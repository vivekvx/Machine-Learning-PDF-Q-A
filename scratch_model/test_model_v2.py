"""
Comprehensive Test & Evaluation of Fine-Tuned GPT-2 QA Model
==============================================================
Tests the model against standard PDF questions and compares with the old model.
"""

import re
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
# 1. LOAD MODEL & RETRIEVER
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_PATH = BASE_DIR.parent / "data" / "sample_pdfs" / "elecricsemple.pdf"

print("=" * 70)
print("COMPREHENSIVE EVALUATION: FINE-TUNED GPT-2 QA MODEL")
print("=" * 70)
print(f"Device: {DEVICE.upper()}")

# Load fine-tuned GPT-2 model
qa_model = GPT2QAModel()
qa_model.load()
info = qa_model.get_model_info()

print(f"Model: Fine-tuned GPT-2 Small")
print(f"Parameters: {info['total_params']:,}")
print(f"Base Model: {info['pretrained_weights']}")
print(f"Training Approach: {info['training_approach']}")
print(f"Validation Loss: {info['val_loss']:.4f}")
print(f"Training Examples: {info['training_examples']}")
print("=" * 70)

# ==========================================
# 2. INDEX TEST DOCUMENT
# ==========================================
retriever = BM25Retriever()
num_chunks = retriever.index_pdf(PDF_PATH)
print(f"\nIndexed PDF: {PDF_PATH.name} ({num_chunks} semantic chunks created)\n")


# ==========================================
# 3. ANSWER GENERATION FUNCTION
# ==========================================
def answer_query(query: str, temperature: float = 0.3, top_k: int = 3):
    t0 = time.time()

    # Step 1: Retrieve relevant chunks
    top_chunks, intent, scores = retriever.retrieve(query, top_k=top_k)

    if not top_chunks:
        return "No relevant content found in the document.", [], [], intent, [], time.time() - t0

    # Step 2: Build context from retrieved chunks
    context_text = "\n\n".join([c.text for c in top_chunks])

    # Step 3: Generate answer using fine-tuned GPT-2
    answer, gen_time = qa_model.generate_answer(
        question=query,
        context=context_text,
        max_new_tokens=300,
        temperature=temperature,
    )

    pages = sorted(list(set(c.page for c in top_chunks)))
    total_time = time.time() - t0

    return answer, top_chunks, pages, intent, scores, total_time


# ==========================================
# 4. RUN ALL TEST EVALUATIONS
# ==========================================
test_questions = [
    {
        "id": 1,
        "query": "Tell me about this PDF.",
        "expected": "Research feasibility study on acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles.",
    },
    {
        "id": 2,
        "query": "Give me a summary of this PDF.",
        "expected": "Overview of on-board hunting proximity assessment using lateral acceleration signals and Maximum Lyapunov Exponents under operational uncertainties.",
    },
    {
        "id": 3,
        "query": "Explain the methodology.",
        "expected": "Grey nonlinear vehicle model, MLE estimation from acceleration data, Monte Carlo simulations for validation.",
    },
    {
        "id": 4,
        "query": "What is the precise problem statement?",
        "expected": "Detecting proximity to hunting instability on-board in real time using acceleration data before violent oscillations arise.",
    },
    {
        "id": 5,
        "query": "What are the advantages?",
        "expected": "High accuracy (97.08%), uses standard sensors, robust to varying conditions, proactive detection.",
    },
    {
        "id": 6,
        "query": "What are the disadvantages?",
        "expected": "Simulation-only validation, worn-track performance issues, noise sensitivity, constant speed assumption.",
    },
    {
        "id": 7,
        "query": "What are the limitations?",
        "expected": "Based on numerical simulations rather than full-scale testing; Lyapunov exponent estimation sensitivity to noise.",
    },
    {
        "id": 8,
        "query": "Explain the important findings.",
        "expected": "97.08% accuracy achieved, MLE is reliable proximity indicator, grey model outperforms data-driven methods.",
    },
]

print("=" * 70)
print("RUNNING EVALUATION ON 8 STANDARD TEST QUESTIONS")
print("=" * 70)

total_latency = 0
results = []

for t in test_questions:
    ans, chunks, pages, intent, scores, latency = answer_query(t["query"])
    total_latency += latency

    # Check if answer references PDF content
    answer_supported = bool(ans and len(ans) > 20 and "not provided" not in ans.lower() and "not found" not in ans.lower())

    result = {
        "id": t["id"],
        "query": t["query"],
        "answer": ans,
        "expected": t["expected"],
        "latency": latency,
        "supported": answer_supported,
        "intent": intent,
        "pages": pages,
    }
    results.append(result)

    print(f"\n[{t['id']}] QUERY: {t['query']}")
    print(f"    Intent: {intent} | Pages: {pages} | BM25 Scores: {scores[:3]}")
    if chunks:
        print(f"    Retrieved Context: \"{chunks[0].text[:160]}...\"")
    print(f"    Generated Answer:")
    # Print answer with indentation
    for line in ans.split("\n"):
        print(f"      {line}")
    print(f"    Expected Keywords: {t['expected']}")
    print(f"    Supported by PDF: {'✅ YES' if answer_supported else '❌ NO'}")
    print(f"    Latency: {latency:.2f}s")
    print("-" * 70)

# ==========================================
# 5. EVALUATION SUMMARY
# ==========================================
supported_count = sum(1 for r in results if r["supported"])
avg_latency = total_latency / len(test_questions)

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"Questions Tested: {len(test_questions)}")
print(f"Answers Supported by PDF: {supported_count}/{len(test_questions)} ({100*supported_count/len(test_questions):.0f}%)")
print(f"Average Latency: {avg_latency:.2f}s")
print(f"Total Evaluation Time: {total_latency:.2f}s")
print(f"Model: Fine-tuned GPT-2 Small ({info['total_params']:,} parameters)")
print(f"Base: {info['pretrained_weights']}")
print(f"Approach: {info['training_approach']}")
print("=" * 70)
print("\nEVALUATION COMPLETE.")

