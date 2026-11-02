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
    from scratch_model.bpe_tokenizer import BPETokenizer
    from scratch_model.model import CustomCausalLM
    from scratch_model.retriever import BM25Retriever
except ImportError:
    from bpe_tokenizer import BPETokenizer
    from model import CustomCausalLM
    from retriever import BM25Retriever

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = BASE_DIR / "checkpoint" / "qa_model.pt"
TOKENIZER_PATH = BASE_DIR / "checkpoint" / "bpe_tokenizer.json"
PDF_PATH = BASE_DIR.parent / "data" / "sample_pdfs" / "elecricsemple.pdf"

print("=" * 70)
print("COMPREHENSIVE AUDIT & EVALUATION OF CUSTOM CAUSAL LM")
print("=" * 70)
print(f"Device: {DEVICE.upper()} (CPU Optimized)")

if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
    raise FileNotFoundError("Checkpoint or Tokenizer missing. Please run train.py first.")

tokenizer = BPETokenizer().load(TOKENIZER_PATH)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = CustomCausalLM(
    vocab_size=checkpoint["vocab_size"],
    d_model=checkpoint["d_model"],
    n_head=checkpoint["n_head"],
    n_layers=checkpoint["n_layers"],
    max_seq_len=checkpoint["max_seq_len"],
    dropout=0.1,
    pad_idx=checkpoint["pad_idx"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded Custom Causal LM: {checkpoint['total_params']:,} parameters")
print(f"Pretrained Weights Used: 0% (100% Trained from Random Initialization)")
print(f"BPE Subword Vocabulary: {tokenizer.vocab_size} tokens")
print(f"Best Validation Loss: {checkpoint['val_loss']:.4f} (Train Loss: {checkpoint['train_loss']:.4f})")
print("=" * 70)

# ==========================================
# 2. INDEX TEST DOCUMENT (elecricsemple.pdf)
# ==========================================
retriever = BM25Retriever()
num_chunks = retriever.index_pdf(PDF_PATH)
print(f"Indexed PDF: {PDF_PATH.name} ({num_chunks} semantic chunks created)\n")


# ==========================================
# 3. GENERATION FUNCTION
# ==========================================
def answer_query(query: str, temperature: float = 0.35, repetition_penalty: float = 1.6):
    t0 = time.time()
    top_chunks, intent, scores = retriever.retrieve(query, top_k=2)

    context_text = " ".join([c.text for c in top_chunks])
    context_words = context_text.split()[:130]
    trimmed_context = " ".join(context_words)

    prompt = f"Context: {trimmed_context}\n\nQuestion: {query.strip()}\n\nAnswer:"
    input_ids = tokenizer.encode(prompt, add_bos=True)
    if len(input_ids) > checkpoint["max_seq_len"] - 60:
        input_ids = input_ids[-(checkpoint["max_seq_len"] - 60):]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    gen_ids = model.generate(
        idx=input_tensor,
        max_new_tokens=70,
        temperature=temperature,
        top_k=30,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[tokenizer.eos_id]
    )

    new_token_ids = gen_ids[len(input_ids):]
    answer = tokenizer.decode(new_token_ids).strip()

    # Clean up trailing prompt tokens or stop tags if generated
    for stop_word in ["context:", "question:", "<|eos|>", "<|pad|>", "<|sep|>"]:
        if stop_word in answer.lower():
            answer = re.split(re.escape(stop_word), answer, flags=re.IGNORECASE)[0].strip()

    # If answer starts with 'answer:', strip it
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()

    elapsed = time.time() - t0

    pages = sorted(list(set(c.page for c in top_chunks)))
    return answer, top_chunks, pages, intent, scores, elapsed


# ==========================================
# 4. RUN ALL 6 TARGET EVALUATIONS
# ==========================================
test_questions = [
    {
        "id": 1,
        "query": "Tell me about the railway vehicle model and Monte Carlo simulations.",
        "expected": "17 degree-of-freedom multi-body railway vehicle model simulated across 1,000 runs under stochastic operating uncertainties (passenger mass, track irregularities, wheel-rail creep forces)."
    },
    {
        "id": 2,
        "query": "Tell me about this PDF.",
        "expected": "Feasibility study on acceleration data-based robust on-board monitoring of hunting proximity in railway vehicles under uncertainty by Kritikakos et al."
    },
    {
        "id": 3,
        "query": "Give me a summary of this PDF.",
        "expected": "Overview of on-board hunting proximity assessment using lateral acceleration signals and Maximum Lyapunov Exponents under operational uncertainties."
    },
    {
        "id": 4,
        "query": "What is the precise problem statement?",
        "expected": "Detecting proximity to hunting instability on-board in real time using accessible acceleration data before violent lateral limit cycle oscillations arise."
    },
    {
        "id": 5,
        "query": "Explain the methodology.",
        "expected": "Multi-body simulation under Monte Carlo uncertainties, extracting acceleration signals from bogies and wheelsets, estimating Maximum Lyapunov Exponents (MLE) to detect hunting."
    },
    {
        "id": 6,
        "query": "What are the limitations?",
        "expected": "Based on numerical multi-body simulations rather than full-scale revenue-service track testing; Lyapunov exponent estimation sensitivity to measurement noise."
    }
]

for t in test_questions:
    ans, chunks, pages, intent, scores, latency = answer_query(t["query"])

    print(f"[{t['id']}] QUERY: {t['query']}")
    print(f"    Intent: {intent} | Matched Pages: {pages} | BM25 Scores: {scores}")
    print(f"    Top Retrieved Snippet: \"{chunks[0].text[:140]}...\"")
    print(f"    Generated Answer: {ans}")
    print(f"    PDF Ground Truth: {t['expected']}")
    print(f"    Inference Latency: {latency:.3f}s")
    print("-" * 70)

print("\nEVALUATION COMPLETE.")
