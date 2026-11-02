import re
import sys
import time
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from scratch_model.bpe_tokenizer import BPETokenizer
    from scratch_model.model import CustomCausalLM
except ImportError:
    from bpe_tokenizer import BPETokenizer
    from model import CustomCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = BASE_DIR / "checkpoint" / "qa_model.pt"
TOKENIZER_PATH = BASE_DIR / "checkpoint" / "bpe_tokenizer.json"

if not CHECKPOINT_PATH.exists() or not TOKENIZER_PATH.exists():
    raise FileNotFoundError("Model checkpoint or tokenizer missing. Please run train.py first.")

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


def generate(prompt: str, max_tokens: int = 60, temperature: float = 0.4):
    input_ids = tokenizer.encode(prompt, add_bos=True)
    if len(input_ids) > checkpoint["max_seq_len"] - 60:
        input_ids = input_ids[-(checkpoint["max_seq_len"] - 60):]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    gen_ids = model.generate(
        idx=input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=30,
        repetition_penalty=1.6,
        stop_token_ids=[tokenizer.eos_id]
    )

    full_output = tokenizer.decode(gen_ids)
    if "Answer:" in full_output:
        return full_output.split("Answer:")[-1].strip()
    return full_output[len(prompt):].strip()


if __name__ == "__main__":
    print(f"Loaded Custom Causal LM ({checkpoint['total_params']:,} parameters)")
    q = input("\nEnter Question: ")
    ctx = input("Enter Context (optional): ")
    prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:" if ctx else f"Question: {q}\n\nAnswer:"
    ans = generate(prompt)
    print(f"\nGenerated Answer:\n{ans}")