import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

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

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
torch.manual_seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = BASE_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "qa_model.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "bpe_tokenizer.json"
DATASET_PATH = BASE_DIR / "training_data.json"

MAX_SEQ_LEN = 256
BATCH_SIZE = 8       # Stable for CPU execution
EPOCHS = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
D_MODEL = 192
N_HEAD = 6
N_LAYERS = 4
VAL_SPLIT = 0.15

print("=" * 65)
print("TRAINING CUSTOM CAUSAL LANGUAGE MODEL FROM SCRATCH")
print("=" * 65)
print(f"Device: {DEVICE.upper()} (CPU Optimized)")


# ==========================================
# 2. LOAD DATASET & TRAIN BPE TOKENIZER
# ==========================================
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_samples = json.load(f)

print(f"Loaded training examples: {len(raw_samples)}")

# Collect text corpus for BPE training
corpus_texts = []
formatted_texts = []

for item in raw_samples:
    ctx = str(item.get("context", "")).strip()
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer", "")).strip()

    if q and a:
        formatted = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer: {a}"
        formatted_texts.append(formatted)
        corpus_texts.append(f"{ctx} {q} {a}")

# Train BPE Tokenizer from scratch
print("Training custom Byte Pair Encoding (BPE) subword tokenizer...")
tokenizer = BPETokenizer()
tokenizer.train(corpus_texts, target_vocab_size=2000, min_pair_freq=2)
tokenizer.save(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

print(f"BPE Tokenizer trained successfully! Vocabulary size: {VOCAB_SIZE} subwords.")
print(f"Tokenizer saved to: {TOKENIZER_PATH}")


# ==========================================
# 3. DATASET & NEXT-TOKEN SEQUENCING
# ==========================================
class CausalLMDataset(Dataset):
    def __init__(self, texts, tokenizer: BPETokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        for t in texts:
            # Format: <|bos|> ... <|eos|>
            toks = self.tokenizer.encode(t, add_bos=True, add_eos=True)
            if len(toks) > self.max_seq_len:
                toks = toks[:self.max_seq_len]
            else:
                toks = toks + [self.tokenizer.pad_id] * (self.max_seq_len - len(toks))

            self.examples.append(toks)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        toks = self.examples[idx]
        # x is input sequence [0 : T-1], y is target sequence [1 : T]
        x = torch.tensor(toks[:-1], dtype=torch.long)
        y = torch.tensor(toks[1:], dtype=torch.long)
        return x, y


# Split into Train / Validation sets
random.shuffle(formatted_texts)
num_val = max(1, int(len(formatted_texts) * VAL_SPLIT))
train_texts = formatted_texts[num_val:]
val_texts = formatted_texts[:num_val]

print(f"Dataset split: {len(train_texts)} train sequences | {len(val_texts)} validation sequences")

train_dataset = CausalLMDataset(train_texts, tokenizer, MAX_SEQ_LEN)
val_dataset = CausalLMDataset(val_texts, tokenizer, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 4. INITIALIZE MODEL FROM SCRATCH
# ==========================================
model = CustomCausalLM(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.1,
    pad_idx=tokenizer.pad_id
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {total_params:,} (100% Randomly Initialized, 0% Pretrained)")
print("=" * 65)
print("COMMENCING CAUSAL LM AUTOREGRESSIVE TRAINING")
print("=" * 65)


# ==========================================
# 5. TRAINING & VALIDATION LOOP
# ==========================================
best_val_loss = float("inf")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, v_loss = model(x, y)
            total_val_loss += v_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    elapsed = time.time() - epoch_start

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f}s",
        flush=True
    )

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "n_head": N_HEAD,
            "n_layers": N_LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
            "pad_idx": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "epoch": epoch,
            "total_params": total_params
        }, CHECKPOINT_PATH)

total_duration = time.time() - start_time
print("=" * 65)
print(f"Training successfully completed in {total_duration:.2f} seconds!")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
print("=" * 65)
