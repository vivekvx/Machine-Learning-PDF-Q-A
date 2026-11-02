"""
GPT-2 Fine-Tuning Script for PDF Question Answering
=====================================================
Fine-tunes GPT-2 Small (117M parameters) on our custom QA dataset.

Model: GPT-2 Small (pretrained by OpenAI, 117M parameters)
Training: Full fine-tuning on our instruction-style QA dataset
Result: A fine-tuned model that generates structured answers from PDF context

NOTE: This model uses GPT-2 pretrained weights as a foundation and is
fine-tuned on our custom dataset. The fine-tuned model is saved as our
project checkpoint. The pretrained weights provide English language
understanding; the fine-tuning teaches it our QA format and domain.
"""

import json
import random
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
torch.manual_seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "training_data_v2.json"
OUTPUT_DIR = BASE_DIR / "checkpoint" / "gpt2_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"  # GPT-2 Small: 117M parameters

# Training hyperparameters
MAX_SEQ_LEN = 320
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch size = 8
EPOCHS = 2
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 20
VAL_SPLIT = 0.1
MAX_GRAD_NORM = 1.0

print("=" * 65)
print("FINE-TUNING GPT-2 SMALL FOR PDF QUESTION ANSWERING")
print("=" * 65)
print(f"Device: {DEVICE.upper()}")
print(f"Base Model: {MODEL_NAME} (117M parameters, pretrained)")
print(f"Training Approach: Full fine-tuning on custom QA dataset")
print(f"Max Sequence Length: {MAX_SEQ_LEN}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")


# ==========================================
# 2. LOAD TOKENIZER & MODEL
# ==========================================
print("\nLoading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Add special tokens for our QA format
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(DEVICE)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Model Size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.1f} MB")


# ==========================================
# 3. LOAD AND PREPARE DATASET
# ==========================================
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        f"Please run prepare_data_v2.py first."
    )

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_samples = json.load(f)

print(f"\nLoaded {len(raw_samples)} training examples")


class QADataset(Dataset):
    """
    Formats each QA pair as:
        Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}<|endoftext|>

    The loss is computed ONLY on the answer portion (labels for context+question are -100).
    This teaches the model to generate answers given context, not to memorize context.
    """

    def __init__(self, samples, tokenizer, max_seq_len):
        self.examples = []
        self.tokenizer = tokenizer
        skipped = 0

        for item in samples:
            ctx = str(item.get("context", "")).strip()
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()

            if not q or not a:
                skipped += 1
                continue

            # Build prompt (context + question) and completion (answer)
            prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer: "
            completion = a + tokenizer.eos_token

            # Tokenize separately to know where answer starts
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)

            # If total exceeds max length, truncate context (keep question + answer intact)
            total_len = len(prompt_ids) + len(completion_ids)
            if total_len > max_seq_len:
                # How much to trim from prompt
                overflow = total_len - max_seq_len
                if overflow < len(prompt_ids) - 20:  # Keep at least some context
                    prompt_ids = prompt_ids[overflow:]
                else:
                    # Answer itself is too long, truncate answer
                    completion_ids = completion_ids[:max_seq_len - len(prompt_ids)]
                    if completion_ids and completion_ids[-1] != tokenizer.eos_token_id:
                        completion_ids[-1] = tokenizer.eos_token_id

            input_ids = prompt_ids + completion_ids

            # Labels: -100 for prompt tokens (no loss), actual ids for answer tokens
            labels = [-100] * len(prompt_ids) + completion_ids

            # Pad to max_seq_len
            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1 if i < (len(input_ids) - pad_len) else 0
                              for i in range(len(input_ids))]

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        if skipped:
            print(f"  Skipped {skipped} incomplete examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# Split into train / validation
random.shuffle(raw_samples)
num_val = max(1, int(len(raw_samples) * VAL_SPLIT))
train_samples = raw_samples[num_val:]
val_samples = raw_samples[:num_val]

print(f"Dataset split: {len(train_samples)} train | {len(val_samples)} validation")

train_dataset = QADataset(train_samples, tokenizer, MAX_SEQ_LEN)
val_dataset = QADataset(val_samples, tokenizer, MAX_SEQ_LEN)

print(f"Processed: {len(train_dataset)} train sequences | {len(val_dataset)} val sequences")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 4. OPTIMIZER & SCHEDULER
# ==========================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8
)

total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=1e-6
)


# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\n" + "=" * 65)
print("COMMENCING FINE-TUNING")
print("=" * 65)

best_val_loss = float("inf")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    num_train_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader, 1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / GRADIENT_ACCUMULATION
        loss.backward()

        total_train_loss += outputs.loss.item()
        num_train_batches += 1

        if step % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Progress indicator every 20 batches
        if step % 20 == 0:
            avg_so_far = total_train_loss / num_train_batches
            print(f"  Epoch {epoch} | Batch {step}/{len(train_loader)} | Running Loss: {avg_so_far:.4f}", flush=True)

    # Final gradient step if needed
    if len(train_loader) % GRADIENT_ACCUMULATION != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / max(num_train_batches, 1)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_val_loss += outputs.loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / max(num_val_batches, 1)
    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
        f"Time: {epoch_time:.1f}s",
        flush=True
    )

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"  -> New best validation loss! Saving checkpoint...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Save training metadata
        meta = {
            "base_model": MODEL_NAME,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_val_loss": best_val_loss,
            "best_train_loss": avg_train_loss,
            "best_epoch": epoch,
            "total_epochs": EPOCHS,
            "max_seq_len": MAX_SEQ_LEN,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "training_examples": len(train_dataset),
            "device": DEVICE,
            "training_approach": "Full fine-tuning of GPT-2 Small on custom QA dataset",
            "pretrained_weights": "GPT-2 Small (OpenAI, 117M params)",
        }
        with open(OUTPUT_DIR / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)


# ==========================================
# 6. TRAINING SUMMARY
# ==========================================
total_time = time.time() - start_time
print("\n" + "=" * 65)
print("FINE-TUNING COMPLETE")
print("=" * 65)
print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {OUTPUT_DIR}")
print(f"Base Model: GPT-2 Small (117M params, pretrained by OpenAI)")
print(f"Fine-tuning: Full model on {len(train_dataset)} QA examples")
print("=" * 65)


# ==========================================
# 7. QUICK GENERATION TEST
# ==========================================
print("\n--- Quick Generation Test ---")
model.eval()

test_prompt = "Context: Railway hunting is a limit cycle oscillation that occurs above the critical speed.\n\nQuestion: What is hunting?\n\nAnswer:"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Prompt: What is hunting?")
print(f"Generated: {generated_text.strip()}")
print("=" * 65)

