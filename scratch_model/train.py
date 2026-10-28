import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "training_data.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001

print("Training device:", DEVICE)


# ==========================================
# 2. LOAD PDF QUESTION-ANSWER DATA
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    examples = json.load(file)

if not examples:
    raise ValueError("The training dataset is empty.")

# Combine questions and answers into training sequences.
texts = []

for item in examples:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()

    if question and answer:
        texts.append(
            f"Question: {question}\n"
            f"Answer: {answer}\n"
        )

if not texts:
    raise ValueError("No valid question-answer pairs found.")

text = "\n".join(texts)

print("Training examples:", len(texts))
print("Training characters:", len(text))


# ==========================================
# 3. CHARACTER TOKENIZER
# ==========================================

vocab = sorted(set(text))

stoi = {
    character: index
    for index, character in enumerate(vocab)
}

itos = {
    index: character
    for character, index in stoi.items()
}

token_data = torch.tensor(
    [stoi[character] for character in text],
    dtype=torch.long
)

VOCAB_SIZE = len(vocab)

print("Vocabulary size:", VOCAB_SIZE)


# ==========================================
# 4. CREATE TRAINING SEQUENCES
# ==========================================

class TextDataset(Dataset):

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, index):
        x = self.tokens[index:index + self.seq_len]
        y = self.tokens[index + 1:index + self.seq_len + 1]
        return x, y


dataset = TextDataset(token_data, SEQ_LEN)

if len(dataset) == 0:
    raise ValueError(
        "Not enough training text for the selected sequence length."
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training sequences:", len(dataset))


# ==========================================
# 5. TRANSFORMER MODEL
# ==========================================

class TinyTransformer(nn.Module):

    def __init__(self, vocab_size, seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, 64
        )

        self.position_embedding = nn.Embedding(
            seq_len, 64
        )

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2
        )

        self.output = nn.Linear(
            64, vocab_size
        )

    def forward(self, x):
        _, length = x.shape

        positions = torch.arange(
            length,
            device=x.device
        )

        hidden = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        # Causal mask: prevent looking at future characters.
        mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        hidden = self.transformer(
            hidden,
            mask=mask
        )

        return self.output(hidden)


# ==========================================
# 6. INITIALIZE MODEL FROM SCRATCH
# ==========================================

model = TinyTransformer(
    VOCAB_SIZE,
    SEQ_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)


# ==========================================
# 7. TRAIN
# ==========================================

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_function(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss / batches:.4f}"
        )


# ==========================================
# 8. SAVE CHECKPOINT
# ==========================================

checkpoint_path = CHECKPOINT_DIR / "qa_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
    },
    checkpoint_path
)

print("\nTraining completed.")
print("Checkpoint saved to:", checkpoint_path)

