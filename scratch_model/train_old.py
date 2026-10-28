import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("Training device:", DEVICE)
print("=" * 50)


# ==========================================
# 2. TINY TRANSFORMER MODEL
# ==========================================

class TinyTransformer(nn.Module):

    def __init__(self, vocab_size, seq_len=32):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, 64)
        self.position_embedding = nn.Embedding(seq_len, 64)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )

        self.output = nn.Linear(64, vocab_size)

    def forward(self, x):
        batch_size, length = x.shape

        positions = torch.arange(
            length,
            device=x.device
        )

        embeddings = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        # Prevent the model from seeing future characters.
        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        hidden = self.transformer(
            embeddings,
            mask=mask
        )

        logits = self.output(hidden)

        return logits


# ==========================================
# 3. TRAINING TEXT
# ==========================================

text = """
Machine learning enables computers to learn patterns from data.
Supervised learning uses labeled examples.
Unsupervised learning discovers patterns in unlabeled data.
A neural network contains layers of trainable parameters.
A transformer uses attention to process sequences of tokens.
Language models learn to predict the next token in a sequence.
Question answering systems retrieve relevant evidence from documents.
Scientific research uses experiments to test hypotheses.
"""


# ==========================================
# 4. CHARACTER TOKENIZER
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

data = torch.tensor(
    [stoi[character] for character in text],
    dtype=torch.long
)

print("Vocabulary size:", len(vocab))
print("Training text length:", len(data))


# ==========================================
# 5. DATASET
# ==========================================

SEQ_LEN = 32


class TextDataset(Dataset):

    def __init__(self, token_data):
        self.data = token_data

    def __len__(self):
        return max(0, len(self.data) - SEQ_LEN)

    def __getitem__(self, index):

        x = self.data[index:index + SEQ_LEN]

        y = self.data[
            index + 1:index + SEQ_LEN + 1
        ]

        return x, y


dataset = TextDataset(data)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# ==========================================
# 6. INITIALIZE MODEL
# ==========================================

model = TinyTransformer(
    vocab_size=len(vocab),
    seq_len=SEQ_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

loss_function = nn.CrossEntropyLoss()

print("Model initialized successfully.")
print("Total parameters:", sum(
    parameter.numel()
    for parameter in model.parameters()
))


# ==========================================
# 7. TRAIN MODEL FROM SCRATCH
# ==========================================

EPOCHS = 100

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
            logits.reshape(-1, len(vocab)),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    average_loss = total_loss / batches

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

output_dir = Path(__file__).resolve().parent / "checkpoint"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = output_dir / "model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "seq_len": SEQ_LEN,
        "vocab_size": len(vocab)
    },
    checkpoint_path
)

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("Model saved at:")
print(checkpoint_path)
print("=" * 50)