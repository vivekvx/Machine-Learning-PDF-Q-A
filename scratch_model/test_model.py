import torch
import torch.nn as nn
from pathlib import Path

# ==========================================
# 1. LOAD CHECKPOINT
# ==========================================

checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoint"
    / "model.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

vocab = checkpoint["vocab"]
SEQ_LEN = checkpoint["seq_len"]
VOCAB_SIZE = checkpoint["vocab_size"]

# Build consistent character mappings
stoi = {char: i for i, char in enumerate(vocab)}
itos = {i: char for i, char in enumerate(vocab)}

# Validate the saved vocabulary
assert len(stoi) == VOCAB_SIZE, "Vocabulary size mismatch!"
assert len(itos) == VOCAB_SIZE, "Token mapping mismatch!"

print("Vocabulary size:", VOCAB_SIZE)
print("Checkpoint loaded successfully.")


# ==========================================
# 2. RECREATE MODEL
# ==========================================

class TinyTransformer(nn.Module):

    def __init__(self, vocab_size, seq_len=32):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, 64)
        self.position_embedding = nn.Embedding(seq_len, 64)

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

        self.output = nn.Linear(64, vocab_size)

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

        mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        hidden = self.transformer(hidden, mask=mask)

        return self.output(hidden)


model = TinyTransformer(
    VOCAB_SIZE,
    SEQ_LEN
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Trained model loaded successfully.")


# ==========================================
# 3. GENERATE TEXT
# ==========================================

def generate_text(prompt, max_new_tokens=100):

    unknown = [char for char in prompt if char not in stoi]

    if unknown:
        raise ValueError(
            f"Prompt contains characters outside the vocabulary: {unknown}"
        )

    # Convert prompt characters into token IDs
    tokens = torch.tensor(
        [[stoi[char] for char in prompt]],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            input_tokens = tokens[:, -SEQ_LEN:]

            logits = model(input_tokens)

            next_token_logits = logits[0, -1, :]

            # Greedy decoding: select the most likely next character
            next_token = torch.argmax(
                next_token_logits,
                dim=-1
            ).reshape(1, 1)

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

    # Convert generated IDs back to characters
    generated_ids = tokens[0].tolist()

    return "".join(itos[token_id] for token_id in generated_ids)


# ==========================================
# 4. TEST
# ==========================================

prompt = "Machine learning"

print("\nPrompt:", repr(prompt))
print("\nGenerated text:")

try:
    result = generate_text(prompt)
    print(result)

except Exception as error:
    print("Generation failed:", error)

