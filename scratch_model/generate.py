import torch
from torch import nn
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path("scratch_model/checkpoint/model.pt")

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

vocab = checkpoint["vocab"]
SEQ_LEN = checkpoint["seq_len"]

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 64)
        self.position_embedding = nn.Embedding(SEQ_LEN, 64)

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(64, vocab_size)

    def forward(self, x):
        _, length = x.shape
        positions = torch.arange(length, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.transformer(h, mask=mask)
        return self.output(h)


model = TinyTransformer(len(vocab)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def generate(prompt, max_new_tokens=100):
    unknown_chars = [ch for ch in prompt if ch not in stoi]

    if unknown_chars:
        print("Some prompt characters are not in the training vocabulary.")
        prompt = "".join(ch for ch in prompt if ch in stoi)

    if not prompt:
        prompt = "Machine learning"

    tokens = torch.tensor(
        [[stoi[ch] for ch in prompt]],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[:, -SEQ_LEN:]
            logits = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return "".join(itos[int(i)] for i in tokens[0])


if __name__ == "__main__":
    print("Device:", DEVICE)
    question = input("Enter a prompt: ")
    print("\nGenerated text:\n")
    print(generate(question))