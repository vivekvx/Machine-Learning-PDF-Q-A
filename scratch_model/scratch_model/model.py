import torch
from torch import nn


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
            layer, num_layers=2
        )

        self.output = nn.Linear(64, vocab_size)

    def forward(self, x):
        _, length = x.shape
        positions = torch.arange(length, device=x.device)

        h = self.token_embedding(x) + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        h = self.transformer(h, mask=mask)
        return self.output(h)