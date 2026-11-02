import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism with lower-triangular masking.
    Ensures that prediction at step t can only depend on steps <= t.
    """

    def __init__(self, d_model: int = 192, n_head: int = 6, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Key, Query, Value projections combined into a single matrix for speed
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: 1s in lower triangle, 0s elsewhere
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, sequence_length, d_model

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble head outputs

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-Forward Network with Gaussian Error Linear Unit (GELU)."""

    def __init__(self, d_model: int = 192, dim_feedforward: int = 768, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, dim_feedforward)
        self.c_proj = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with Residual Connections."""

    def __init__(self, d_model: int = 192, n_head: int = 6, max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4 * d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomCausalLM(nn.Module):
    """
    Decoder-Only Causal Language Model built 100% from scratch in PyTorch.
    Follows modern GPT pre-layer norm transformer principles.
    Trained via autoregressive next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_head: int = 6,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.wte = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        # Token + Positional embeddings
        x = self.drop(self.wte(idx) + self.wpe(positions))

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten B, T for cross entropy calculation
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_idx,
                label_smoothing=0.05
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.5,
        top_k: int = 30,
        repetition_penalty: float = 1.6,
        stop_token_ids: list[int] = None
    ) -> list[int]:
        """
        Autoregressive next-token text generation.
        Supports temperature scaling, top-k sampling, and repetition penalty.
        """
        self.eval()
        stop_ids = set(stop_token_ids or [])

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on last token prediction

            # Apply repetition penalty on previously generated tokens
            if repetition_penalty > 1.0:
                for token_id in set(idx[0].tolist()):
                    if token_id != self.pad_idx:
                        logits[:, token_id] /= repetition_penalty

            # Apply temperature
            if temperature <= 0.05:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            next_token = idx_next.item()
            if next_token in stop_ids:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx[0].tolist()
