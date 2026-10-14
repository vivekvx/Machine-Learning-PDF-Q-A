"""Randomly initialized, decoder-only Transformer built directly with MLX layers."""
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon=1e-5):
        super().__init__()
        self.weight = mx.ones((width,))
        self.epsilon = epsilon

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.epsilon) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def __call__(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_width)
        future = mx.triu(mx.full((length, length), -1e9, dtype=scores.dtype), k=1)
        weights = mx.softmax(scores + future, axis=-1)
        result = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(result)


class FeedForward(nn.Module):
    def __init__(self, width, multiplier):
        super().__init__()
        inner = width * multiplier
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def __call__(self, x):
        return self.down(nn.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, width, heads, multiplier, epsilon):
        super().__init__()
        self.attn_norm = RMSNorm(width, epsilon)
        self.attn = CausalAttention(width, heads)
        self.ff_norm = RMSNorm(width, epsilon)
        self.ff = FeedForward(width, multiplier)

    def __call__(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ff(self.ff_norm(x))


class ScratchTransformer(nn.Module):
    def __init__(self, model_config, vocab_size):
        super().__init__()
        width = model_config["embedding_dim"]
        context = model_config["context_length"]
        self.context_length = context
        self.vocab_size = vocab_size
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = [Block(width, model_config["attention_heads"],
                             model_config["feed_forward_multiplier"],
                             model_config["normalization_epsilon"])
                       for _ in range(model_config["layers"])]
        self.final_norm = RMSNorm(width, model_config["normalization_epsilon"])
        self.head = nn.Linear(width, vocab_size, bias=False)

    def __call__(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError("Expected [batch, sequence] token IDs")
        batch, length = token_ids.shape
        if length > self.context_length:
            raise ValueError("Sequence exceeds configured context")
        positions = mx.arange(length)
        x = self.token(token_ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


def parameter_count(model):
    return sum(value.size for _, value in tree_flatten(model.parameters()))


def next_token_loss(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")
