"""Fresh bidirectional Transformer encoder with neural span and answerability heads.

This variant is extractive, not a decoder-only generative language model.
"""
import math
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_spec = importlib.util.spec_from_file_location("slma_base_architecture", Path(__file__).resolve().parent.parent / "model.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
CausalAttention, FeedForward, RMSNorm = _base.CausalAttention, _base.FeedForward, _base.RMSNorm


class FullAttention(CausalAttention):
    """Reuse scratch QKV/projection layers with no future-token mask."""
    def __call__(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        weights = mx.softmax((q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_width), axis=-1)
        result = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.proj(result)


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, multiplier, epsilon):
        super().__init__()
        self.attn_norm = RMSNorm(width, epsilon)
        self.attn = FullAttention(width, heads)
        self.ff_norm = RMSNorm(width, epsilon)
        self.ff = FeedForward(width, multiplier)

    def __call__(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ff(self.ff_norm(x))


class ExtractiveTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        width = config["embedding_dim"]
        self.context_length = config["context_length"]
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.context_length, width)
        self.blocks = [EncoderBlock(width, config["attention_heads"], config["feed_forward_multiplier"],
                                    config["normalization_epsilon"]) for _ in range(config["layers"])]
        self.final_norm = RMSNorm(width, config["normalization_epsilon"])
        self.start_head = nn.Linear(width, 1)
        self.end_head = nn.Linear(width, 1)
        self.answerability_head = nn.Linear(width, 1)

    def __call__(self, ids, passage_start, passage_end):
        if ids.ndim != 2 or ids.shape[1] > self.context_length:
            raise ValueError("Expected [batch, sequence] within context length")
        positions = mx.arange(ids.shape[1])
        x = self.token(ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # The first token is a global classifier because *all* positions can
        # attend to every passage and question token in FullAttention.
        answerability = self.answerability_head(x[:, 0, :])[:, 0]
        allowed = (positions >= passage_start) & (positions < passage_end)
        mask = mx.where(allowed, mx.zeros_like(positions, dtype=mx.float32),
                        mx.full(positions.shape, -1e9, dtype=mx.float32))
        start = self.start_head(x)[:, :, 0] + mask[None, :]
        end = self.end_head(x)[:, :, 0] + mask[None, :]
        return start, end, answerability


def parameter_count(model):
    return sum(value.size for _, value in tree_flatten(model.parameters()))


def loss(model, ids, passage_start, passage_end, answerable, start_target, end_target,
         answerability_weight=1.0, span_weight=1.0):
    start, end, answerability_logit = model(ids, passage_start, passage_end)
    label = mx.array([float(answerable)], dtype=mx.float32)
    # Stable logistic BCE with logit input.
    z = answerability_logit
    answer_loss = mx.mean(mx.maximum(z, 0) - z * label + mx.log1p(mx.exp(-mx.abs(z))))
    if answerable:
        start_loss = mx.mean(nn.losses.cross_entropy(start, mx.array([start_target], dtype=mx.int32)))
        end_loss = mx.mean(nn.losses.cross_entropy(end, mx.array([end_target], dtype=mx.int32)))
        span_loss = 0.5 * (start_loss + end_loss)
    else:
        span_loss = mx.array(0.0)
    return answerability_weight * answer_loss + span_weight * span_loss


def predict(model, tok, row, encoded, *, maximum_span_tokens=8):
    ids = mx.array([encoded["ids"]], dtype=mx.int32)
    start, end, answerability = model(ids, encoded["passage_start"], encoded["passage_end"])
    mx.eval(start, end, answerability)
    probability = float(mx.sigmoid(answerability)[0].item())
    if probability < 0.5:
        return {"prediction": "UNANSWERABLE", "answerability_probability": probability,
                "predicted_start": None, "predicted_end": None}
    s = start[0].tolist()
    e = end[0].tolist()
    left, right = encoded["passage_start"], encoded["passage_end"]
    best = max(((s[i] + e[j], i, j) for i in range(left, right)
                for j in range(i, min(right, i + maximum_span_tokens))), key=lambda item: item[0])
    answer = tok.decode(encoded["ids"][best[1]:best[2] + 1]).strip()
    return {"prediction": answer, "answerability_probability": probability,
            "predicted_start": best[1], "predicted_end": best[2]}
