"""Scratch bidirectional Transformer with direct trainable token alignment."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractive_qa.model import EncoderBlock, RMSNorm  # noqa: E402


class AlignedPointer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        width = config["embedding_dim"]
        self.context_length = config["context_length"]
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.context_length, width)
        self.blocks = [EncoderBlock(width, config["attention_heads"],
                                    config["feed_forward_multiplier"],
                                    config["normalization_epsilon"])
                       for _ in range(config["layers"])]
        self.final_norm = RMSNorm(width, config["normalization_epsilon"])
        self.importance = nn.Embedding(vocab_size, 1)
        self.importance.weight = mx.zeros((vocab_size, 1))
        self.score = nn.Linear(2, 1)
        self.score.weight = mx.array([[config["initial_lexical_gain"], config["initial_contextual_gain"]]], dtype=mx.float32)
        self.score.bias = mx.zeros((1,))

    def score_pair(self, pair):
        ids = mx.array([pair["ids"]], dtype=mx.int32)
        length = ids.shape[1]
        if length > self.context_length:
            raise ValueError("Pair exceeds model context")
        q_end, s_start = pair["question_end"], pair["sentence_start"]
        a, b = pair["value_start"], pair["value_end"]
        if not 1 <= q_end < s_start <= a < b <= length:
            raise ValueError("Invalid question or candidate positions")
        raw = self.token(ids)[0]
        qraw, sraw = raw[:q_end], raw[s_start:]
        qraw = qraw / mx.maximum(mx.linalg.norm(qraw, axis=-1, keepdims=True), 1e-6)
        sraw = sraw / mx.maximum(mx.linalg.norm(sraw, axis=-1, keepdims=True), 1e-6)
        matches = mx.max(qraw @ sraw.T, axis=-1)
        weights = nn.softplus(self.importance(ids[0, :q_end])[:, 0]) + 1e-4
        lexical = mx.sum(weights * matches) / mx.sum(weights)

        hidden = raw[None, :, :] + self.position(mx.arange(length))[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)[0]
        question = mx.mean(hidden[:q_end], axis=0)
        sentence = mx.mean(hidden[s_start:], axis=0)
        question = question / mx.maximum(mx.linalg.norm(question), 1e-6)
        sentence = sentence / mx.maximum(mx.linalg.norm(sentence), 1e-6)
        contextual = mx.sum(question * sentence)
        return self.score(mx.stack((lexical, contextual)))[0]

    def __call__(self, pairs):
        if not pairs:
            raise ValueError("At least one source span required")
        return mx.stack([self.score_pair(pair) for pair in pairs])


def parameter_count(model):
    return sum(item.size for _, item in tree_flatten(model.parameters()))


def loss(model, pairs, gold_index):
    if not 0 <= gold_index < len(pairs):
        raise ValueError("Gold pointer outside source spans")
    logits = model(pairs)
    return nn.losses.cross_entropy(logits[None, :], mx.array([gold_index], dtype=mx.int32))[0]


def predict(model, pairs, candidates):
    if len(pairs) != len(candidates):
        raise ValueError("Candidate alignment mismatch")
    logits = model(pairs)
    mx.eval(logits)
    probabilities = mx.softmax(logits).tolist()
    selected = max(range(len(probabilities)), key=lambda i: probabilities[i])
    source = candidates[selected]
    return {"prediction": source["value"], "candidate_index": selected,
            "evidence": {key: source[key] for key in ("start", "end", "sentence_start", "sentence_end")},
            "confidence_uncalibrated": probabilities[selected]}
