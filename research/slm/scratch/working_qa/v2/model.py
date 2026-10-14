"""Pairwise neural evidence reader, scratch initialized; not a generative LM."""
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from extractive_qa.model import EncoderBlock, RMSNorm  # noqa: E402


class PairReader(nn.Module):
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
        self.hidden = nn.Linear(4 * width + 1, width)
        self.out = nn.Linear(width, 1)

    def score_pair(self, pair):
        ids = mx.array([pair["ids"]], dtype=mx.int32)
        if ids.shape[1] > self.context_length:
            raise ValueError("Candidate pair exceeds context length")
        raw = self.token(ids)[0]
        encoded = raw[None, :, :] + self.position(mx.arange(ids.shape[1]))[None, :, :]
        for block in self.blocks:
            encoded = block(encoded)
        hidden = self.final_norm(encoded)[0]
        q_end, s_start = pair["question_end"], pair["sentence_start"]
        a, b = pair["value_start"], pair["value_end"]
        if not 1 <= q_end < s_start <= a < b <= ids.shape[1]:
            raise ValueError("Invalid question, sentence, or value boundaries")
        question = mx.mean(hidden[:q_end], axis=0)
        sentence = mx.mean(hidden[s_start:], axis=0)
        value = mx.mean(hidden[a:b], axis=0)
        # Shared scratch token embeddings provide a learnable lexical alignment
        # signal. No string comparisons, field rules, or gold labels enter it.
        qraw, sraw = raw[:q_end], raw[s_start:]
        qraw = qraw / mx.maximum(mx.linalg.norm(qraw, axis=-1, keepdims=True), 1e-6)
        sraw = sraw / mx.maximum(mx.linalg.norm(sraw, axis=-1, keepdims=True), 1e-6)
        overlap = mx.mean(mx.max(qraw @ sraw.T, axis=-1))
        features = mx.concatenate((question, sentence, value, question * sentence,
                                   overlap.reshape(1)), axis=0)
        return self.out(mx.tanh(self.hidden(features)))[0]

    def __call__(self, pairs):
        if not pairs:
            raise ValueError("At least one candidate is required")
        return mx.stack([self.score_pair(pair) for pair in pairs])


def parameter_count(model):
    return sum(value.size for _, value in tree_flatten(model.parameters()))


def loss(model, pairs, gold_index):
    logits = model(pairs)
    labels = mx.array([float(i == gold_index) for i in range(len(pairs))])
    bce = mx.maximum(logits, 0) - logits * labels + mx.log1p(mx.exp(-mx.abs(logits)))
    if gold_index is None:
        return mx.mean(bce)
    if not 0 <= gold_index < len(pairs):
        raise ValueError("Gold index outside candidate set")
    if len(pairs) == 1:
        return bce[0]
    negative = mx.concatenate((bce[:gold_index], bce[gold_index + 1:]))
    return 0.5 * bce[gold_index] + 0.5 * mx.mean(negative)


def predict(model, pairs, candidates):
    logits = model(pairs)
    mx.eval(logits)
    values = logits.tolist()
    chosen = max(range(len(values)), key=lambda i: values[i])
    if values[chosen] <= 0:
        return {"prediction": "UNANSWERABLE", "candidate_index": None,
                "score_logit": values[chosen], "evidence": None}
    candidate = candidates[chosen]
    return {"prediction": candidate["value"], "candidate_index": chosen,
            "score_logit": values[chosen],
            "evidence": {key: candidate[key] for key in ("start", "end", "sentence_start", "sentence_end")}}
