"""Scratch bidirectional reader: one neural choice among value spans and NULL.

Candidate enumeration only identifies bracketed source values.  It does not
inspect the question or select an answer.  The trained network does both.
"""
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extractive_qa.model import EncoderBlock, RMSNorm  # noqa: E402


class CandidateReader(nn.Module):
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
        self.candidate_hidden = nn.Linear(4 * width, width)
        self.candidate_out = nn.Linear(width, 1)
        self.null_out = nn.Linear(width, 1)

    def __call__(self, ids, question_end, candidates):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > self.context_length:
            raise ValueError("Expected batch 1 within context length")
        if not 1 <= question_end < ids.shape[1] or not candidates:
            raise ValueError("Question and candidate spans required")
        positions = mx.arange(ids.shape[1])
        hidden = self.token(ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)[0]
        question = mx.mean(hidden[:question_end], axis=0)
        scores = [self.null_out(question)[0]]
        for item in candidates:
            a, b = item["start"], item["end"]
            s, e = item["sentence_start"], item["sentence_end"]
            if not 0 <= s <= a < b <= e <= ids.shape[1]:
                raise ValueError("Candidate span outside sentence or input")
            value = mx.mean(hidden[a:b], axis=0)
            sentence = mx.mean(hidden[s:e], axis=0)
            features = mx.concatenate((question, sentence, value,
                                       question * sentence), axis=0)
            scores.append(self.candidate_out(mx.tanh(self.candidate_hidden(features)))[0])
        return mx.stack(scores)


def parameter_count(model):
    return sum(value.size for _, value in tree_flatten(model.parameters()))


def loss(model, ids, question_end, candidates, target):
    scores = model(ids, question_end, candidates)
    if not 0 <= target < len(candidates) + 1:
        raise ValueError("Target outside candidate set")
    return nn.losses.cross_entropy(scores[None, :], mx.array([target], dtype=mx.int32))[0]


def predict(model, encoded, candidates):
    scores = model(mx.array([encoded["ids"]], dtype=mx.int32),
                   encoded["question_end"], encoded["candidates"])
    mx.eval(scores)
    probabilities = mx.softmax(scores).tolist()
    choice = max(range(len(probabilities)), key=lambda i: probabilities[i])
    if choice == 0:
        return {"prediction": "UNANSWERABLE", "candidate_index": None,
                "confidence_uncalibrated": probabilities[0], "evidence": None}
    candidate = candidates[choice - 1]
    return {"prediction": candidate["value"], "candidate_index": choice - 1,
            "confidence_uncalibrated": probabilities[choice],
            "evidence": {"start": candidate["start"], "end": candidate["end"],
                         "sentence_start": candidate["sentence_start"],
                         "sentence_end": candidate["sentence_end"]}}
