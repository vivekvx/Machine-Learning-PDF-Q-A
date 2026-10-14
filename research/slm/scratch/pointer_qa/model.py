"""Answerable-only neural pointer over bounded source-value spans.

The encoder and scorer are randomly initialized MLX Transformer components.
Candidate enumeration supplies valid contiguous source spans, never a label.
There is deliberately no NULL or answerability path here.
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from working_qa.v2.model import PairReader, parameter_count  # noqa: E402


class PointerReader(PairReader):
    """Question/candidate-sentence encoder with a learned score per span."""


def pointer_loss(model, pairs, gold_index):
    if not 0 <= gold_index < len(pairs):
        raise ValueError("Gold pointer outside valid passage spans")
    logits = model(pairs)
    return nn.losses.cross_entropy(logits[None, :], mx.array([gold_index], dtype=mx.int32))[0]


def pointer_predict(model, pairs, candidates):
    if len(pairs) != len(candidates):
        raise ValueError("Candidate input and source spans differ")
    scores = model(pairs)
    mx.eval(scores)
    probabilities = mx.softmax(scores).tolist()
    chosen = max(range(len(probabilities)), key=lambda i: probabilities[i])
    source = candidates[chosen]
    return {"prediction": source["value"], "candidate_index": chosen,
            "evidence": {key: source[key] for key in ("start", "end", "sentence_start", "sentence_end")},
            "confidence_uncalibrated": probabilities[chosen]}
