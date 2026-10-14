"""Compact learned support calibrator on frozen candidate-specific pointer scores."""
import math

import mlx.core as mx
import mlx.nn as nn

from working_qa.v2.prepare import pair_inputs

FEATURE_WIDTH = 8


class SupportCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(FEATURE_WIDTH, 1)

    def __call__(self, features):
        return self.output(features)[..., 0]


def loss(model, features, label):
    z = model(features)
    y = mx.array(float(label), dtype=mx.float32)
    return mx.mean(mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z))))


def candidate_features(scores, index):
    if not 0 <= index < len(scores) or len(scores) < 2:
        raise ValueError("Candidate index invalid")
    top_other = max(scores[:index] + scores[index + 1:])
    peak = max(scores)
    exps = [math.exp(value - peak) for value in scores]
    normalizer = sum(exps)
    probs = [value / normalizer for value in exps]
    entropy = -sum(p * math.log(max(p, 1e-30)) for p in probs)
    mean = sum(scores) / len(scores)
    spread = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
    return [scores[index], top_other, scores[index] - top_other,
            probs[index], max(probs), entropy, scores[index] - mean, spread]


def scenario_features(pointer, tok, passage, question, maximum_pair_tokens):
    pairs, candidates = pair_inputs(tok, passage, question)
    if max(len(pair["ids"]) for pair in pairs) > maximum_pair_tokens:
        raise ValueError("Frozen pointer pair token cap exceeded")
    logits = pointer(pairs)
    mx.eval(logits)
    scores = [float(value) for value in logits.tolist()]
    if not all(math.isfinite(value) for value in scores):
        raise FloatingPointError("Nonfinite frozen pointer score")
    features = [candidate_features(scores, i) for i in range(len(candidates))]
    selected = max(range(len(scores)), key=lambda i: scores[i])
    return features, candidates, selected, sum(len(pair["ids"]) for pair in pairs)
