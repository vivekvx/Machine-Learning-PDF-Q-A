"""Small learned checker on frozen pointer scores and question-relative context.

The contextual features are deliberately narrow and auditable. They read the
whole supplied passage and question, but never source labels or answer keys.
"""
import re

import mlx.core as mx
import mlx.nn as nn

from evidence_checker.v2.model import candidate_features
from pointer_qa.prepare import FIELDS, SYNONYMS
from working_qa.prepare import COLORS, DEVICES, enumerate_candidates

FEATURE_NAMES = ("candidate_logit", "best_other_logit", "candidate_margin",
    "candidate_softmax", "top_softmax", "score_entropy", "above_mean", "score_spread",
    "question_entity_in_sentence", "question_field_in_sentence", "joint_fact_match",
    "current_correction_for_question", "superseded_candidate_for_question")
FEATURE_WIDTH = len(FEATURE_NAMES)
CORRECTION = re.compile(r"Correction: For ([a-z]+ [a-z]+), the current (method|metric|capacity|owner) is \[ ([^\]]+) \]")
NAMES = tuple(f"{color} {device}" for color in COLORS for device in DEVICES)


def question_fact(question):
    entities = [name for name in NAMES if re.search(r"\b" + re.escape(name) + r"\b", question)]
    fields = []
    for field in FIELDS:
        for synonym in SYNONYMS[field]:
            if re.search(r"\b" + re.escape(synonym) + r"\b", question):
                fields.append(field)
                break
    if len(entities) != 1 or len(fields) != 1:
        raise ValueError("Question entity or field cannot be resolved unambiguously")
    return entities[0], fields[0]


def context_features(passage, question, candidate):
    """Read the full question/passage; no source fact metadata or labels used."""
    entity, field = question_fact(question)
    sentence = passage[candidate["sentence_start"]:candidate["sentence_end"]]
    entity_match = bool(re.search(r"\b" + re.escape(entity) + r"\b", sentence))
    field_match = bool(re.search(r"\b" + re.escape(field) + r"\b", sentence))
    correction_matches = [(m.group(1), m.group(2), m.group(3)) for m in CORRECTION.finditer(passage)]
    relevant = [value for e, f, value in correction_matches if e == entity and f == field]
    if len(relevant) > 1:
        raise ValueError("Conflicting current corrections require abstention")
    is_current = bool(relevant and sentence.lstrip().startswith("Correction:") and
                      candidate["value"] == relevant[0] and entity_match and field_match)
    superseded = bool(relevant and entity_match and field_match and
                      candidate["value"] != relevant[0])
    return [float(entity_match), float(field_match), float(entity_match and field_match),
            float(is_current), float(superseded)]


def all_features(passage, question, scores):
    candidates = enumerate_candidates(passage)
    if len(scores) != len(candidates) or len(scores) < 2:
        raise ValueError("Pointer score/source span mismatch")
    return [candidate_features(scores, i) + context_features(passage, question, candidate)
            for i, candidate in enumerate(candidates)]


class ContextChecker(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(FEATURE_WIDTH, 1)

    def __call__(self, features):
        return self.output(features)[..., 0]


def loss(model, features, label):
    z = model(features)
    y = mx.array(float(label), dtype=mx.float32)
    return mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z)))
