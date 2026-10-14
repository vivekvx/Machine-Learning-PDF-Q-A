"""Trainable sufficiency head over features from the frozen scratch pointer.

Candidate enumeration and offsets locate source spans. They never decide the
support label. The four-layer pointer encoder is frozen during this phase.
"""
import mlx.core as mx
import mlx.nn as nn

from working_qa.prepare import enumerate_candidates
from working_qa.v2.prepare import pair_inputs


FEATURE_WIDTH = 128 * 7 + 4


class SufficiencyHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.first = nn.Linear(FEATURE_WIDTH, hidden_dim)
        self.second = nn.Linear(hidden_dim, 1)

    def __call__(self, features):
        return self.second(nn.tanh(self.first(features)))[..., 0]


def binary_loss(model, features, label):
    z = model(features)
    target = mx.array(float(label), dtype=mx.float32)
    return mx.mean(mx.maximum(z, 0) - z * target + mx.log1p(mx.exp(-mx.abs(z))))


def _mean(hidden, start, end):
    if not 0 <= start < end <= hidden.shape[0]:
        raise ValueError("Invalid full-context token span")
    return mx.mean(hidden[start:end], axis=0)


def frozen_features(pointer, tok, passage, question, candidate_index, max_tokens):
    """Expose question, full passage, selected sentence/value, and pointer rank.

    Full-context encoding lets the candidate see correction sentences even
    when they occur after its own sentence. No gold answer/label is an input.
    """
    candidates = enumerate_candidates(passage)
    if not 0 <= candidate_index < len(candidates):
        raise ValueError("Candidate outside source passage")
    pairs, paired = pair_inputs(tok, passage, question)
    if paired != candidates:
        raise ValueError("Candidate ordering changed")
    prefix = f"Question: {question}\nPassage: "
    text = prefix + passage
    ids = tok.encode(text)
    qend = len(tok.encode(f"Question: {question}\n"))
    pstart = len(tok.encode(prefix))
    candidate = candidates[candidate_index]

    def offset(char):
        return len(tok.encode(prefix + passage[:char]))

    sstart, send = offset(candidate["sentence_start"]), offset(candidate["sentence_end"])
    vstart, vend = offset(candidate["start"]), offset(candidate["end"])
    if len(ids) > max_tokens or len(ids) > pointer.context_length:
        raise ValueError("Full context exceeds registered token cap")
    if not 0 < qend < pstart <= sstart <= vstart < vend <= send <= len(ids):
        raise ValueError("Invalid question/passage/value alignment")
    if tok.decode(ids) != text or tok.decode(ids[vstart:vend]) != candidate["value"]:
        raise ValueError("Tokenizer full-context span roundtrip failed")

    token_ids = mx.array([ids], dtype=mx.int32)
    hidden = pointer.token(token_ids) + pointer.position(mx.arange(len(ids)))[None, :, :]
    for block in pointer.blocks:
        hidden = block(hidden)
    hidden = pointer.final_norm(hidden)[0]
    q = _mean(hidden, 0, qend)
    p = _mean(hidden, pstart, len(ids))
    s = _mean(hidden, sstart, send)
    v = _mean(hidden, vstart, vend)
    scores = pointer(pairs)
    probs = mx.softmax(scores)
    selected = scores[candidate_index]
    others = mx.max(mx.concatenate((scores[:candidate_index], scores[candidate_index + 1:])))
    scalars = mx.stack((selected, selected - others, probs[candidate_index],
                        mx.max(probs)))
    features = mx.concatenate((q, p, s, v, q * s, q * v, p * v, scalars))
    if features.shape != (FEATURE_WIDTH,):
        raise ValueError("Frozen feature width changed")
    mx.eval(features)
    return features.tolist(), {"full_input_tokens": len(ids),
                               "pair_input_tokens": sum(len(pair["ids"]) for pair in pairs)}
