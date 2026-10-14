"""Question/candidate-sentence pairing using only bracketed value locations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from working_qa.prepare import enumerate_candidates  # noqa: E402


def pair_inputs(tok, passage, question):
    pairs, candidates = [], enumerate_candidates(passage)
    for candidate in candidates:
        sentence = passage[candidate["sentence_start"]:candidate["sentence_end"]]
        prefix = f"Question: {question}\nPassage: "
        text = prefix + sentence
        ids = tok.encode(text)
        question_end = len(tok.encode(f"Question: {question}\n"))
        sentence_start = len(tok.encode(prefix))
        value_start = len(tok.encode(prefix + sentence[:candidate["start"] - candidate["sentence_start"]]))
        value_end = len(tok.encode(prefix + sentence[:candidate["end"] - candidate["sentence_start"]]))
        if not (0 < question_end < sentence_start <= value_start < value_end <= len(ids)):
            raise ValueError("Invalid candidate pair alignment")
        if tok.decode(ids[value_start:value_end]) != candidate["value"] or tok.decode(ids) != text:
            raise ValueError("Tokenizer candidate roundtrip failed")
        pairs.append({"ids": ids, "question_end": question_end, "sentence_start": sentence_start,
                      "value_start": value_start, "value_end": value_end})
    if not pairs:
        raise ValueError("No bracketed candidate values")
    return pairs, candidates
