"""Offline answerable-only neural pointer demo; no abstention claim or fallback."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402
from pointer_qa.v2.model import predict  # noqa: E402
from pointer_qa.v2.train import config, load_plan  # noqa: E402
from pointer_qa.answerability.train import selected_pointer  # noqa: E402


def answer_only(passage, question):
    """Requires exactly four marked source spans and an answerable question."""
    if not passage.strip() or not question.strip():
        raise ValueError("Passage and question are required")
    load_plan()
    before = preflight("smoke")
    if not before["ready"]:
        raise RuntimeError("Memory preflight refused: " + "; ".join(before["reasons"]))
    pointer, tok = selected_pointer()
    pairs, candidates = pair_inputs(tok, passage, question)
    settings = config()["training"]
    if len(candidates) != 4 or max(len(x["ids"]) for x in pairs) > settings["maximum_pair_tokens"]:
        raise ValueError("Demo supports exactly four bracketed values within the registered token budget")
    if max(x["value_end"] - x["value_start"] for x in pairs) > settings["maximum_answer_tokens"]:
        raise ValueError("Marked answer span exceeds registered token limit")
    reasons = guard_reasons(snapshot(), before["resources"], before["safety"])
    if reasons:
        raise RuntimeError("Memory guard refused: " + "; ".join(reasons))
    tick = time.monotonic()
    found = predict(pointer, pairs, candidates)
    elapsed = (time.monotonic() - tick) * 1000
    evidence = found["evidence"]
    if passage[evidence["start"]:evidence["end"]] != found["prediction"]:
        raise ValueError("Pointer output did not copy a source span")
    return {"model": "SLM-A scratch pointer v2 answerable-only",
            "assumption": "question has an answer in one of four bracketed source spans",
            "prediction": found["prediction"], "source_span": evidence,
            "supporting_sentence": passage[evidence["sentence_start"]:evidence["sentence_end"]],
            "confidence_uncalibrated": found["confidence_uncalibrated"],
            "latency_ms": elapsed,
            "answerability_checked": False, "rule_fallback_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(json.dumps(answer_only(args.passage, args.question), indent=2))
