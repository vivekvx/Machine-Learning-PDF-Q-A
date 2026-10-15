"""One guarded offline MLX selection request from JSON stdin; no network/model API."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

SCRATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRATCH))
from run import guard_reasons, preflight, snapshot  # noqa: E402
from pointer_qa.answerability.train import selected_pointer  # noqa: E402
from pointer_qa.v2.train import config, tokenizer  # noqa: E402
from pointer_qa.v2.model import predict  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def marked(item):
    sentence = item["sentence"]
    a = item["value_start"] - item["sentence_start"]
    b = item["value_end"] - item["sentence_start"]
    if not 0 <= a < b <= len(sentence) or sentence[a:b] != item["value"]:
        raise ValueError("Candidate source span mismatch")
    return sentence[:a] + "[ " + item["value"] + " ]" + sentence[b:]


def choose_four(tok, question, ranked):
    accepted, marked_sentences, rejected = [], [], []
    seen = set()
    settings = config()["training"]
    for item in ranked:
        if item["value"].casefold() in seen:
            continue
        try:
            sentence = marked(item)
            pairs, candidates = pair_inputs(tok, sentence, question)
            if len(candidates) != 1 or candidates[0]["value"] != item["value"]:
                raise ValueError("frozen parser could not isolate span")
            if len(pairs[0]["ids"]) > settings["maximum_pair_tokens"] or \
               pairs[0]["value_end"] - pairs[0]["value_start"] > settings["maximum_answer_tokens"]:
                raise ValueError("frozen tokenizer budget exceeded")
        except ValueError as exc:
            rejected.append({"value": item["value"], "reason": str(exc)})
            continue
        accepted.append(item)
        marked_sentences.append(sentence)
        seen.add(item["value"].casefold())
        if len(accepted) == 4:
            break
    if len(accepted) < 4:
        return None, None, None, accepted, rejected
    passage = " ".join(marked_sentences)
    pairs, parsed = pair_inputs(tok, passage, question)
    if len(pairs) != 4 or [x["value"] for x in parsed] != [x["value"] for x in accepted]:
        raise ValueError("Four-candidate frozen input contract failed")
    return passage, pairs, parsed, accepted, rejected


def run(payload):
    question = payload.get("question", "").strip()
    ranked = payload.get("candidates", [])
    if not question or not isinstance(ranked, list):
        raise ValueError("Question and ranked candidates required")
    pointer_path = SCRATCH / "pointer_qa" / "v2" / "runs" / "main" / "checkpoint-step-2000" / "model.safetensors"
    token_path = SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"
    selection = json.loads((SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json").read_text())
    if sha(pointer_path) != selection["checkpoint_sha256"] or sha(token_path) != selection["tokenizer_sha256"]:
        raise ValueError("Frozen pointer or tokenizer hash mismatch")
    before = preflight("smoke")
    if not before["ready"]:
        raise RuntimeError("Memory preflight refused: " + "; ".join(before["reasons"]))
    tok = tokenizer()
    passage, pairs, parsed, four, rejected = choose_four(tok, question, ranked)
    if len(four) < 4:
        return {"status": "INSUFFICIENT_CANDIDATES", "reason": "fewer_than_four_representable_source_spans",
                "valid_candidates": len(four), "rejected": rejected, "model_loaded": False}
    guard = guard_reasons(snapshot(os.getpid()), before["resources"], before["safety"])
    if guard:
        raise RuntimeError("Resource guard refused before inference: " + "; ".join(guard))
    mx.reset_peak_memory()
    pointer, loaded_tok = selected_pointer()
    if loaded_tok.vocab_size != tok.vocab_size:
        raise ValueError("Loaded pointer tokenizer differs")
    started = time.monotonic()
    found = predict(pointer, pairs, parsed)
    if found["prediction"] != four[found["candidate_index"]]["value"]:
        raise ValueError("Frozen pointer output does not match selected source candidate")
    elapsed = (time.monotonic() - started) * 1000
    after = snapshot(os.getpid())
    guard = guard_reasons(after, before["resources"], before["safety"])
    if guard:
        raise RuntimeError("Resource guard refused after inference: " + "; ".join(guard))
    return {"status": "SELECTED", "model": "SLM-A scratch pointer v2", "prediction": found["prediction"],
            "candidate_index": found["candidate_index"], "neural_confidence_uncalibrated": found["confidence_uncalibrated"],
            "four_candidates": four, "marked_passage": passage, "rejected": rejected,
            "pointer_latency_ms": elapsed, "peak_mlx_bytes": mx.get_peak_memory(),
            "resource": {"available_bytes": after["available_bytes"],
                         "memory_pressure_free_percent": after["memory_pressure_free_percent"],
                         "swap_used_growth_bytes": after["swap_used_bytes"] - before["resources"]["swap_used_bytes"],
                         "swap_out_growth_bytes": after["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"],
                         "process_rss_bytes": after["process_rss_bytes"], "disk_free_bytes": after["disk_free_bytes"]},
            "pointer_checkpoint_sha256": selection["checkpoint_sha256"],
            "pointer_tokenizer_sha256": selection["tokenizer_sha256"]}


if __name__ == "__main__":
    try:
        print(json.dumps(run(json.load(sys.stdin)), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stdout)
        raise SystemExit(1)
