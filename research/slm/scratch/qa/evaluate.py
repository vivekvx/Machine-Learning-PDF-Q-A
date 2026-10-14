"""One-time held-out evaluation of the selected synthetic QA checkpoint."""
import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx

from experiment import V1, load_plan, model_and_optimizer, rows, tokenizer
from prepare import SPECIAL, digest, prompt, verify
from checkpoint import load_checkpoint
from run import guard_reasons, preflight, snapshot

FIELDS = ("timeout", "port", "mode", "owner")
ENTITY = re.compile(r"\b(?:TM|TD|TO|VB|XF)-(?:[A-F0-9]{8}|\d{4})\b")


def rule_prediction(passage, question):
    entity_match = ENTITY.search(question)
    fields = [field for field in FIELDS if re.search(r"\b" + field + r"\b", question)]
    if not entity_match or len(fields) != 1:
        return "UNANSWERABLE"
    entity, field = entity_match.group(), fields[0]
    exact_entity = re.compile(r"(?<![A-Za-z0-9-])" + re.escape(entity) + r"(?![A-Za-z0-9-])")
    evidence = next((sentence.strip() for sentence in re.split(r"(?<=\.)\s+", passage)
                     if exact_entity.search(sentence)), "")
    match = re.search(r"\b" + field + r"\s*(?::|=)?\s*(.*?)\s*(?:\band\b|\.)", evidence)
    return match.group(1).strip() if match and match.group(1).strip() else "UNANSWERABLE"


def generate(model, tok, passage, question, limit):
    prefix = tok.encode(prompt(passage, question))
    if len(prefix) + limit > model.context_length:
        raise ValueError("Prompt and output exceed model context")
    ids, generated = list(prefix), []
    tick = time.monotonic()
    for _ in range(limit):
        scores = model(mx.array([ids], dtype=mx.int32))
        mx.eval(scores)
        next_id = int(mx.argmax(scores[0, -1]).item())
        if next_id == SPECIAL["eos"]:
            break
        generated.append(next_id)
        ids.append(next_id)
    encoded = b"".join(tok.token_bytes.get(i, b"") for i in generated if i not in SPECIAL.values())
    text = encoded.decode("utf-8", errors="replace")
    return text, time.monotonic() - tick, len(generated), len(generated) >= limit


def normalized(value):
    return " ".join(re.findall(r"[\w]+", value.casefold()))


def token_f1(prediction, expected):
    p, g = normalized(prediction).split(), normalized(expected).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum(min(p.count(token), g.count(token)) for token in set(p) & set(g))
    return 2 * common / (len(p) + len(g))


def score(predictions):
    answerable = [r for r in predictions if r["answerable"]]
    unanswerable = [r for r in predictions if not r["answerable"]]
    marker_tp = sum(r["prediction"].strip() == "UNANSWERABLE" for r in unanswerable)
    marker_fp = sum(r["prediction"].strip() == "UNANSWERABLE" for r in answerable)
    return {"count": len(predictions), "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "exact_match": sum(r["exact_match"] for r in predictions) / len(predictions),
            "token_f1": sum(r["token_f1"] for r in predictions) / len(predictions),
            "answerable_exact_match": sum(r["exact_match"] for r in answerable) / len(answerable),
            "answerable_token_f1": sum(r["token_f1"] for r in answerable) / len(answerable),
            "unanswerable_exact_marker_accuracy": marker_tp / len(unanswerable),
            "abstention_precision": marker_tp / (marker_tp + marker_fp) if marker_tp + marker_fp else 0.0,
            "abstention_recall": marker_tp / len(unanswerable),
            "answerable_evidence_support": sum(r["evidence_support"] for r in answerable) / len(answerable),
            "latency_median_seconds": statistics.median(r["latency_seconds"] for r in predictions),
            "latency_mean_seconds": statistics.mean(r["latency_seconds"] for r in predictions),
            "truncated_count": sum(r["truncated"] for r in predictions)}


def assess(row, prediction, latency, generated_tokens, truncated):
    # Exact marker is intentionally strict; normalized EM applies only to answers.
    exact = (normalized(prediction) == normalized(row["expected_answer"]) if row["answerable"]
             else prediction.strip() == "UNANSWERABLE")
    support = bool(row["answerable"] and normalized(prediction) == normalized(row["expected_answer"]) and
                   normalized(row["expected_answer"]) in normalized(row["target_sentence"]))
    return {"id": row["id"], "family": row["family"], "fact_id": row["fact_id"],
            "answerable": row["answerable"], "expected_answer": row["expected_answer"],
            "prediction": prediction, "evidence_id": f'{row["id"]}:target_sentence' if row["answerable"] else None,
            "exact_match": exact, "token_f1": token_f1(prediction, row["expected_answer"]),
            "evidence_support": support, "latency_seconds": latency,
            "generated_tokens": generated_tokens, "truncated": truncated}


def evaluate(run_dir):
    run_dir = Path(run_dir).resolve()
    if (run_dir / "evaluation.json").exists():
        raise FileExistsError("Held-out evaluation already exists")
    run = json.loads((run_dir / "run.json").read_text())
    if run["status"] != "completed" or not run.get("worker_result"):
        raise ValueError("No selected completed checkpoint")
    verify()
    load_plan()
    config = json.loads((V1 / "config.json").read_text())
    before = preflight("smoke")
    if not before["ready"]:
        raise RuntimeError(f"Evaluation preflight refused: {before['reasons']}")
    tok = tokenizer()
    model, optimizer = model_and_optimizer(config, fresh_seed=False)
    best_step = run["worker_result"]["best_step"]
    provenance = run["worker_result"]["provenance"]
    checkpoint = run_dir / f"checkpoint-step-{best_step:04d}"
    load_checkpoint(checkpoint, model, optimizer, expected_provenance=provenance)
    model.eval()
    mx.metal.reset_peak_memory()
    results = {}
    for split in ("validation", "test"):
        # Only this post-selection code path opens the sealed test file.
        source = V1 / ("test.sealed.jsonl" if split == "test" else "validation.jsonl")
        subset = [json.loads(line) for line in source.read_text().splitlines()]
        predicted, rule = [], []
        for row in subset:
            reasons = guard_reasons(snapshot(), before["resources"], before["safety"])
            if reasons:
                raise RuntimeError(f"Evaluation stopped by resource guard: {reasons}")
            generated, latency, tokens, truncated = generate(model, tok, row["passage"], row["question"],
                                                             config["training"]["maximum_generation_tokens"])
            predicted.append(assess(row, generated, latency, tokens, truncated))
            tick = time.monotonic()
            rule_text = rule_prediction(row["passage"], row["question"])
            rule.append(assess(row, rule_text, time.monotonic() - tick, 0, False))
        for condition, data in (("model", predicted), ("rule", rule)):
            destination = run_dir / f"{split}_{condition}_predictions.jsonl"
            with destination.open("x") as stream:
                for record in data:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        results[split] = {"model": score(predicted), "rule": score(rule)}
    gate = config["heldout_pass_gate"]
    metric = results["test"]["model"]
    passed = (metric["count"] >= gate["minimum_test_examples"] and
              metric["answerable_exact_match"] >= gate["minimum_answerable_exact_match"] and
              metric["unanswerable_exact_marker_accuracy"] >= gate["minimum_unanswerable_exact_marker_accuracy"] and
              metric["answerable_evidence_support"] >= gate["minimum_answerable_evidence_support"])
    output = {"schema_version": 1, "selected_step": best_step, "selected_checkpoint_sha256": digest(checkpoint / "model.safetensors"),
              "config_sha256": digest(V1 / "config.json"), "plan_sha256": digest(V1 / "plan.json"),
              "tokenizer_sha256": digest(V1 / "tokenizer.json"), "results": results,
              "heldout_pass_gate": gate, "heldout_passed": passed,
              "peak_mlx_bytes": mx.metal.get_peak_memory(),
              "prediction_files_sha256": {p.name: digest(p) for p in run_dir.glob("*_predictions.jsonl")}}
    (run_dir / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir), indent=2))
