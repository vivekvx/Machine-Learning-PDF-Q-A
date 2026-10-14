"""Validation-gated, one-time held-out evaluation; neural and rule outputs separate."""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from extractive_qa.model import ExtractiveTransformer, predict  # noqa: E402
from extractive_qa.prepare import V1, sha, tokenized, verify  # noqa: E402
from extractive_qa.train import config, load_plan, tokenizer  # noqa: E402

ENTITY = re.compile(r"\b(?:SR|DS|OL|MB|IC)-[A-F0-9]{8}\b")
FIELDS = ("method", "quantity", "port", "owner")


def rule(passage, question):
    entity_match = ENTITY.search(question)
    fields = [field for field in FIELDS if re.search(r"\b" + field + r"\b", question)]
    if not entity_match or len(fields) != 1:
        return {"prediction": "UNANSWERABLE", "char_start": None, "char_end": None}
    entity = entity_match.group()
    exact = re.compile(r"(?<![A-Za-z0-9-])" + re.escape(entity) + r"(?![A-Za-z0-9-])")
    for line_match in re.finditer(r"[^\n]+", passage):
        line = line_match.group()
        if not exact.search(line):
            continue
        field_match = re.search(r"\b" + fields[0] + r"\s*=\s*\[\s*(.*?)\s*\]", line)
        if not field_match:
            break
        value = field_match.group(1).strip()
        left = line_match.start() + field_match.start(1)
        return {"prediction": value, "char_start": left, "char_end": left + len(value)}
    return {"prediction": "UNANSWERABLE", "char_start": None, "char_end": None}


def normalized(text):
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def f1(prediction, expected):
    p, g = normalized(prediction).split(), normalized(expected).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum(min(p.count(token), g.count(token)) for token in set(p) & set(g))
    return 2 * common / (len(p) + len(g))


def assess(row, neural, rule_result, latency, encoded):
    prediction = neural["prediction"]
    rule_text = rule_result["prediction"]
    neural_exact = (normalized(prediction) == normalized(row["expected_answer"]) if row["answerable"]
                    else prediction == "UNANSWERABLE")
    rule_exact = (normalized(rule_text) == normalized(row["expected_answer"]) if row["answerable"]
                  else rule_text == "UNANSWERABLE")
    neural_support = bool(row["answerable"] and neural["predicted_start"] == encoded["answer_start_token"]
                          and neural["predicted_end"] == encoded["answer_end_token"])
    rule_support = bool(row["answerable"] and rule_result["char_start"] == row["answer_char_start"]
                        and rule_result["char_end"] == row["answer_char_end"])
    common = {"id": row["id"], "fact_id": row["fact_id"], "family": row["family"],
              "answerable": row["answerable"], "expected_answer": row["expected_answer"],
              "gold_span_tokens": [encoded["answer_start_token"], encoded["answer_end_token"]],
              "gold_span_chars": [row["answer_char_start"], row["answer_char_end"]]}
    return ({**common, **neural, "exact_match": neural_exact, "token_f1": f1(prediction, row["expected_answer"]),
             "span_support": neural_support, "latency_seconds": latency},
            {**common, **rule_result, "exact_match": rule_exact, "token_f1": f1(rule_text, row["expected_answer"]),
             "span_support": rule_support})


def summarize(records):
    positives = [r for r in records if r["answerable"]]
    negatives = [r for r in records if not r["answerable"]]
    tp = sum(r["prediction"] != "UNANSWERABLE" for r in positives)
    fn = len(positives) - tp
    fp = sum(r["prediction"] != "UNANSWERABLE" for r in negatives)
    tn = len(negatives) - fp
    result = {"count": len(records), "answerable_count": len(positives), "unanswerable_count": len(negatives),
              "exact_match": statistics.mean(r["exact_match"] for r in records),
              "token_f1": statistics.mean(r["token_f1"] for r in records),
              "answerable_exact_match": statistics.mean(r["exact_match"] for r in positives),
              "answerable_token_f1": statistics.mean(r["token_f1"] for r in positives),
              "answerable_span_support": statistics.mean(r["span_support"] for r in positives),
              "unanswerable_accuracy": statistics.mean(r["exact_match"] for r in negatives),
              "answerability_confusion_matrix": {"true_answerable_predicted_answerable": tp,
                                                 "true_answerable_predicted_unanswerable": fn,
                                                 "true_unanswerable_predicted_answerable": fp,
                                                 "true_unanswerable_predicted_unanswerable": tn}}
    latencies = [r["latency_seconds"] for r in records if "latency_seconds" in r]
    if latencies:
        result["median_latency_seconds"] = statistics.median(latencies)
        result["mean_latency_seconds"] = statistics.mean(latencies)
    return result


def evaluate(run_dir):
    run_dir = Path(run_dir).resolve()
    if (run_dir / "evaluation.json").exists():
        raise FileExistsError("Held-out evaluation already exists")
    run = json.loads((run_dir / "run.json").read_text())
    if run["status"] != "completed" or not run.get("worker_result"):
        raise ValueError("No completed selected checkpoint")
    verify()
    load_plan()
    c = config()
    before = preflight("smoke")
    if not before["ready"]:
        raise RuntimeError(f"Evaluation preflight refused: {before['reasons']}")
    tok = tokenizer()
    model = ExtractiveTransformer(c["model"], tok.vocab_size)
    optimizer = optim.AdamW(learning_rate=c["training"]["learning_rate"], weight_decay=c["training"]["weight_decay"])
    best_step = run["worker_result"]["best_step"]
    checkpoint = run_dir / f"checkpoint-step-{best_step:04d}"
    load_checkpoint(checkpoint, model, optimizer, expected_provenance=run["worker_result"]["provenance"])
    model.eval()
    mx.metal.reset_peak_memory()
    conditions = {}
    for split in ("validation", "test"):
        # The sealed test file is opened only after the validation gate is saved.
        source = V1 / ("test.sealed.jsonl" if split == "test" else "validation.jsonl")
        neural_records, rule_records = [], []
        for line in source.read_text().splitlines():
            row = json.loads(line)
            reasons = guard_reasons(snapshot(), before["resources"], before["safety"])
            if reasons:
                raise RuntimeError(f"Evaluation resource guard: {reasons}")
            encoded = tokenized(tok, row)
            tick = time.monotonic()
            neural = predict(model, tok, row, encoded)
            duration = time.monotonic() - tick
            deterministic = rule(row["passage"], row["question"])
            n, r = assess(row, neural, deterministic, duration, encoded)
            neural_records.append(n)
            rule_records.append(r)
        for label, records in (("neural", neural_records), ("rule", rule_records)):
            with (run_dir / f"{split}_{label}_predictions.jsonl").open("x") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        conditions[split] = {"neural": summarize(neural_records), "rule": summarize(rule_records)}
        if split == "validation":
            gate = c["validation_gate_for_demo"]
            m = conditions["validation"]["neural"]
            passed = (m["count"] >= gate["minimum_examples"] and
                      m["answerable_exact_match"] >= gate["minimum_answerable_exact_match"] and
                      m["unanswerable_accuracy"] >= gate["minimum_unanswerable_accuracy"] and
                      m["answerable_span_support"] >= gate["minimum_answerable_span_support"])
            (run_dir / "validation_gate.json").write_text(json.dumps({"gate": gate, "metrics": conditions["validation"],
                                                                         "passed": passed, "checkpoint_step": best_step,
                                                                         "test_file_opened": False}, indent=2) + "\n")
    output = {"schema_version": 1, "checkpoint_step": best_step, "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
              "config_sha256": sha(V1 / "config.json"), "plan_sha256": sha(V1 / "plan.json"),
              "tokenizer_sha256": sha(V1 / "tokenizer.json"), "validation_gate_passed": passed,
              "results": conditions, "peak_mlx_bytes": mx.metal.get_peak_memory(),
              "prediction_file_sha256": {p.name: sha(p) for p in run_dir.glob("*_predictions.jsonl")}}
    (run_dir / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir), indent=2))
