"""Model-only full validation of the selected answerable-only pointer."""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1"
V2 = HERE / "v2"
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from checkpoint import load_checkpoint  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402
from pointer_qa.prepare import sha, verify  # noqa: E402
from pointer_qa.v2.model import AlignedPointer, predict  # noqa: E402
from pointer_qa.v2.train import config, load_plan, provenance, rows, tokenizer  # noqa: E402


def token_f1(prediction, expected):
    a, b = re.findall(r"\w+", prediction.lower()), re.findall(r"\w+", expected.lower())
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b)) if a and b else float(a == b)


def summarize(records):
    groups = defaultdict(list)
    by_doc = defaultdict(list)
    for item in records:
        groups["all"].append(item)
        groups["paraphrase" if item["question_paraphrase"] else "direct"].append(item)
        groups[item["asked_field"]].append(item)
        groups["heldout_sentence_style" if item["gold_sentence_style"] == 3 else "seen_sentence_style"].append(item)
        by_doc[item["document_id"]].append(item["candidate_index"])
    return {"count": len(records),
            "answerable_exact_match": sum(x["exact"] for x in records) / len(records),
            "token_f1": sum(x["token_f1"] for x in records) / len(records),
            "span_copy_support": sum(x["span_copied"] for x in records) / len(records),
            "changed_question_document_fraction": sum(len(set(indexes)) > 1 for indexes in by_doc.values()) / len(by_doc),
            "groups": {name: {"count": len(items), "exact": sum(x["exact"] for x in items) / len(items)}
                       for name, items in groups.items()},
            "mean_latency_ms": sum(x["latency_ms"] for x in records) / len(records),
            "median_latency_ms": sorted(x["latency_ms"] for x in records)[len(records) // 2]}


def evaluate():
    verify()
    p, c = load_plan(), config()
    output = V2 / "validation"
    if output.exists():
        raise FileExistsError("Full validation is immutable and already exists")
    run = json.loads((V2 / "runs" / "main" / "run.json").read_text())
    if run["status"] != "completed" or run["worker_result"]["status"] != "main_completed":
        raise RuntimeError("Guarded main run did not complete")
    best_step = run["worker_result"]["best_step"]
    checkpoint = V2 / "runs" / "main" / f"checkpoint-step-{best_step:04d}"
    pre = preflight("smoke")
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    if not pre["ready"]:
        result = {"status": "preflight_refused", "reasons": pre["reasons"]}
        (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    tok = tokenizer()
    model = AlignedPointer(c["model"], tok.vocab_size)
    opt = optim.AdamW(learning_rate=c["training"]["learning_rate"],
                      weight_decay=c["training"]["weight_decay"])
    load_checkpoint(checkpoint, model, opt, expected_provenance=provenance())
    model.eval()
    mx.metal.reset_peak_memory()
    examples = rows("validation")
    records = []
    started = time.monotonic()
    with (output / "predictions.jsonl").open("x") as stream:
        for index, row in enumerate(examples):
            if index % 16 == 0:
                observed = snapshot(os.getpid())
                reasons = guard_reasons(observed, pre["resources"], pre["safety"])
                if reasons:
                    status = {"status": "stopped_by_guard", "at_index": index, "reasons": reasons}
                    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                    return status
            pairs, candidates = pair_inputs(tok, row["passage"], row["question"])
            tick = time.monotonic()
            found = predict(model, pairs, candidates)
            latency = 1000 * (time.monotonic() - tick)
            span = found["evidence"]
            record = {"id": row["id"], "document_id": row["document_id"],
                      "question_paraphrase": row["question_paraphrase"],
                      "asked_field": row["asked_field"],
                      "gold_sentence_style": row["facts"][row["gold_candidate_index"]]["sentence_style"],
                      "expected_answer": row["expected_answer"],
                      "prediction": found["prediction"],
                      "candidate_index": found["candidate_index"],
                      "evidence": span,
                      "confidence_uncalibrated": found["confidence_uncalibrated"],
                      "exact": (found["candidate_index"] == row["gold_candidate_index"] and
                                found["prediction"] == row["expected_answer"]),
                      "token_f1": token_f1(found["prediction"], row["expected_answer"]),
                      "span_copied": row["passage"][span["start"]:span["end"]] == found["prediction"],
                      "latency_ms": latency}
            records.append(record)
            stream.write(json.dumps(record) + "\n")
    summary = summarize(records)
    gate = c["validation_gate"]
    passed = (summary["count"] >= gate["minimum_examples"] and
              summary["answerable_exact_match"] >= gate["answerable_exact_match"] and
              summary["groups"]["paraphrase"]["exact"] >= gate["paraphrase_distractor_exact_match"] and
              summary["span_copy_support"] == gate["span_copy_support"] and
              summary["changed_question_document_fraction"] >= gate["question_sensitivity_changed_document_fraction"])
    summary.update({"status": "completed", "gate_passed": passed,
                    "checkpoint_step": best_step,
                    "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
                    "config_sha256": sha(V2 / "config.json"),
                    "plan_sha256": sha(V2 / "plan.json"),
                    "manifest_sha256": sha(V1 / "manifest.json"),
                    "tokenizer_sha256": sha(V1 / "tokenizer.json"),
                    "predictions_sha256": sha(output / "predictions.jsonl"),
                    "peak_mlx_bytes": mx.metal.get_peak_memory(),
                    "wall_seconds": time.monotonic() - started,
                    "sealed_test_opened": False})
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if passed:
        selection = {"pointer_gate_passed": True, "checkpoint_step": best_step,
                     "checkpoint_sha256": summary["checkpoint_sha256"],
                     "validation_predictions_sha256": summary["predictions_sha256"],
                     "dataset_manifest_sha256": summary["manifest_sha256"],
                     "tokenizer_sha256": summary["tokenizer_sha256"],
                     "inference_protocol": "four bracketed source-value spans; neural argmax without NULL; exact character-span copy"}
        (output / "pointer_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(), indent=2))
