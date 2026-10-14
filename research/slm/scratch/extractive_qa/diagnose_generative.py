"""Read-only diagnosis of the frozen v3 generative model; train/validation only."""
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
os.environ["SLMA_QA_VERSION"] = "v3"
sys.path[:0] = [str(SCRATCH / "qa"), str(SCRATCH)]
from experiment import model_and_optimizer, rows, tokenizer  # noqa: E402
from prepare import V1, SPECIAL, answer_sequence, digest  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402

RUN = V1 / "run-001"
OUT = HERE / "v1" / "generative_diagnosis.json"


def diagnose():
    if OUT.exists():
        raise FileExistsError("Diagnosis already exists")
    config = json.loads((V1 / "config.json").read_text())
    prior = json.loads((RUN / "run.json").read_text())
    if prior["status"] != "completed":
        raise ValueError("Frozen run was not completed")
    model, optimizer = model_and_optimizer(config, fresh_seed=False)
    best = prior["worker_result"]["best_step"]
    checkpoint = RUN / f"checkpoint-step-{best:04d}"
    load_checkpoint(checkpoint, model, optimizer, expected_provenance=prior["worker_result"]["provenance"])
    model.eval()
    tok = tokenizer()
    marker_id = tok.encode("UNANSWERABLE")
    if not marker_id:
        raise ValueError("Marker encoding empty")
    summary = {}
    for split in ("train", "validation"):
        grouped = defaultdict(lambda: defaultdict(list))
        subset = rows(split)
        for row in subset:
            ids, mask, prefix_len, answer_len = answer_sequence(tok, row)
            if mask != [0] * (prefix_len - 1) + [1] * (answer_len + 1):
                raise ValueError("Answer mask changed")
            if len(ids) > config["training"]["maximum_sequence_length"]:
                raise ValueError("Sequence exceeds declared cap")
            scores = model(mx.array([ids[:-1]], dtype=mx.int32))[0]
            mx.eval(scores)
            first = mx.softmax(scores[prefix_len - 1], axis=-1)
            gold = ids[prefix_len]
            selected = int(mx.argmax(scores[prefix_len - 1]).item())
            per = nn.losses.cross_entropy(scores[None, :, :], mx.array([ids[1:]], dtype=mx.int32), reduction="none")[0]
            mx.eval(per)
            label = "answerable" if row["answerable"] else "unanswerable"
            item = grouped[label]
            item["gold_first_probability"].append(float(first[gold].item()))
            item["marker_first_probability"].append(float(first[marker_id[0]].item()))
            item["gold_first_argmax"].append(int(selected == gold))
            item["marker_first_argmax"].append(int(selected == marker_id[0]))
            item["gold_answer_token_ce"].append(float(mx.sum(per * mx.array(mask, dtype=mx.float32)).item()) / sum(mask))
            item["first_token_ce"].append(float(per[prefix_len - 1].item()))
            item["prefix_tokens"].append(prefix_len)
            item["answer_tokens_including_eos"].append(answer_len + 1)
            item["mask_nonzero_positions"].append(sum(mask))
        summary[split] = {label: {name: {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
                                  for name, values in item.items()} | {"count": len(subset) // 2}
                          for label, item in grouped.items()}
    prior_predictions = [json.loads(line) for line in (RUN / "validation_model_predictions.jsonl").read_text().splitlines()]
    output = {"schema_version": 1, "source_checkpoint_sha256": digest(checkpoint / "model.safetensors"),
              "source_config_sha256": digest(V1 / "config.json"), "source_validation_sha256": digest(V1 / "validation.jsonl"),
              "test_files_opened": False, "marker_token_ids": marker_id, "split_category_summary": summary,
              "frozen_validation_decoding": {"count": len(prior_predictions),
                  "exact_marker_predictions": sum(r["prediction"].strip() == "UNANSWERABLE" for r in prior_predictions),
                  "truncated_predictions": sum(r["truncated"] for r in prior_predictions),
                  "generated_token_mean": statistics.mean(r["generated_tokens"] for r in prior_predictions),
                  "maximum_generation_tokens": config["training"]["maximum_generation_tokens"]},
              "interpretation_limit": "Probabilities and losses are observational diagnostics; no weights or data were changed."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))
