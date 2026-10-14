"""Opt-in untuned MLX Qwen baseline over the same 80 validation IDs per condition."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
SLM = HERE.parent
sys.path.insert(0, str(SLM / "portable"))
from scoring import aggregate, audit_prediction, manual_audit_rows
from response_format import parse_response

OUT = Path(os.environ.get("SLM_RETRIEVAL_BASELINE_OUT", str(HERE / "v1")))
MODEL = SLM / ".cache/qwen3-0.6b-4bit"
EXPERIMENT = SLM / "portable/v1/experiment.json"
MLX_CONFIG = SLM / "portable/v1/mlx.json"
GOLD = SLM / "portable/data/v1/validation_gold.jsonl"
CONDITIONS = {
    "gold": GOLD,
    "tfidf_faiss": OUT / "validation_tfidf_faiss.jsonl",
    "minilm_faiss": OUT / "validation_minilm_faiss.jsonl",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def preflight(condition):
    errors = []
    path = CONDITIONS[condition]
    config = json.loads(MLX_CONFIG.read_text())
    experiment = json.loads(EXPERIMENT.read_text())
    if not path.exists():
        return {"ready": False, "errors": [f"Missing condition input: {path}"]}
    fixed = list(read_rows(GOLD))
    samples = list(read_rows(path))
    if len(fixed) != 80 or len(samples) != 80 or [x["question_id"] for x in samples] != [x["question_id"] for x in fixed]:
        errors.append("Condition differs from fixed 80 validation IDs or order")
    if any(x["split"] != "validation" for x in samples):
        errors.append("Non-validation row in condition")
    if any(x["question"] != y["question"] or x["document_id"] != y["document_id"] or
           x["answerability"] != y["answerability"] for x, y in zip(samples, fixed)):
        errors.append("Question/document/label drift from fixed validation source")
    if condition != "gold":
        manifest_path = OUT / f"validation_{condition}.manifest.json"
        if not manifest_path.exists():
            errors.append("Missing retrieval manifest")
        else:
            meta = json.loads(manifest_path.read_text())
            if meta["source_sha256"] != digest(GOLD) or meta["output_sha256"] != digest(path):
                errors.append("Retrieval input/manifest hash mismatch")
            if meta["config_sha256"] != digest(OUT / "config.json"):
                errors.append("Retrieval versioned config changed")
    elif digest(GOLD) != json.loads((OUT / "config.json").read_text())["validation_source_sha256"]:
        errors.append("Gold source checksum mismatch")
    weights = MODEL / "model.safetensors"
    tokenizer_config = MODEL / "tokenizer_config.json"
    if not weights.exists():
        errors.append("Pinned local MLX model weights missing; no automatic download")
    elif digest(weights) != config["model_file_sha256"]:
        errors.append("Pinned MLX model weight hash mismatch")
    if not tokenizer_config.exists():
        errors.append("Pinned local tokenizer configuration missing")
    elif digest(tokenizer_config) != config["tokenizer_config_sha256"]:
        errors.append("Pinned tokenizer config hash mismatch")
    if experiment["maximum_generation_tokens"] != config["maximum_generation_tokens"]:
        errors.append("Generation-token config mismatch")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        errors.append("MLX baseline requires Apple Silicon")
    try:
        from importlib.metadata import version
        if version("mlx-lm") != config["mlx_lm"] or version("mlx") != config["mlx"]:
            errors.append("MLX dependency pin mismatch")
    except Exception as exc:
        errors.append(f"MLX dependency check failed: {exc}")
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(HERE)
    if memory.available < 1_200_000_000 or swap.free < 750_000_000:
        errors.append("Memory preflight below 1.2 GB available or 0.75 GB free swap")
    if disk.free < 1_000_000_000:
        errors.append("Less than 1 GB free disk")
    return {"ready": not errors, "errors": errors, "condition": condition,
            "validation_input_sha256": digest(path), "fixed_gold_sha256": digest(GOLD),
            "retrieval_baseline_config_sha256": digest(OUT / "config.json"),
            "experiment_config_sha256": digest(EXPERIMENT),
            "mlx_config_sha256": digest(MLX_CONFIG),
            "model_revision": config["revision"], "model_weights_sha256": config["model_file_sha256"],
            "tokenizer_config_sha256": config["tokenizer_config_sha256"],
            "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                         "physical_memory_bytes": memory.total, "available_bytes": memory.available,
                         "swap_used_bytes": swap.used, "swap_free_bytes": swap.free,
                         "disk_free_bytes": disk.free}}


def predict(model, tokenizer, messages, max_tokens):
    from mlx_lm import stream_generate

    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    chunks = []
    last = None
    for response in stream_generate(model, tokenizer, ids, max_tokens=max_tokens):
        chunks.append(response.text)
        last = response
    if last is None:
        raise RuntimeError("Generator returned no response metadata")
    return "".join(chunks), len(ids), last.generation_tokens, last.finish_reason, last.peak_memory


def extended_metrics(scored):
    base = aggregate(scored)
    labels = Counter(r["answerability"] for r in scored)
    abstained = [r for r in scored if r["strict_abstention"]]
    true_abstentions = sum(r["answerability"] == "unanswerable" for r in abstained)
    latency_by_label = {}
    for label in ("answerable", "unanswerable"):
        values = [r["latency_seconds"] for r in scored if r["answerability"] == label]
        latency_by_label[label] = statistics.mean(values) if values else None
    return {**base,
            "abstention_precision": true_abstentions / len(abstained) if abstained else None,
            "abstention_recall": true_abstentions / labels["unanswerable"] if labels["unanswerable"] else None,
            "format_status_counts": dict(Counter(r["format_status"] for r in scored)),
            "generation_truncated_count": sum(r["generation_truncated"] for r in scored),
            "mean_answer_latency_seconds_by_label": latency_by_label,
            "maximum_generation_tokens_observed": max((r["generation_tokens"] for r in scored), default=None),
            "minimum_swap_free_bytes": min((r["system_swap_free_bytes"] for r in scored), default=None)}


def run(condition, output, smoke=False):
    if output.exists() or output.with_suffix(".manual_audit.jsonl").exists():
        raise FileExistsError("Refusing to overwrite versioned evaluation output")
    pre = preflight(condition)
    if not pre["ready"]:
        raise RuntimeError(json.dumps(pre))
    from mlx_lm import load
    model, tokenizer = load(str(MODEL))
    eos = list(tokenizer.eos_token_ids)
    if tokenizer.convert_tokens_to_ids("<|im_end|>") not in eos:
        raise RuntimeError("MLX tokenizer does not stop on Qwen <|im_end|>")
    source = list(read_rows(CONDITIONS[condition]))
    if smoke:
        source = source[:1]
    max_tokens = json.loads(EXPERIMENT.read_text())["maximum_generation_tokens"]
    scored = []
    result = {"schema_version": 1, "status": "running", "condition": condition,
              "preflight": pre, "expected_questions": len(source), "inference": {"adapter": None, "batch_size": 1,
              "enable_thinking": False, "sampler": "MLX-LM default greedy", "max_new_tokens": max_tokens,
              "eos_token_ids": eos, "stopping": "EOS or max_new_tokens"},
              "predictions": scored}
    temp = output.with_suffix(".partial.json")
    if temp.exists():
        raise FileExistsError("Existing partial evaluation output must be reviewed")
    try:
        for index, row in enumerate(source, 1):
            if psutil.virtual_memory().available < 900_000_000 or psutil.swap_memory().free < 400_000_000:
                raise MemoryError("Runtime memory guard: below 0.9 GB available or 0.4 GB free swap")
            started = time.perf_counter()
            raw, prompt_tokens, generation_tokens, finish_reason, model_peak_gb = predict(
                model, tokenizer, row["messages"][:-1], max_tokens)
            elapsed = time.perf_counter() - started
            if prompt_tokens != row["prompt_tokens"]:
                raise RuntimeError(f"Tokenizer drift for {row['question_id']}: {prompt_tokens} != {row['prompt_tokens']}")
            parsed = parse_response(raw, finish_reason)
            system = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory = {"process_rss_bytes": psutil.Process().memory_info().rss,
                      "model_peak_bytes": int(model_peak_gb * 1e9),
                      "system_available_bytes": system.available,
                      "system_swap_free_bytes": swap.free}
            scored_row = audit_prediction(row, parsed["normalized_response"], elapsed, memory)
            # The portable scorer accepts case variants; this experiment's contract
            # requires the exact uppercase token after surrounding whitespace.
            scored_row["strict_abstention"] = parsed["exact_unanswerable"]
            scored_row.update({"raw_response": raw, **parsed, "generation_tokens": generation_tokens,
                               "finish_reason": finish_reason, "prompt_tokens": prompt_tokens,
                               "input_passage_sha256": {p["passage_id"]: p["sha256"] for p in row["selected_passages"]},
                               "input_sha256": pre["validation_input_sha256"],
                               "model_weights_sha256": pre["model_weights_sha256"],
                               "experiment_config_sha256": pre["experiment_config_sha256"],
                               "error": None})
            scored.append(scored_row)
            result["metrics"] = extended_metrics(scored)
            temp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            if index % 10 == 0:
                print(f"{condition}: {index}/{len(source)}; swap free {swap.free / 1e9:.2f} GB", flush=True)
        result["status"] = "smoke_complete" if smoke else "complete"
    except Exception as exc:
        result["status"] = "stopped_or_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failed_question_id"] = row["question_id"] if "row" in locals() else None
    result["metrics"] = extended_metrics(scored)
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temp.rename(output)
    review = manual_audit_rows(source, scored)
    with output.with_suffix(".manual_audit.jsonl").open("x", encoding="utf-8") as stream:
        for item in review:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Load untuned model and generate")
    parser.add_argument("--smoke", action="store_true", help="Run only the first fixed question before the full evaluation")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"mode": "preflight_only", **preflight(args.condition)}, indent=2))
    else:
        if args.output is None:
            parser.error("--output required with --execute")
        result = run(args.condition, args.output, args.smoke)
        print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                          "error": result.get("error")}, indent=2))
        if result["status"] not in ("complete", "smoke_complete"):
            raise SystemExit(1)
