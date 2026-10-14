"""Memory-guarded, batch-1 Qwen3-0.6B MLX validation baseline. No training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from metrics import exact_match, is_abstention, summarize, token_f1

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "mlx-community/Qwen3-0.6B-4bit"
MODEL_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
MODEL_DIR = ROOT / ".cache" / "qwen3-0.6b-4bit"
MODEL_SHA256 = "392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2"
MIN_FREE_BEFORE = 1_200_000_000
MIN_FREE_DURING = 900_000_000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def prompt_for(row, tokenizer, max_sequence_length, generation_tokens):
    passages = row["selected_passages"]
    instructions = ("Answer the question using only the provided research-paper passages. "
                    "If the passages do not establish an answer, output exactly UNANSWERABLE. "
                    "Otherwise output only a concise answer. Do not invent citations or page numbers.")
    def render(items):
        context = "\n\n".join(f"[section {p['section_index']}, evidence {p['evidence_id']}] {p['text']}" for p in items)
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": f"Document: {row['document_id']} — {row['title']}\nPassages:\n{context}\n\nQuestion: {row['question']}"}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    current = [dict(p) for p in passages]
    while True:
        text = render(current)
        count = len(tokenizer.encode(text))
        if count + generation_tokens <= max_sequence_length:
            return text, count, current
        if len(current) > 1:
            current.pop()
        elif current and len(current[0]["text"]) > 150:
            current[0]["text"] = current[0]["text"][:max(150, len(current[0]["text"]) // 2)]
        else:
            raise ValueError(f"question {row['question_id']} cannot fit configured sequence length")


def run(args):
    import mlx.core as mx
    import mlx_lm
    import psutil
    from mlx_lm import load, generate

    if args.batch_size != 1:
        raise ValueError("This 8 GB baseline supports batch size 1 only")
    if args.max_sequence_length < 256 or args.generation_tokens < 1:
        raise ValueError("invalid sequence or generation limit")
    model_file = MODEL_DIR / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Pinned local model missing: {MODEL_DIR}")
    observed_hash = sha256(model_file)
    if observed_hash != MODEL_SHA256:
        raise ValueError(f"model checksum mismatch or incomplete download: {observed_hash}")
    available_before = psutil.virtual_memory().available
    if available_before < MIN_FREE_BEFORE:
        raise MemoryError(f"preflight available memory {available_before} < {MIN_FREE_BEFORE}")
    rows = list(read_jsonl(args.input))[:1 if args.smoke else args.max_examples]
    if not rows:
        raise ValueError("no evaluation rows")
    output = {"status": "started", "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
              "base_model_repo": "Qwen/Qwen3-0.6B", "conversion_source_base_revision": "not specified by conversion card",
              "model_format": "4-bit MLX conversion",
              "mlx_lm_version": mlx_lm.__version__ if hasattr(mlx_lm, "__version__") else "0.32.0",
              "platform": platform.platform(), "machine": platform.machine(),
              "physical_memory_bytes": psutil.virtual_memory().total,
              "available_before_bytes": available_before,
              "config": {"max_sequence_length": args.max_sequence_length,
                         "generation_tokens": args.generation_tokens, "batch_size": 1,
                         "smoke": args.smoke},
              "input_sha256": sha256(args.input),
              "model_safetensors_sha256": {model_file.name: observed_hash},
              "predictions": []}
    def save():
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        model, tokenizer = load(str(MODEL_DIR))
        output["available_after_load_bytes"] = psutil.virtual_memory().available
        if output["available_after_load_bytes"] < MIN_FREE_DURING:
            raise MemoryError("available unified memory fell below runtime guard after model load")
        for row in rows:
            if psutil.virtual_memory().available < MIN_FREE_DURING:
                raise MemoryError("available unified memory fell below runtime guard")
            prompt, prompt_tokens, used = prompt_for(row, tokenizer, args.max_sequence_length, args.generation_tokens)
            start = time.perf_counter()
            response = generate(model, tokenizer, prompt=prompt, max_tokens=args.generation_tokens, verbose=False)
            elapsed = time.perf_counter() - start
            response = response.strip()
            abstained = is_abstention(response)
            expected_abstention = row["answerability"] == "unanswerable"
            prediction = {"question_id": row["question_id"], "document_id": row["document_id"],
                          "condition": row["condition"], "answerability": row["answerability"],
                          "expected_answers": row["expected_answers"], "prediction": response,
                          "exact_match": exact_match(response, row["expected_answers"]),
                          "token_f1": token_f1(response, row["expected_answers"]),
                          "abstention_correct": int(abstained == expected_abstention),
                          "prompt_tokens": prompt_tokens, "passage_ids": [p["evidence_id"] for p in used],
                          "latency_seconds": elapsed,
                          "process_rss_bytes": psutil.Process().memory_info().rss,
                          "mlx_active_bytes": mx.get_active_memory(),
                          "mlx_peak_bytes": mx.get_peak_memory(),
                          "available_memory_bytes": psutil.virtual_memory().available}
            output["predictions"].append(prediction)
            save()
            if prediction["available_memory_bytes"] < MIN_FREE_DURING:
                raise MemoryError("available unified memory fell below runtime guard after generation")
        output["metrics"] = {"overall": summarize(output["predictions"]),
                             "answerable": summarize([p for p in output["predictions"] if p["answerability"] == "answerable"]),
                             "unanswerable": summarize([p for p in output["predictions"] if p["answerability"] == "unanswerable"])}
        output["available_after_run_bytes"] = psutil.virtual_memory().available
        output["status"] = "complete"
        save()
    except (MemoryError, RuntimeError, ValueError) as exc:
        output["status"] = "stopped_unsafe_or_error"
        output["error"] = str(exc)
        output["metrics"] = {"partial": summarize(output["predictions"])}
        save()
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "validation_gold_eval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_validation.json")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    except Exception as exc:
        print(f"Baseline stopped: {exc}", file=sys.stderr)
        sys.exit(1)
