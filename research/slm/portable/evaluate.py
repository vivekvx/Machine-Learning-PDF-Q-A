"""Opt-in fixed-validation inference and metrics; no training or downloads."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import psutil

from scoring import aggregate, audit_prediction, manual_audit_rows
from training_data import local_checkpoint_complete

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data/v1"
MODEL = ROOT / ".cache/qwen3-0.6b-4bit"
CONDITIONS = {"gold": "validation_gold.jsonl",
              "tfidf_faiss": "validation_tfidf_faiss.jsonl",
              "minilm_faiss": "validation_minilm_faiss.jsonl"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def input_file(condition):
    return DATA / CONDITIONS[condition]


def preflight(backend, condition, model_dir=None, adapter=None):
    path = input_file(condition)
    errors = []
    if not path.exists():
        errors.append(f"Missing condition file: {path}; run retrieve.py for this condition")
        return {"ready": False, "errors": errors}
    sample = rows(path)
    if len(sample) != 80 or any(r["split"] != "validation" for r in sample):
        errors.append("Validation file must have the fixed 80 validation questions")
    reference_ids = [r["question_id"] for r in rows(DATA / "validation_gold.jsonl")]
    if [r["question_id"] for r in sample] != reference_ids:
        errors.append("Condition question order differs from fixed gold sample")
    if condition == "gold":
        expected = json.loads((DATA / "manifest.json").read_text())["file_sha256"][path.name]
    else:
        retrieval_manifest = json.loads((DATA / f"validation_{condition}.manifest.json").read_text())
        expected = retrieval_manifest["output_sha256"]
        if retrieval_manifest["source_sha256"] != sha(DATA / "validation_gold.jsonl"):
            errors.append("Retrieval condition built from a different gold sample")
        if retrieval_manifest["retrieval_config_sha256"] != sha(HERE / "v1/retrieval.json"):
            errors.append("Retrieval configuration checksum mismatch")
    if sha(path) != expected:
        errors.append("Condition checksum mismatch")
    disk = shutil.disk_usage(HERE).free
    if disk < 1_000_000_000:
        errors.append("Less than 1 GB disk free for evaluation output")
    machine = {"platform": platform.platform(), "machine": platform.machine(),
               "physical_memory_bytes": psutil.virtual_memory().total,
               "available_bytes": psutil.virtual_memory().available,
               "swap_free_bytes": psutil.swap_memory().free,
               "disk_free_bytes": disk}
    backend_config = json.loads((HERE / f"v1/{'mlx' if backend == 'mlx' else 'gpu_peft'}.json").read_text())
    if backend == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            errors.append("MLX backend requires Apple Silicon macOS")
        weights = MODEL / "model.safetensors"
        if not weights.exists() or sha(weights) != backend_config["model_file_sha256"]:
            errors.append("Pinned local MLX model file missing or checksum mismatch")
        if sha(MODEL / "tokenizer_config.json") != backend_config["tokenizer_config_sha256"]:
            errors.append("MLX tokenizer configuration checksum mismatch")
        if machine["available_bytes"] < 1_200_000_000 or machine["swap_free_bytes"] < 750_000_000:
            errors.append("MLX inference memory preflight below 1.2 GB available or 0.75 GB free swap")
        try:
            if importlib.metadata.version("mlx-lm") != backend_config["mlx_lm"]:
                errors.append("mlx-lm version differs from tested pin")
        except importlib.metadata.PackageNotFoundError:
            errors.append("mlx-lm is not installed")
        if adapter and not (adapter / "adapters.safetensors").exists():
            errors.append("MLX adapter is missing adapters.safetensors")
    else:
        if not model_dir or not (model_dir / "config.json").exists():
            errors.append("GPU model snapshot path with config.json is required")
        elif backend_config["revision"] not in model_dir.resolve().parts:
            errors.append("GPU model directory is not the pinned revision snapshot")
        if model_dir and (model_dir / "config.json").exists() and not local_checkpoint_complete(model_dir):
            errors.append("Pinned GPU base model weights are not fully cached")
        if model_dir and (model_dir / "tokenizer_config.json").exists():
            if sha(model_dir / "tokenizer_config.json") != backend_config["tokenizer_config_sha256"]:
                errors.append("GPU tokenizer configuration checksum mismatch")
        else:
            errors.append("GPU tokenizer configuration missing")
        for package, expected_version in backend_config["dependency_versions"].items():
            try:
                if importlib.metadata.version(package) != expected_version:
                    errors.append(f"{package} version differs from proposed pin {expected_version}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing GPU dependency: {package}")
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA GPU unavailable")
            elif torch.cuda.mem_get_info()[0] < backend_config["minimum_cuda_free_bytes"]:
                errors.append("Free CUDA memory below configured 6 GB preflight")
        except Exception as exc:
            errors.append(f"PyTorch import/CUDA preflight failed: {exc}")
        if adapter and not (adapter / "adapter_model.safetensors").exists():
            errors.append("PEFT adapter is missing adapter_model.safetensors")
    return {"ready": not errors, "errors": errors, "hardware": machine,
            "backend": backend, "condition": condition, "input_sha256": sha(path),
            "backend_config_sha256": sha(HERE / f"v1/{'mlx' if backend == 'mlx' else 'gpu_peft'}.json"),
            "model_revision": backend_config["revision"]}


def mlx_engine(adapter):
    import mlx.core as mx
    from mlx_lm import generate, load

    model, tokenizer = load(str(MODEL), adapter_path=str(adapter) if adapter else None)
    def predict(messages, max_tokens):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return text.strip(), count, mx.get_peak_memory()
    return predict


def gpu_engine(model_dir, adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                 quantization_config=quant, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    def predict(messages, max_tokens):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            output = model.generate(**tokens, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0, tokens["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, tokens["input_ids"].shape[1], torch.cuda.max_memory_allocated()
    return predict


def write_jsonl(path, data):
    with path.open("w", encoding="utf-8") as stream:
        for row in data:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def execute(args, pre):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {args.output}")
    audit_path = args.output.with_suffix(".manual_audit.jsonl")
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite manual audit: {audit_path}")
    input_rows = rows(input_file(args.condition))
    experiment = json.loads((HERE / "v1/experiment.json").read_text())
    predict = mlx_engine(args.adapter) if args.backend == "mlx" else gpu_engine(args.model_dir, args.adapter)
    predictions = []
    result = {"status": "running", "preflight": pre, "input_rows": len(input_rows),
              "config_sha256": sha(HERE / "v1/experiment.json"),
              "adapter_path": str(args.adapter) if args.adapter else None,
              "predictions": predictions}
    try:
        for row in input_rows:
            if args.backend == "mlx" and (psutil.virtual_memory().available < 900_000_000 or
                                          psutil.swap_memory().free < 400_000_000):
                raise MemoryError("MLX runtime memory guard stopped evaluation")
            start = time.perf_counter()
            text, prompt_count, peak = predict(row["messages"][:-1], experiment["maximum_generation_tokens"])
            elapsed = time.perf_counter() - start
            if prompt_count != row["prompt_tokens"]:
                raise RuntimeError(f"Tokenizer/template drift: {row['question_id']}")
            memory = {"process_rss_bytes": psutil.Process().memory_info().rss,
                      "model_peak_bytes": peak,
                      "system_available_bytes": psutil.virtual_memory().available,
                      "system_swap_free_bytes": psutil.swap_memory().free}
            predictions.append(audit_prediction(row, text, elapsed, memory))
            result["metrics"] = aggregate(predictions)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "stopped_or_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["metrics"] = aggregate(predictions)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_jsonl(audit_path, manual_audit_rows(input_rows, predictions))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "gpu"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--model-dir", type=Path, help="Pinned local HF snapshot path for GPU")
    parser.add_argument("--adapter", type=Path, help="Backend-specific adapter directory")
    parser.add_argument("--output", type=Path, help="New output JSON path")
    parser.add_argument("--execute", action="store_true", help="Actually load the model and run 80 prompts")
    args = parser.parse_args()
    pre = preflight(args.backend, args.condition, args.model_dir, args.adapter)
    if not args.execute:
        print(json.dumps({"mode": "preflight_only", **pre}, indent=2))
        return
    if not pre["ready"]:
        raise SystemExit(json.dumps(pre, indent=2))
    if args.output is None:
        raise SystemExit("--output is required with --execute")
    result = execute(args, pre)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
