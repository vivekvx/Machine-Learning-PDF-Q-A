"""Opt-in, guarded MLX or CUDA/PEFT training; default invocation is preflight only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import psutil

from training_data import (MLXCompletionDataset, PeftCompletionDataset,
                           local_checkpoint_complete, padded_collator, rows, tokenized_row)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data/v1/train.jsonl"
VALID = HERE / "data/v1/validation_gold.jsonl"
MODEL = ROOT / ".cache/qwen3-0.6b-4bit"
CONFIG = HERE / "v1/experiment.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_for(backend):
    return json.loads((HERE / f"v1/{'mlx' if backend == 'mlx' else 'gpu_peft'}.json").read_text())


def load_tokenizer(backend, model_dir):
    if backend == "mlx":
        from mlx_lm.utils import load_tokenizer
        return load_tokenizer(str(MODEL))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def preflight(backend, model_dir, output_dir):
    cfg = config_for(backend)
    experiment = json.loads(CONFIG.read_text())
    errors = []
    if output_dir is not None and output_dir.exists():
        errors.append("Output directory already exists; training never overwrites it")
    manifest = json.loads((HERE / "data/v1/manifest.json").read_text())
    if sha(DATA) != manifest["file_sha256"]["train.jsonl"]:
        errors.append("Training data checksum mismatch")
    train = list(rows(DATA))
    valid = list(rows(VALID))
    if len(train) != 352 or len(valid) != 80:
        errors.append("Unexpected fixed train/validation sample size")
    if {r["document_id"] for r in train} & {r["document_id"] for r in valid}:
        errors.append("Train/validation document overlap")
    if any(r["split"] != "train" for r in train) or any(r["split"] != "validation" for r in valid):
        errors.append("Invalid split label")
    disk = shutil.disk_usage(HERE).free
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    if disk < cfg["minimum_free_disk_bytes"]:
        errors.append("Insufficient free disk")
    if backend == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            errors.append("MLX requires Apple Silicon macOS")
        if memory.available < cfg["minimum_available_bytes"]:
            errors.append("Available unified memory below configured 2 GB minimum")
        if swap.free < cfg["minimum_free_swap_bytes"]:
            errors.append("Free swap below configured 2 GB minimum")
        weights = MODEL / "model.safetensors"
        if not weights.exists() or sha(weights) != cfg["model_file_sha256"]:
            errors.append("Pinned MLX 4-bit checkpoint missing or checksum mismatch")
        if sha(MODEL / "tokenizer_config.json") != cfg["tokenizer_config_sha256"]:
            errors.append("MLX tokenizer configuration checksum mismatch")
        for package, version in (("mlx-lm", cfg["mlx_lm"]), ("mlx", cfg["mlx"])):
            try:
                if importlib.metadata.version(package) != version:
                    errors.append(f"{package} version differs from tested pin {version}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing dependency: {package}")
    else:
        if not model_dir or not (model_dir / "config.json").exists():
            errors.append("Local pinned HF model snapshot with config.json is required")
        elif cfg["revision"] not in model_dir.resolve().parts:
            errors.append("GPU model path is not the pinned revision snapshot")
        if model_dir and (model_dir / "config.json").exists() and not local_checkpoint_complete(model_dir):
            errors.append("Pinned GPU base model weights are not fully cached")
        if model_dir and (model_dir / "tokenizer_config.json").exists():
            if sha(model_dir / "tokenizer_config.json") != cfg["tokenizer_config_sha256"]:
                errors.append("GPU tokenizer configuration checksum mismatch")
        else:
            errors.append("GPU tokenizer configuration missing")
        for package, version in cfg["dependency_versions"].items():
            try:
                if importlib.metadata.version(package) != version:
                    errors.append(f"{package} differs from proposed pin {version}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing GPU dependency: {package}")
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA GPU unavailable")
            elif torch.cuda.mem_get_info()[0] < cfg["minimum_cuda_free_bytes"]:
                errors.append("Free CUDA memory below configured 6 GB minimum")
            elif not torch.cuda.is_bf16_supported():
                errors.append("CUDA GPU does not support bfloat16 compute")
        except Exception as exc:
            errors.append(f"PyTorch import/CUDA preflight failed: {exc}")
    if not errors:
        try:
            tokenizer = load_tokenizer(backend, model_dir)
            for row in train:
                tokenized_row(row, tokenizer, experiment["maximum_sequence_tokens"])
        except Exception as exc:
            errors.append(f"Tokenizer, chat template, or assistant mask preflight failed: {exc}")
    return {"ready": not errors, "errors": errors, "backend": backend,
            "model_revision": cfg["revision"], "training_data_sha256": sha(DATA),
            "config_sha256": sha(CONFIG),
            "backend_config_sha256": sha(HERE / f"v1/{'mlx' if backend == 'mlx' else 'gpu_peft'}.json"),
            "train_rows": len(train), "validation_rows": len(valid),
            "available_bytes": memory.available, "swap_free_bytes": swap.free,
            "disk_free_bytes": disk, "physical_memory_bytes": memory.total}


def mlx_worker(output_dir):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load, lora
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.callbacks import TrainingCallback

    cfg = config_for("mlx")
    experiment = json.loads(CONFIG.read_text())
    np.random.seed(experiment["seed"])
    model, tokenizer = load(str(MODEL))
    dataset = MLXCompletionDataset(DATA, tokenizer, experiment["maximum_sequence_tokens"])
    class Record(TrainingCallback):
        def on_train_loss_report(self, item):
            record = {**item, "time": time.time(), "available_bytes": psutil.virtual_memory().available,
                      "swap_free_bytes": psutil.swap_memory().free,
                      "process_rss_bytes": psutil.Process().memory_info().rss}
            with (output_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    args = dict(lora.CONFIG_DEFAULTS)
    args.update(model=str(MODEL), train=True, data=str(DATA), seed=experiment["seed"],
                adapter_path=str(output_dir / "mlx_adapter"), fine_tune_type="lora",
                optimizer=cfg["optimizer"],
                num_layers=cfg["trainable_last_layers"], batch_size=cfg["batch_size"],
                iters=cfg["train_steps"], learning_rate=cfg["learning_rate"],
                steps_per_report=1, steps_per_eval=cfg["train_steps"] + 1,
                save_every=cfg["train_steps"], max_seq_length=cfg["maximum_sequence_tokens"],
                grad_checkpoint=cfg["gradient_checkpointing"], mask_prompt=True,
                lora_parameters={"rank": cfg["lora_rank"], "scale": cfg["lora_scale"],
                                 "dropout": cfg["lora_dropout"], "keys": cfg["target_modules"]})
    trainer.maybe_set_recommended_wired_limit = lambda: None
    lora.train_model(types.SimpleNamespace(**args), model, dataset, [], Record())
    mx.eval(model.parameters())


def gpu_worker(output_dir, model_dir):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    cfg = config_for("gpu")
    experiment = json.loads(CONFIG.read_text())
    set_seed(experiment["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = PeftCompletionDataset(DATA, tokenizer, experiment["maximum_sequence_tokens"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                 quantization_config=quant, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
    n_layers = model.config.num_hidden_layers
    lora_cfg = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"], bias="none",
                          target_modules=cfg["target_modules"],
                          layers_to_transform=list(range(n_layers - cfg["trainable_last_layers"], n_layers)),
                          layers_pattern="layers", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    class Record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            record = {"step": state.global_step, "time": time.time(), "logs": logs or {},
                      "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                      "process_rss_bytes": psutil.Process().memory_info().rss}
            with (output_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    targs = TrainingArguments(output_dir=str(output_dir / "trainer"),
                              per_device_train_batch_size=cfg["batch_size"],
                              gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                              max_steps=cfg["train_steps"], learning_rate=cfg["learning_rate"],
                              optim=cfg["optimizer"], max_grad_norm=cfg["max_grad_norm"],
                              logging_steps=1, save_strategy="no", eval_strategy="no", bf16=True,
                              gradient_checkpointing=cfg["gradient_checkpointing"],
                              dataloader_num_workers=0, report_to="none",
                              seed=experiment["seed"], data_seed=experiment["seed"])
    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=lambda examples: padded_collator(examples, tokenizer.pad_token_id),
                      callbacks=[Record()])
    trainer.train()
    model.save_pretrained(output_dir / "peft_adapter")
    tokenizer.save_pretrained(output_dir / "peft_adapter")


def guarded_run(args, pre):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    (output_dir / "preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    shutil.copy2(CONFIG, output_dir / "experiment_config.json")
    shutil.copy2(HERE / f"v1/{'mlx' if args.backend == 'mlx' else 'gpu_peft'}.json",
                 output_dir / "backend_config.json")
    env = os.environ.copy()
    env["SLM_PORTABLE_WORKER"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--backend", args.backend,
               "--output-dir", str(output_dir), "--worker"]
    if args.model_dir:
        command += ["--model-dir", str(args.model_dir)]
    start = time.monotonic()
    status = {"status": "running", "backend": args.backend,
              "training_data_sha256": pre["training_data_sha256"], "preflight": pre}
    with (output_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            reasons = []
            if mem.available < 1_000_000_000:
                reasons.append("system available memory below 1 GB")
            if args.backend == "mlx" and swap.free < 750_000_000:
                reasons.append("free swap below 0.75 GB")
            if args.backend == "mlx" and pre["swap_free_bytes"] - swap.free > 1_000_000_000:
                reasons.append("swap growth above 1 GB")
            if time.monotonic() - start > 3600:
                reasons.append("one-hour limit")
            if reasons:
                status["stop_reasons"] = reasons
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(1)
        status["exit_code"] = process.wait()
    status["elapsed_seconds"] = time.monotonic() - start
    status["status"] = "complete" if status["exit_code"] == 0 else (
        "stopped_by_guard" if status.get("stop_reasons") else "worker_failed")
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "gpu"), required=True)
    parser.add_argument("--model-dir", type=Path, help="Pinned local HF snapshot for GPU")
    parser.add_argument("--output-dir", type=Path, help="New isolated output directory")
    parser.add_argument("--execute", action="store_true", help="Start configured training after preflight")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if os.getenv("SLM_PORTABLE_WORKER") != "1" or not args.output_dir:
            raise SystemExit("Worker mode is available only from an explicit guarded launch")
        if args.backend == "mlx":
            mlx_worker(args.output_dir)
        else:
            gpu_worker(args.output_dir, args.model_dir)
        return
    pre = preflight(args.backend, args.model_dir, args.output_dir)
    if not args.execute:
        print(json.dumps({"mode": "preflight_only", **pre}, indent=2))
        return
    if not args.output_dir:
        raise SystemExit("--output-dir is required with --execute")
    if not pre["ready"]:
        raise SystemExit(json.dumps(pre, indent=2))
    status = guarded_run(args, pre)
    print(json.dumps(status, indent=2))
    if status["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
