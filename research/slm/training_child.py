"""Isolated, five-step MLX-LM QLoRA worker. Launch through run_feasibility.py."""
import json
import os
import time
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm import load
from mlx_lm import lora
from mlx_lm.tuner import trainer
from mlx_lm.tuner.callbacks import TrainingCallback

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / ".cache/qwen3-0.6b-4bit"
DATA = ROOT / "feasibility_data"
ADAPTER = ROOT / "feasibility_adapter"
STEPS = ROOT / "training_steps.jsonl"


class QwenMaskedDataset:
    """Return Qwen chat tokens and the first assistant-target token offset."""

    def __init__(self, path, tokenizer):
        self._data = [json.loads(line) for line in path.read_text().splitlines() if line]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def process(self, row):
        messages = row["messages"]
        full = self.tokenizer.apply_chat_template(messages, enable_thinking=False)
        prefix = self.tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, enable_thinking=False
        )
        if full[:len(prefix)] != prefix or len(full) > 384 or len(prefix) >= len(full) - 1:
            raise ValueError(f"Invalid masked example {row['question_id']}")
        if len(full) != row["full_tokens"] or len(prefix) != row["prompt_tokens"]:
            raise ValueError(f"Tokenizer drift on {row['question_id']}")
        return full, len(prefix)


class StepRecorder(TrainingCallback):
    def on_train_loss_report(self, report):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        record = {
            **report,
            "timestamp": time.time(),
            "step_seconds": 1 / report["iterations_per_second"],
            "available_bytes": mem.available,
            "swap_used_bytes": swap.used,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "mlx_peak_bytes": int(report["peak_memory"] * 1e9),
        }
        with STEPS.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
            out.flush()
            os.fsync(out.fileno())


def main():
    if os.environ.get("SLM_FEASIBILITY_WORKER") != "1":
        raise RuntimeError("Training worker requires an explicit guarded launch")
    np.random.seed(0)
    if STEPS.exists():
        raise RuntimeError("Existing step log: move it before a new run")
    if ADAPTER.exists():
        raise RuntimeError("Existing adapter: move it before a new run")
    model, tokenizer = load(str(MODEL))
    train = QwenMaskedDataset(DATA / "train.jsonl", tokenizer)
    for row in train._data:
        train.process(row)
    args = dict(lora.CONFIG_DEFAULTS)
    args.update(model=str(MODEL), train=True, adapter_path=str(ADAPTER),
                data=str(DATA),
                fine_tune_type="lora", num_layers=2, batch_size=1, iters=5,
                learning_rate=1e-5, steps_per_report=1, steps_per_eval=100,
                save_every=100, val_batches=1, max_seq_length=384,
                grad_checkpoint=True, grad_accumulation_steps=1,
                lora_parameters={"rank": 4, "dropout": 0.0, "scale": 8.0},
                mask_prompt=True)
    # MLX-LM's trainer otherwise changes the process's recommended wired limit.
    trainer.maybe_set_recommended_wired_limit = lambda: None
    lora.train_model(types.SimpleNamespace(**args), model, train, [], StepRecorder())
    mx.eval(model.parameters())
    print(json.dumps({"completed_steps": 5, "adapter": str(ADAPTER)}), flush=True)


if __name__ == "__main__":
    main()
