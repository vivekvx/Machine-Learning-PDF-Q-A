"""Load the isolated adapter and run one answerable and one unanswerable validation prompt."""
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "feasibility_data/valid.jsonl"
ADAPTER = ROOT / "feasibility_adapter"
MODEL = ROOT / ".cache/qwen3-0.6b-4bit"
OUT = ROOT / "adapter_validation.json"


def main():
    if OUT.exists():
        raise RuntimeError("Existing validation output: move it before rerunning")
    if psutil.virtual_memory().available < 1_200_000_000:
        raise MemoryError("Insufficient available memory for adapter verification")
    if psutil.swap_memory().free < 300_000_000:
        raise MemoryError("Insufficient free swap for adapter verification")
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    chosen = [next(row for row in rows if row["answerability"] == label)
              for label in ("answerable", "unanswerable")]
    result = {"status": "started", "input_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256((ADAPTER / "adapters.safetensors").read_bytes()).hexdigest(),
              "predictions": [], "available_before_bytes": psutil.virtual_memory().available,
              "swap_free_before_bytes": psutil.swap_memory().free}
    try:
        model, tokenizer = load(str(MODEL), adapter_path=str(ADAPTER))
        result["adapter_loaded"] = True
        for row in chosen:
            if psutil.virtual_memory().available < 900_000_000 or psutil.swap_memory().free < 250_000_000:
                raise MemoryError("Runtime memory guard stopped adapter inference")
            prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
            start = time.perf_counter()
            response = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False).strip()
            result["predictions"].append({"question_id": row["question_id"],
                                          "answerability": row["answerability"],
                                          "expected_answer": row["messages"][-1]["content"],
                                          "prediction": response, "latency_seconds": time.perf_counter() - start,
                                          "mlx_peak_bytes": mx.get_peak_memory(),
                                          "available_bytes": psutil.virtual_memory().available,
                                          "swap_free_bytes": psutil.swap_memory().free})
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed_or_stopped"
        result["error"] = str(exc)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
