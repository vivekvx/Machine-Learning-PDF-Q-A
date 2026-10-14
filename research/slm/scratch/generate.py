"""Greedy, bounded generation from a locally trained SLM-A checkpoint."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from checkpoint import digest, load_checkpoint
from data import DATA
from model import ScratchTransformer
from run import CONFIG, preflight
from tokenizer import ByteBPE, SPECIAL
from train_worker import provenance


def generate(checkpoint_dir, prompt, max_new_tokens):
    if not 1 <= max_new_tokens <= 64:
        raise ValueError("Generation is capped at 64 new tokens")
    readiness = preflight("smoke")
    if not readiness["ready"]:
        raise RuntimeError("Generation refused by memory preflight: " + "; ".join(readiness["reasons"]))
    config = json.loads(CONFIG.read_text())
    tokenizer = ByteBPE.load(DATA / "tokenizer.json")
    ids = tokenizer.encode(prompt, bos=True)
    if len(ids) + max_new_tokens > config["model"]["context_length"]:
        raise ValueError("Prompt plus requested tokens exceeds context; no silent truncation")
    mx.random.seed(config["seed"])
    model = ScratchTransformer(config["model"], tokenizer.vocab_size)
    optimizer = optim.AdamW(learning_rate=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    metadata = load_checkpoint(checkpoint_dir, model, optimizer,
                               expected_provenance=provenance())
    generated = []
    mx.metal.reset_peak_memory()
    for _ in range(max_new_tokens):
        logits = model(mx.array([ids], dtype=mx.int32))
        token = int(mx.argmax(logits[0, -1]).item())
        if token == SPECIAL["eos"]:
            break
        ids.append(token)
        generated.append(token)
    return {"prompt": prompt, "generated_text": tokenizer.decode(generated),
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "checkpoint_step": metadata["step"],
            "checkpoint_metadata_sha256": digest(Path(checkpoint_dir) / "metadata.json"),
            "mlx_peak_bytes": mx.metal.get_peak_memory()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = generate(args.checkpoint, args.prompt, args.max_new_tokens)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
