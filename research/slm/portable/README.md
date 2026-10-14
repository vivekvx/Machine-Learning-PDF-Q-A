# Portable QASPER/Qwen3 experiment v1

This directory prepares a fixed, document-disjoint research experiment. All
training and model evaluation entrypoints default to **preflight only**. They
load no model weights, start no training, and download nothing unless the
operator explicitly runs separate setup commands.

## Rebuild the data

From the repository root, using the already cached MLX tokenizer:

```sh
research/slm/.venv/bin/python research/slm/portable/prepare.py
research/slm/.venv/bin/python -m unittest discover -s research/slm/portable/tests -v
```

The output is `portable/data/v1/train.jsonl` (352 train-document rows),
`validation_gold.jsonl` (80 validation-document rows), `selection_audit.jsonl`,
an evidence-review queue, and a SHA-256 manifest. The builder never opens a
test question file. It rejects over-budget rows instead of truncating passages.

## Retrieval conditions

`v1/retrieval.json` separately defines the oracle gold-paragraph condition,
per-document word/bigram TF-IDF plus FAISS, and MiniLM dense embeddings plus
FAISS. The query to either retriever is question text only. Ranking happens
before the code attaches reference answers or evidence IDs for auditing.
Candidates that do not fit the 512-token prompt-plus-generation budget are
logged in each output row; passage text is never clipped.

In an environment with `requirements-retrieval.txt` installed:

```sh
python research/slm/portable/retrieve.py --condition tfidf_faiss
python research/slm/portable/retrieve.py --condition minilm_faiss
```

The MiniLM command requires the exact pinned model revision already cached
locally and a working PyTorch installation. It uses `local_files_only=True`.
On this machine, the isolated MLX environment has a working tokenizer and
the repository's application environment has FAISS/sklearn but a broken
PyTorch library. The verified local TF-IDF build used the existing packages
without installing another copy:

```sh
research/slm/.venv/bin/python research/slm/portable/retrieve.py \
  --condition tfidf_faiss \
  --extra-site-packages venv/lib/python3.14/site-packages
```

`--extra-site-packages` is only a local environment workaround. A fresh
environment should install retrieval dependencies normally.

## Model and training preflight

MLX uses the cached pinned 4-bit conversion and
`research/slm/requirements-baseline.lock.txt`:

```sh
research/slm/.venv/bin/python research/slm/portable/train.py --backend mlx
research/slm/.venv/bin/python research/slm/portable/evaluate.py \
  --backend mlx --condition gold
```

The proposed GPU environment is Linux with CUDA and a separate Python 3.11
or 3.12 virtual environment. Install `portable/requirements-gpu.txt` there.
Explicitly obtain the pinned `Qwen/Qwen3-0.6B` Hugging Face snapshot at
`c1899de289a04d12100db370d81485cdf75e47ca` before preflight; pass its
snapshot directory (whose final path component is that commit) as
`--model-dir`. The scripts never fetch missing weights:

```sh
python research/slm/portable/train.py --backend gpu \
  --model-dir /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
python research/slm/portable/evaluate.py --backend gpu --condition gold \
  --model-dir /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

Both commands above only report preflight status. A future operator must add
`--execute` and a **new** `--output-dir` for training or `--output` for
evaluation. Training also requires the preflight to pass memory, swap/storage,
dependency, checksum, split, and tokenizer-mask checks. The configured 200
training steps are a proposal and were **not run**. The parent training process
monitors memory and can terminate its worker; it never changes macOS GPU
limits. Output paths are never overwritten.

The earlier five-step feasibility entrypoint `research/slm/run_feasibility.py`
also defaults to preflight now; it requires `--execute` for a future repeat.
Its worker refuses direct launch. Existing five-step logs and adapter weights
were not changed.

## Evaluation and interpretation

`evaluate.py` uses the same 80 question IDs in gold and retrieved conditions.
It reports answerable exact match/token F1, unanswerable strict abstention,
false abstention on answerable rows, source-identifier citation checks,
automated lexical support proxies, per-question latency, peak model allocation,
process RSS, and available system memory. It writes ten review rows with blank
manual judgments. Lexical support is not semantic entailment; human review is
required. The test split is reserved for a final locked evaluation.

MLX adapters and PyTorch/PEFT adapters use different formats and quantized
base models. Do not load one backend's adapter into the other. A future GPU
trained model needs an explicitly validated export/conversion and inference
comparison before making a claim about 8 GB laptop deployment.

The cross-backend tokenizer parity test needs the upstream tokenizer cached
at the pinned base revision. It only needs tokenizer files, not model weights.
One explicit setup method is to run `huggingface_hub.snapshot_download` for
`Qwen/Qwen3-0.6B` at the pinned commit with `allow_patterns` limited to
`tokenizer*`, `special_tokens_map.json`, and `config.json`.
