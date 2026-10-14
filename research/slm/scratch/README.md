# SLM-A: from-scratch MLX feasibility experiment

This isolated workspace builds a **randomly initialized** decoder-only Transformer and a byte-level BPE tokenizer learned only from deterministic synthetic training text. It uses no pretrained weights, tokenizer, model service, QASPER records, or existing RAG code. The synthetic corpus is an engineering fixture; its losses and generated text are **not academic question-answering results**.

## Rebuild and inspect

Use Apple Silicon macOS and Python 3.14.6. Commands run from the repository root:

```sh
uv venv --python /opt/homebrew/opt/python@3.14/bin/python3.14 research/slm/scratch/.venv
uv pip install --python research/slm/scratch/.venv/bin/python -r research/slm/scratch/requirements.txt
research/slm/scratch/.venv/bin/python research/slm/scratch/prepare_data.py --output-dir /tmp/slma-rebuild
PYTHONPATH=research/slm/scratch research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -v
research/slm/scratch/.venv/bin/python research/slm/scratch/run.py --mode smoke
research/slm/scratch/.venv/bin/python research/slm/scratch/freeze.py --output /tmp/slma-rebuild-manifest.json
```

The committed `data/v1/` files are immutable. A rebuild must go to a fresh directory; compare its manifest and file SHA-256 hashes with `data/v1/manifest.json`. The tokenizer is fitted to `train.jsonl` only. Validation records have separate IDs and text. The loader verifies hashes and overlap.

## Guarded, bounded experiment

The following are **opt-in**. The runner first records two samples of available memory, swap use and activity, memory pressure, and disk headroom. On macOS, `vm.swapusage` free space describes currently allocated swap files, not a fixed swap ceiling, so the two historical `swap_free` thresholds in the unchanged `config.v1.json` are diagnostic only. The effective guard still checks available memory, memory pressure, disk space, swap growth, swap-out activity, process RSS, and the time cap. While training runs, a separate parent process checks resources every 0.5 seconds and terminates the worker if a stop threshold is crossed. See `MEMORY_GUARD_AUDIT.md` for sources and the measured five-step result. Never run `train_worker.py` directly.

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/run.py --mode smoke --execute --run-dir research/slm/scratch/runs/v1-smoke
research/slm/scratch/.venv/bin/python research/slm/scratch/run.py --mode overfit --resume research/slm/scratch/runs/v1-smoke/checkpoint --execute --run-dir research/slm/scratch/runs/v1-overfit
research/slm/scratch/.venv/bin/python research/slm/scratch/generate.py --checkpoint research/slm/scratch/runs/v1-smoke/checkpoint --prompt 'Record 0001.' --max-new-tokens 32
```

Only run the overfit stage after a complete five-step smoke result. It resumes that checkpoint, repeats one synthetic training window for at most 12 further steps, and stops sooner if training loss reaches 1.5. A lower loss on this tiny fixture shows only that the implementation can fit the fixture. `generate.py` uses greedy decoding with a 64-token hard cap, refuses context overflow, and also requires a safe memory preflight.

Each run uses a fresh directory and refuses overwrite. `run.json` records preflight, active resource snapshots, completed steps, throughput and termination status. `steps.jsonl` records individual loss, duration, token throughput and MLX allocation. The checkpoint stores model and AdamW tensors in safetensors with hashes, step number, sample cursor, and config/data/source provenance. Resume rejects changed provenance. All files live under this scratch workspace.

## Six-sequence memorization probe

The separate `overfit/v1/` fixture contains six short synthetic sequences. `overfit_run.py` starts from fresh random weights with the same architecture, tokenizer, seed, batch size, AdamW learning rate, and weight decay as `config.v1.json`. Its separate configuration caps the experiment at 200 steps and 120 seconds, logs full-set loss every ten steps, and saves checkpoints every 25 steps. The parent reuses the memory-pressure, available-RAM, swap-growth, swap-out, RSS, and disk guards. This is a memorization check, not a generalization benchmark.

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/prepare_overfit.py --output-dir /tmp/slma-overfit-rebuild
research/slm/scratch/.venv/bin/python research/slm/scratch/overfit_run.py
research/slm/scratch/.venv/bin/python research/slm/scratch/overfit_run.py --execute --run-dir research/slm/scratch/runs/new-overfit-run
research/slm/scratch/.venv/bin/python research/slm/scratch/export_curve.py --run-dir research/slm/scratch/runs/new-overfit-run
research/slm/scratch/.venv/bin/python research/slm/scratch/freeze_overfit.py --run-dir research/slm/scratch/runs/new-overfit-run
```

Use fresh output directories; the committed v1 fixture and run are immutable. See [OVERFIT_REPORT.md](OVERFIT_REPORT.md) for the completed measurement and its limits.

## Official Python documentation pilot

The separate `real_text/v1/` corpus contains audited prose spans from four pinned CPython 3.12.15 documentation pages. Three entire source documents are train-only and one is validation-only. `real_text/prepare.py --verify` checks source hashes and rights notices, reconstructs selected text, and refits the 1,024-ID byte BPE tokenizer from training documents in memory. It does not change the frozen v1 corpus. See [REAL_TEXT_PILOT_REPORT.md](REAL_TEXT_PILOT_REPORT.md) and [the rights ledger](real_text/RIGHTS_PROVENANCE.md).

```sh
python3 research/slm/scratch/real_text/prepare.py --verify
PYTHONPATH=research/slm/scratch research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -v
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/pilot.py
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/pilot.py --execute --run-dir research/slm/scratch/runs/NEW_UNIQUE_RUN_NAME
```

The last command is opt-in and capped at 40 optimizer steps. It creates a fresh model, uses the original parent watchdog thresholds, and refuses an existing run directory. Do not invoke `real_text/worker.py` directly. The completed v1 run is under `runs/v1-python-docs-pilot/`.

## Limits

Batch size is 1 and gradient accumulation is 1. Gradient checkpointing is off because the initial model is small; adding complexity without a measured memory need would obscure the feasibility result. The current 256-token windows are packed across synthetic records within a split. This fixture is not suitable for real QA, and no synthetic metric should be compared with QASPER or the frozen Qwen baseline.
