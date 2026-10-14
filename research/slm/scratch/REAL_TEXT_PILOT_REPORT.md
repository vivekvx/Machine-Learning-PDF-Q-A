# SLM-A real-text pilot — Python 3.12 documentation, v1

## Outcome

A fresh random instance of the existing four-layer decoder was trained for **40 capped optimizer steps** on a small selection of official Python 3.12 documentation prose. It used no pretrained weights, synthetic checkpoint, external model API, QASPER record, or production RAG file. The new tokenizer was fitted only on training documents. The guarded run completed, and a saved checkpoint reproduced logits exactly. The generated text remains repetitive and unusable as technical prose; this is an infrastructure and short optimization check, not evidence of language competence or academic QA ability.

## Source, rights, and corpus

- Official source: [CPython `v3.12.15`](https://github.com/python/cpython/releases/tag/v3.12.15), peeled commit `e848e4b09ca4bc67c0041c9ce7526dae471d4c36`; tag timestamp **2026-09-30 22:04:19 UTC**; fetched **2026-10-12 04:55 UTC**. Exact URLs and source hashes are in the [corpus manifest](real_text/v1/manifest.json).
- The pinned official [documentation license](real_text/v1/raw/Doc/license.rst) states **PSF License Version 2**. The complete [root LICENSE](real_text/v1/raw/LICENSE) is retained. Attribution, copyright notice, change-summary requirement, selected page authors, and residual provenance limits are in the [rights ledger](real_text/RIGHTS_PROVENANCE.md).
- Train: **3 distinct source documents**, 25 selected prose spans, 11,123 UTF-8 bytes. Validation: **1 separate source document**, 8 spans, 3,960 bytes. Split by document path before tokenizer fitting or token windows; no test set is used.
- Source documents: `heapq.rst`, `bisect.rst`, and `array.rst` for train; `functools.rst` for validation. Selection uses reviewed source-line spans, removes RST syntax/page furniture, and excludes code examples, external links, tables, and cited third-party recipes. Exact selected-paragraph duplicates: **0**. The sample is narrow and was not independently reviewed by a legal specialist.
- The [quality audit](real_text/v1/quality_audit.json) records first/last retained spans for each source document and checks every retained span for exact duplicate paragraphs, medical keywords, and obvious RST/page-furniture patterns. These automated checks are not an independent human quality assessment.
- Fixed 128-target-token, noncrossing windows: **21 train**, **8 validation**. A short document tail is dropped; no validation token is used for optimization.

## Tokenizer and model

The byte-level BPE tokenizer is learned from the three training documents alone. Target and actual vocabulary size: **1,024** (4 special IDs, 256 byte IDs, 764 learned merges). Validation never enters merge selection. Encode/decode roundtrip passed on all four documents; byte fallback gives **0 unknown tokens** by design. Measured efficiency is **3.995 bytes/token** on train and **2.432 bytes/token** on validation; used vocabulary IDs are 720 and 372 respectively. Lower held-out compression is expected with such a small fitting corpus and is a limitation.

The unchanged decoder architecture is 4 layers, width 128, 4 attention heads, context 256, with the existing causal attention, RMSNorm, feed-forward blocks, and untied output head. At vocabulary 1,024, the actual count is **1,082,496 parameters**. It was freshly seeded with `20261012`. No previous checkpoint was loaded. Training uses AdamW, batch size 1, learning rate `0.0005`, weight decay `0.01`, and sequence length 128. These real-text settings are in [v1/config.json](real_text/v1/config.json); the synthetic configuration and run artifacts were left unchanged.

## Measured guarded run

Artifact: [v1-python-docs-pilot](runs/v1-python-docs-pilot/run.json). The [curve](runs/v1-python-docs-pilot/curve.jsonl), [per-step log](runs/v1-python-docs-pilot/steps.jsonl), [checkpoint metadata](runs/v1-python-docs-pilot/checkpoints/step-0040/metadata.json), and [sample generations](runs/v1-python-docs-pilot/generations.json) are preserved.

| Fixed-window metric | Step 0 | Step 20 | Step 30 | Step 40 |
| --- | ---: | ---: | ---: | ---: |
| Train loss, nats/token | 7.0885 | 6.4067 | 6.4636 | **6.2874** |
| Train perplexity | 1,198.1 | 605.9 | 641.4 | **537.8** |
| Validation loss, nats/token | 7.1157 | 6.1480 | **6.1380** | 6.1983 |
| Validation perplexity | 1,231.1 | 467.8 | **463.1** | 491.9 |

Training loss improved by **0.8011 nats/token**. Validation loss is lower than initial by **0.9174**, but rose from step 30 to step 40. This four-document pilot provides only a weak indication that the optimizer can reduce held-out next-token loss; it cannot establish robust generalization. Perplexities are `exp(mean cross-entropy)` on these fixed windows.

- **40/40 steps**, 5,120 training target tokens. Mean step time **7.71 ms** (median **6.21 ms**, maximum **44.82 ms**); aggregate step throughput **16,602 target tokens/s**. The full parent process including launch, evaluation, checkpointing, and generation took **1.53 s**; the worker's measured interval was **0.87 s**. Step throughput excludes evaluation and I/O.
- Peak MLX allocation **44,336,956 bytes** (42.3 MiB). Active MLX allocation at worker end **30,311,480 bytes**. Highest sampled process RSS **61,652,992 bytes**. RSS excludes some GPU/unified-memory accounting; MLX and system measurements should be read together.
- Preflight: macOS arm64, **8 GiB** physical memory, Python **3.14.6**, MLX **0.32.3**, mlx-metal **0.32.3**, psutil **7.1.0**. Immediately before launch: **2,131,968,000 bytes** available, **52%** memory-pressure-free, and **28,038,516,736 bytes** disk free. Allocated swap used **4,617,928,704 bytes**; free space in currently allocated swap files was informational, not treated as a fixed ceiling.
- During the run: lowest sampled available memory **1,692,516,352 bytes**; lowest pressure-free **41%**; observed swap-used growth **0 bytes**; cumulative swap-out counter growth **147,456 bytes**; lowest sampled disk free **28,025,090,048 bytes**. The run was short, so the 0.5-second system sampler could miss brief transients. The existing watchdog still enforced available memory, pressure, swap growth/activity, RSS, disk, and wall-time aborts.
- Two isolated checkpoints were saved at steps 20 and 40. Loading the final model and AdamW state yielded **0.0 maximum absolute logit difference** on a validation-window probe. Checkpoint integrity hashes and provenance were checked.

Example greedy continuations after 40 steps: the training prompt ending “priority queue” produced repeated `inge` fragments; the validation prompt ending “purposes of this module” also produced repeated `ing` fragments. These are failures of language generation at this training scale, despite reduced loss.

## Reproduction and checks

From the repository root:

```bash
python3 research/slm/scratch/real_text/prepare.py --verify
PYTHONPATH=research/slm/scratch research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -v
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/pilot.py
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/pilot.py --execute --run-dir research/slm/scratch/runs/NEW_UNIQUE_RUN_NAME
```

The verification command deterministically rebuilds the tokenizer in memory, reconstructs extracted rows, and checks all corpus hashes without overwriting v1. The write mode uses exclusive output creation and refuses to overwrite v1. For a new corpus version, copy the preparation code, pinned raw sources, and config into a new versioned directory and change its path. The exact v1 environment and run measurements are recorded in [environment.json](real_text/environment.json) and the [run preflight](runs/v1-python-docs-pilot/preflight.json). **20 focused tests passed** in [the test log](checks/v4-real-text-unittest.log). A first broad test discovery failed only because the existing tests require `PYTHONPATH=research/slm/scratch`; rerunning with that path passed all tests.

## Readiness

The data, tokenizer, safety gate, optimizer, checkpoint restore, and measurement path are **verified for this tiny pilot**. Technical prose quality is limited by the hand-selected spans and occasional missing API-heading context; validation is a single document. Tokenizer efficiency drops on held-out text, and greedy generation is repetitive. There is no measured QA performance or evidence that longer training will remain safe under different system load. The next safe experiment is a separately versioned, rights-audited expansion of first-party documentation with more document-disjoint validation pages and a fresh preflight. Do not infer a safe larger model size or launch longer training from this result alone.
