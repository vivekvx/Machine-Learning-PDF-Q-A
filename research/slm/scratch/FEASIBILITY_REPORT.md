# SLM-A feasibility report — 2026-10-12

**Update:** The later [memory-guard audit](MEMORY_GUARD_AUDIT.md) corrects the interpretation of allocated swap headroom and records a completed guarded five-step smoke test. The refusal measurements below remain as historical records.

## Scope and verified construction

The scratch workspace contains a newly initialized decoder-only MLX Transformer, a pure-Python byte BPE trained only on synthetic training text, bounded training and greedy generation entrypoints, tests, and hash-checked artifacts. It imports no pretrained tokenizer or weights and makes no model download or API call. The existing RAG application, QASPER research, Qwen baseline, and Track B were not changed.

The configured architecture is four layers, width 128, four attention heads, 256-token context, 2,000-token actual vocabulary, and batch size one. Direct tensor inspection during implementation returned **1,332,352 trainable parameters**; the architecture formula independently gives the same count. The final gated parameter-count test remains pending while the resource preflight fails. Synthetic data has 96 training and 24 validation records, zero overlapping IDs or text, 1,068 and 636 encoded tokens respectively, and four and two full 256-token next-token windows. A clean rebuild matched the committed data and tokenizer byte for byte.

The isolated environment uses macOS 26.6.2 on arm64, Python 3.14.6, `mlx==0.32.3`, `mlx-metal==0.32.3`, and `psutil==7.1.0`. `data/v1/manifest.json` contains individual SHA-256 hashes and the configuration hash. The source/environment manifest is `reproducibility.v1.json`. Each future checkpoint will store model/optimizer hashes and source/data provenance.

## Actual resource gate and benchmark status

The saved `runs/v1-preflight-refusal/run.json` records **1,537,179,648 bytes available RAM, 517,079,040 bytes free swap, 36% free memory pressure, and 23,284,449,280 bytes disk headroom**. Free swap was below the fixed 750,000,000-byte preflight floor, so the watchdog refused to launch. No training worker or generation process was started. This threshold was retained from the existing small-Mac safety approach and was not relaxed.

A second saved check after the user's resource update, `runs/v1-preflight-refusal-2/run.json`, measured **617,807,872 bytes free swap**, 1,868,300,288 bytes available RAM, 45% free memory pressure, and 22,622,511,104 bytes disk headroom. Swap was still 132,192,128 bytes below the unchanged floor; again no worker launched.

| Measurement | Actual result |
| --- | --- |
| Five-step smoke steps | 0; preflight refused |
| Controlled overfit steps | 0; requires completed smoke checkpoint |
| Training or validation loss | Not measured |
| Step timings and tokens/sec | Not measured |
| Peak MLX training allocation | Not measured |
| Training swap growth | Not measured; no training process started |
| Checkpoint/resume at configured scale | Not verified; no training checkpoint |
| Generated text | Not measured |

Four pure data/safety tests passed: reproducible rebuilding, split isolation, tokenizer round trip including Unicode, and resource threshold logic. Three MLX execution tests for tensor count, causal prefix invariance, and gradient/checkpoint reproducibility were skipped by the same preflight gate. Syntax compilation passed. These test outcomes verify the data pipeline and guard behavior, **not successful training**.

## Recommendation

**NO-GO for a larger model or prolonged training now.** The next safe size remains the 1.33M-parameter v1 configuration. Once the Mac has at least 750 MB free swap plus the other configured preflight margins, run the gated MLX tests, then the five-step smoke test. Only if it completes without guard stops should the capped single-window overfit stage run. The 6.90M and 20.50M candidates in `ARCHITECTURE.md` are untested plans, not memory predictions. Synthetic loss reduction would establish implementation feasibility only; meaningful academic QA would require a separately licensed, curated corpus and an independent evaluation design.
