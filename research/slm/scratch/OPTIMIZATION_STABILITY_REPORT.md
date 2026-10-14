# SLM-A optimization stability audit — v1

## Scope and provenance

This audit used the unchanged, randomly initialized 1,082,496-parameter SLM-A architecture and the frozen CPython v3.12.15 corpus v2. The training-visible corpus comprises six training documents (4,543 tokens) and three validation documents; the 33 training and 23 validation windows each contain 128 next-token targets. The sealed test set was not opened or evaluated. The prior v1 and v2 artifact ledgers verify 41/41 and 80/80 files unchanged. New code and artifacts are under `optimization/` and `runs/v3-optimization-low-lr/`; the RAG application and QASPER work were untouched.

The lower-rate trial was registered in `optimization/v1/config.json` before execution. Its SHA-256 is `db13ba32df23057bdb078c1fdd591032431e21ca0d6328720083eb6497e0a126`. The only deliberate optimization change was the constant AdamW learning rate, **0.0005 → 0.0001**. The same v2 corpus, train-only tokenizer, architecture, random seed 20261012, 33-window cyclic sample order, 128-token sequence, batch size 1, 80 optimizer steps, weight decay 0.01, and ten-step evaluation/checkpoint schedule were retained. MLX 0.32.3's installed `AdamW` signature defaults to betas `[0.9, 0.999]`, epsilon `1e-8`, and `bias_correction=False`; no learning-rate scheduler is present. The trial adds finite-tensor and norm instrumentation and explicit evaluation mode. A focused test confirmed evaluation mode produces identical logits for this architecture, which has no dropout or other mode-dependent layer. Instrumentation changes timing, so the old and new throughput numbers are **not** a fair speed comparison.

## Measurement and correctness audit

The old `steps.jsonl` field `loss_before_update` is the loss of **one changing minibatch**, measured before that step's optimizer update. Step 60 used a `pathlib` window (6.2762), while step 80 used an `array` window (7.0793); these two numbers are not comparable as a learning curve. Training visits 33 document-grouped windows cyclically, so the final 14 windows receive a third pass. The `curve.jsonl` train and validation losses are evaluated **after** each scheduled update on the same fixed 33 and 23 windows, with an equal-weight mean of 128-target window losses. These values are comparable across steps.

The source uses `ids[:-1]` as input and `ids[1:]` as the target, causal attention masks future positions, and `nn.losses.cross_entropy` computes next-token loss. Tests matched every window to its tokenizer encoding and source-document offset. The frozen step-60 and step-80 model weights and AdamW state passed hash/provenance checks; restored optimizer counters are 60 and 80, and every inspected tensor is finite. Saved probe logits and independently recomputed fixed train/validation losses match the frozen records exactly (difference 0). Different checkpoint weight hashes and optimizer states confirm the parameters changed between these steps. Historical per-step gradient norms were **not recorded**, so a transient high-rate gradient spike cannot be ruled in or out. A focused update test confirmed finite, nonzero gradients and a parameter-changing AdamW update; it cannot retroactively certify every old step.

The exact frozen re-evaluation was:

| Frozen step | Fixed train loss | Fixed validation loss | Validation perplexity |
| ---: | ---: | ---: | ---: |
| 60 | 6.395854 | 6.336194 | 564.64 |
| 80 | 6.808182 | 6.960753 | 1054.43 |

Thus the old train loss rose **0.412328** and validation loss rose **0.624559** on identical fixed windows. This is a confirmed loss reversal, not an artifact of comparing the two minibatches. Because *both* fixed train and validation losses rose, the old step-60→80 change is not by itself evidence of conventional train-improves/validation-worsens overfitting. Plausible contributors include a large constant rate for this small corpus and repeated, document-grouped minibatches. The available old logs do not isolate the cause.

## Controlled lower-rate comparison

| Step | 0.0005 train | 0.0005 val | 0.0001 train | 0.0001 val |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7.1137 | 7.1274 | 7.1137 | 7.1274 |
| 10 | 6.6233 | 6.5459 | 6.7540 | 6.7454 |
| 20 | 6.5488 | 6.5248 | 6.5705 | 6.5675 |
| 30 | 6.4855 | 6.3720 | 6.4429 | 6.4008 |
| 40 | 6.5678 | 6.4309 | 6.3585 | 6.3061 |
| 50 | 6.4118 | 6.3974 | 6.3191 | 6.3079 |
| 60 | 6.3959 | 6.3362 | 6.2701 | 6.2849 |
| 70 | 6.5057 | 6.5813 | 6.2302 | **6.2543** |
| 80 | 6.8082 | 6.9608 | 6.1279 | 6.3935 |

Both runs start with identical fixed-window losses, supporting matched initialization and data. At step 80, the lower-rate trial's validation loss is 0.5673 lower than the control's, and its best observed validation loss (step 70, 6.2543; perplexity 520.27) is 0.0818 lower than the control's best (step 60, 6.3362). But lower-rate validation loss increased **0.1391** from step 70 to 80 while fixed training loss fell **0.1023**. This late divergence is compatible with overfitting, validation heterogeneity, or short-run noise; three validation documents and one paired trial cannot distinguish them. No model-quality or academic-QA claim follows from these token losses.

The 80 lower-rate gradients were finite and nonzero (L2 norm 1.130–3.782); parameter L2 norms stayed 65.647–65.927, and all recorded model/optimizer tensors stayed finite. Scheduled checkpoints were saved at steps 10–80. Full model (28 tensors) and AdamW (58 tensors) restoration for the step-70 best and step-80 final checkpoints matched saved tensors exactly; their optimizer counters, probe logits, and fixed-window losses also matched. Greedy fixed-prompt samples are saved in the new run; they remain repetitive and are not evidence of technical understanding.

## Resource and timing measurements

The isolated scratch environment was Python 3.14.6, MLX/MLX-Metal 0.32.3, and psutil 7.1.0 on an Apple M2 with 8 GiB unified memory. The existing preflight passed immediately before the trial: 1.515 GB system-available memory, 37% pressure-free, 23.095 GB disk free, and 5.971 GB swap used. macOS's 0.472 GB free *allocated* swap was not treated as a fixed swap capacity. The live parent watchdog retained the existing available-memory, pressure, disk, RSS, swap-used growth, swap-out activity, and 180-second wall-time stops. None triggered.

The trial completed 80 steps in 7.64 seconds parent wall time. Instrumented mean step time was 0.0723 seconds; aggregate 1,770 target tokens/s. Peak MLX allocation was 42.68 MB, sampled worker RSS 66.0 MB, minimum sampled system-available memory 1.229 GB, minimum pressure-free 30%, maximum swap-used growth 0, maximum cumulative swap-out increase 3.36 MB, and minimum sampled disk free 23.002 GB. Samples every 0.5 seconds can miss brief system spikes. The historical control recorded 44.34 MB MLX peak and 3.05 seconds wall time, but had substantially less per-step instrumentation; throughput must not be compared as an optimizer effect. The trial checkpoint directory occupies about 99 MB.

## Reproduction and safeguards

From repository root, using the existing isolated environment:

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/optimization/audit.py --output /tmp/slma-frozen-reeval.json
research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -p 'test_optimization_stability.py' -v
research/slm/scratch/.venv/bin/python research/slm/scratch/optimization/trial.py
# Explicit opt-in, only if preflight reports ready; use a new output directory:
research/slm/scratch/.venv/bin/python research/slm/scratch/optimization/trial.py --execute research/slm/scratch/runs/v3-optimization-replication
research/slm/scratch/.venv/bin/python research/slm/scratch/optimization/analyze.py --run-dir research/slm/scratch/runs/v3-optimization-replication --output-json /tmp/slma-replication-analysis.json --output-csv /tmp/slma-replication-curve.csv
```

The trial refuses an existing run directory, enforces the registered 80-step/180-second caps, checks the frozen corpus hashes and training/validation windows, and does not open the sealed test source or answers. A failed preflight writes a refusal record without starting the worker. The parent monitors live resources and terminates the worker on a guard violation. The versioned fixed-window comparison, complete 80-step norm/timing log, checkpoints, resource snapshots, and new artifact hashes are in `optimization/v1/` and `runs/v3-optimization-low-lr/`. Five focused stability tests and six existing expanded-corpus tests passed.

## Recommendation

**NO-GO for extended larger-corpus pretraining now.** The old reversal is real, the lower rate reduces it in this single paired run, and the late lower-rate validation rise remains unresolved. The next safe step is to prepare a larger, explicitly licensed non-medical technical corpus with multiple independent validation documents, then preregister a short guarded pilot at the same model size and an independently selected learning-rate schedule. Repeat seeds or document splits before claiming a robust optimizer improvement. Keep the sealed test split untouched until the protocol is frozen.
