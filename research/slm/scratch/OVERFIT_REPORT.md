# SLM-A six-sequence synthetic overfit report — 2026-10-12

## Decision

**Yes: the randomly initialized 1,332,352-parameter Transformer demonstrably learned the six synthetic training sequences.** Full-set next-token loss fell from **7.7397 to 0.1131** (98.54% lower), and greedy token continuations from fixed mid-sequence prefixes reproduced **all six exact remaining token sequences** by step 175 and again at step 200. This is a memorization result on training text, not evidence of language understanding, held-out generalization, or academic QA ability.

The training loop, live memory watchdog, intermediate safetensors checkpoints, and full-model restore check worked in this bounded experiment. That supports a **GO for designing a small licensed technical-text pilot**, conditional on source-level rights and split controls. It does not authorize unbounded training or a larger architecture.

## Correctness audit of the earlier five-step run

The prior smoke run used four 256-token training windows, cycling through them at steps 1–5. Its per-step pre-update losses (7.668, 7.256, 7.163, 7.295, 9.419) were measured on different windows until step 5 returned to window 1. The first window's loss rose from 7.668 before any update to 7.899 after five steps. Five updates were too few to infer learnability, and the step sequence is not a same-sample learning curve. The new fixed six-sequence full-set evaluation provides that curve.

The model's attention adds an upper-triangular future-token mask before softmax; the causal-prefix invariance test passes. For each synthetic row, training inputs are `[BOS] + text`, excluding the final `EOS`; targets are exactly the same token list shifted one position, ending in `EOS`. No padding or missing-label positions enter the loss. First-step L1 gradient sums were nonzero for token embeddings (41.01), attention (1,225.27), feed-forward layers (2,388.94), and output head (181.31). The first optimizer update changed token-embedding weights by L1 sum 19.78. Focused tests also verify target shifting, causal masking, finite gradients, and optimizer/checkpoint behavior. These checks found no label, masking, or gradient-flow defect.

## Fixture and unchanged architecture

`overfit/v1/data.jsonl` has **six unique, deterministic synthetic sentences**, with 25–28 tokens each after adding BOS/EOS and **151 next-token targets** total. It contains no QASPER, academic QA, medical, or external text. The fixture manifest records its hash and the unchanged tokenizer/model-config hashes. The existing byte BPE was learned from the earlier synthetic training split; no pretrained tokenizer or weights were used.

The base `config.v1.json`, model source, tokenizer, five-step run, and prior reproducibility files were preserved. The experiment used the same model seed (`20261012`), architecture (4 layers, width 128, 4 heads, context 256), batch size 1, and AdamW hyperparameters (**learning rate 0.002; weight decay 0.01**). No hyperparameter was changed after observing results. The new experiment-only caps were **200 optimizer steps**, **120 seconds**, full-set loss every 10 steps, and checkpoints every 25 steps. Early stop required both full-set loss ≤0.10 and 6/6 exact continuations; loss remained 0.1131 at step 200, so the run ended at the hard step cap.
The pinned environment was macOS 26.6.2 on arm64, Python 3.14.6, MLX/MLX Metal 0.32.3, and psutil 7.1.0.

## Training curve and reproduction

The complete, reproducible curve is `runs/v1-six-sequence-overfit/curve.jsonl` and `curve.csv`; all 200 individual step records are in `steps.jsonl`.
Full-set loss was recomputed at least every ten steps. Exact continuation counts were recomputed at checkpoint steps; intervening curve rows carry the last count and mark `continuations_evaluated: false`.

| Step | Full-set next-token loss | Exact continuations |
| ---: | ---: | ---: |
| 0 | 7.7397 | 0/6 |
| 25 | 1.6828 | 0/6 |
| 50 | 1.1183 | 0/6 |
| 100 | 0.5543 | 1/6 |
| 150 | 0.2527 | 4/6 |
| 175 | 0.1343 | 6/6 |
| 200 | 0.1131 | 6/6 |

The reproduction test starts from token-aligned prefixes near each sequence's midpoint. A BPE token boundary can fall inside a printed word. For example, from `Record B: The violet sensor moves ` the model produced `south. The signal is green.\n` exactly, including the EOS target; the other five exact token continuations are saved in `final_continuations.json`. These are **training-sequence** continuations, not held-out predictions.

Eight intermediate checkpoints were saved at steps 25, 50, 75, 100, 125, 150, 175, and 200. Reloading the final model and AdamW tensors gave a maximum absolute logit difference of **0.0** on the six checked inputs and identical greedy token continuations. Checkpoint file hashes and the versioned run/source hashes are recorded in `provenance.json` and verified.

## Time and resource measurements

The corrected macOS preflight passed, and the separate parent process applied the same live abort checks as the five-step run. No abort threshold fired.

| Measurement | Actual value |
| --- | ---: |
| Completed steps / training tokens | 200 / 5,032 |
| Parent wall time / worker wall time | 3.557 / 3.173 s |
| Sum of measured optimizer-step times | 1.433 s |
| Step duration minimum / median / maximum | 0.0049 / 0.0066 / 0.0408 s |
| Aggregate tokens per measured optimizer-step second | 3,511 tokens/s |
| Peak MLX active allocation | 43,836,796 bytes |
| Peak sampled worker RSS | 68,255,744 bytes |
| Minimum sampled available system RAM | 1,701,707,776 bytes |
| Minimum reported free memory-pressure percentage | 42% |
| Swap-used growth | 0 bytes |
| Maximum swap-out counter increase | 917,504 bytes |
| Minimum sampled disk headroom | 29,545,304,064 bytes |

Step throughput excludes full-set evaluations, continuation generation, checkpoint I/O, and parent monitoring; wall time includes them. MLX active allocation excludes cached Metal buffers and is not the machine's total memory footprint. Resource samples are system-wide except worker RSS and MLX metrics; background applications can affect them.

All **12 focused tests** passed, including byte-reproducible fixture construction, source/hash/cap checks, next-token targets and EOS, gradient flow, causal masking, checkpoint reproducibility, and curve export. The test log is `checks/v3-overfit-unittest.log`. The fixture, code, eight checkpoints, logs, and prior-artifact hashes are frozen in `runs/v1-six-sequence-overfit/provenance.json` (18 source and 34 run files; no hash mismatches).

## Next safe experiment, not started

A candidate is a small prose-only extract from the **Python 3.12.15 standard-library documentation**, for example selected `itertools` and `pathlib` sections. The official [Python license page](https://docs.python.org/3.12/license.html) states that Python software and documentation are under the PSF License Version 2, with retention of its copyright/license notice; incorporated material may have separate licenses. Before any download or training, pin exact page URLs and release, inspect page-level rights and copied material, retain attribution, select only non-medical technical prose, and make document-disjoint train/validation sections. Then train a fresh tokenizer on training text only and run a small guarded pilot with a held-out validation loss. No such corpus was fetched or used in this experiment.

**Readiness:** the scratch optimizer, target shifting, memory guard, and checkpoint restoration passed a synthetic memorization probe. Real-corpus readiness still requires verified rights, data cleaning, document-level split design, tokenizer-vocabulary assessment, and a held-out evaluation. No claim of QA performance or ability to train for long periods follows from this short run.
The run is fully specified and hashed, but independent-run bitwise reproducibility was not tested by launching a second 200-step job.
