# SLM-A expanded CPython documentation experiment, v2

## Result and scope

The expanded corpus and bounded training run completed. The same four-layer, 128-width, four-head decoder was initialized from random weights with a new train-only tokenizer; its measured size remained **1,082,496 parameters**. There was no checkpoint warm-start, QASPER mixing, Qwen training, production RAG change, or sealed-test model evaluation. The lowest validation loss occurred at **step 60**. Both train and validation loss then rose, and the generated text remained degenerate. This is a reproducible next-token optimization experiment, **not evidence of useful technical prose or QA ability**.

## Source, rights, splits, and quality before training

The source is the official [CPython `v3.12.15` tag](https://github.com/python/cpython/releases/tag/v3.12.15), commit `e848e4b09ca4bc67c0041c9ce7526dae471d4c36` (tagged **2026-09-30 22:04:19 UTC**). The [v2 manifest](real_text/v2/manifest.json) pins every source SHA-256, the source URLs, document text hashes, split file hashes, transformations, and counts. The [rights ledger](real_text/v2/RIGHTS_LEDGER.md) retains the official PSF License Version 2 notice, attribution names found in page headers, required copyright notice, and provenance limits. The original license files are retained. Selected prose excludes code examples, tables, external recipes, figures, medical terms, and quoted third-party specification text. Individual contributor rights were not independently audited.

Documents were assigned to splits **before** tokenizer training and sequence creation. The test split was sealed for model selection: its text was used only for deterministic extraction, aggregate counts, source/rights checks, and automated leakage checks. The training and validation worker never opens `test.sealed.jsonl` or its raw sources; it produces no test predictions or loss.

| Split | Source documents | Retained prose spans | UTF-8 bytes | BPE tokens | Fixed 128-target-token windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | **6** | 48 | 16,079 | **4,543** | **33** |
| Validation | **3** | 23 | 9,333 | **3,788** | **23** (8, 7, 8 by document) |
| Sealed test | **3** | 24 | 6,576 | **2,614** | Not evaluated |

Training domains are algorithms (2 documents, 1,857 tokens), data structures (1, 1,055), iterators (1, 394), filesystem (1, 429), and serialization (1, 808). Validation has functional programming (1, 1,523), data structures (1, 950), and numerical computing (1, 1,315). Sealed test contains diagnostics, text processing, and input/output documents. **The domain mix differs across splits**, so held-out loss also reflects topic shift. This is broader than v1's single validation page but still a small, nonrepresentative corpus.

The [leakage audit](real_text/v2/leakage_audit.json) found **0 cross-split exact paragraph pairs, 0 near-duplicate paragraph pairs, and 0 near-duplicate document pairs**. It uses normalized exact text and word-5-gram Jaccard with a prespecified **0.6** near-duplicate threshold and at least five shared shingles. The maximum cross-split paragraph similarity was **0.0065**. This audit cannot rule out semantic paraphrases or common technical facts. Train/validation [quality samples](real_text/v2/quality_samples.json) show retained prose and source line numbers; sealed test text is absent from that sample file. Automated medical-keyword checks found no hits in selected prose.

## Tokenizer and configuration

A new deterministic byte BPE tokenizer was fitted on the **six training documents only**. Target and actual vocabulary: **1,024** IDs, including four special IDs and 256 byte-fallback IDs. It roundtrips all documents, with **0 unknown tokens by construction**. Measured compression: **3.539 bytes/token** on training text, **2.464** on validation, and **2.516** on sealed test (aggregate statistic only). The lower held-out efficiency is a limitation of the small training vocabulary. The tokenizer artifact lists only training document IDs and their hashes.

The [v2 config](real_text/v2/config.json) retains the v1 architecture dimensions and vocabulary size, with batch size 1, 128-token target windows, AdamW learning rate `0.0005`, weight decay `0.01`, seed `20261012`, evaluation and checkpoints every 10 steps, a hard **80-step** cap, and a **180-second** wall cap. Fresh random initialization was verified by the worker path; no model or optimizer checkpoint was loaded before step 1. Fixed train and validation windows never cross source-document boundaries. The existing memory/swap/disk watchdog remained active.

## Learning curve and checkpoint selection

Full per-step timings/losses are in [steps.jsonl](runs/v2-expanded-cpython/steps.jsonl); fixed-window evaluations are in [curve.jsonl](runs/v2-expanded-cpython/curve.jsonl) and [curve.csv](runs/v2-expanded-cpython/curve.csv). Perplexity is `exp(mean next-token cross-entropy)` over fixed windows. The best checkpoint is the lowest validation-loss checkpoint among the scheduled evaluations; ties select the earlier step.

| Step | Train loss | Validation loss | Validation perplexity |
| ---: | ---: | ---: | ---: |
| 0 | 7.1137 | 7.1274 | 1,245.6 |
| 10 | 6.6233 | 6.5459 | 696.4 |
| 20 | 6.5488 | 6.5248 | 681.8 |
| 30 | 6.4855 | 6.3720 | 585.2 |
| 40 | 6.5678 | 6.4309 | 620.7 |
| 50 | 6.4118 | 6.3974 | 600.3 |
| **60 — best** | **6.3959** | **6.3362** | **564.6** |
| 70 | 6.5057 | 6.5813 | 721.5 |
| **80 — final** | **6.8082** | **6.9608** | **1,054.4** |

Eight isolated checkpoints were saved. The [step-60 best](runs/v2-expanded-cpython/checkpoints/step-0060/metadata.json) and [step-80 final](runs/v2-expanded-cpython/checkpoints/step-0080/metadata.json) checkpoints both restored the model and AdamW state with **0.0 maximum absolute difference** on their saved validation-logit probes. The best model SHA-256 is `52a4e06b02574da144d749e0e2fc9538f0cccb4a0dd008a1f50ac78a50b49bc4`; the final model hash differs as expected.

Between best and final, validation loss rose **0.6246 nats/token**, and train loss rose **0.4123**. The validation increase occurred in **all three documents**: `functools` +0.6051, `collections` +0.7073, and `decimal` +0.5716. The [checkpoint comparison](runs/v2-expanded-cpython/checkpoint_comparison.json) includes a descriptive document bootstrap interval of **+0.5716 to +0.7073**; with only three selected validation documents and validation-based checkpoint selection, it is not a generalization confidence interval. Fixed evaluation windows make this a real within-run deterioration, rather than sampling different windows at each evaluation. Since **training loss also increased**, the pattern is not classic train-improves/validation-worsens overfitting. A plausible cause is unstable optimization or interference from the deterministic sequential minibatch order; this mechanism was **not independently tested**. Small-sample and domain-shift uncertainty remains.

The fixed greedy prompts in [generations.json](runs/v2-expanded-cpython/generations.json) produce repeated `e` fragments from the best checkpoint and repeated `x` characters from the final checkpoint on both train and validation prompts. Neither checkpoint produces coherent technical prose. Do not infer task competence from the loss reduction.

## Resource measurements and safety

The [guarded run](runs/v2-expanded-cpython/run.json) completed **80/80 steps**, processing **10,240 training target tokens**. Aggregate optimizer-step throughput was **15,963 tokens/s**, with mean step time **8.02 ms**, median **7.87 ms**, maximum **47.99 ms**. Parent wall time including launch, evaluations, checkpoints, and generation was **3.05 s**; throughput excludes those overheads.

Preflight passed on an Apple M2 MacBook Air with **8 GiB unified memory**, Python **3.14.6**, MLX **0.32.3**, mlx-metal **0.32.3**, and psutil **7.1.0**. At the last preflight sample, available memory was **1,397,620,736 bytes**, memory-pressure-free was **34%**, disk free was **23,457,501,184 bytes**, and allocated swap used was **5,247,926,272 bytes**. The currently allocated swap-file free figure is informational, not a fixed macOS swap limit.

During execution, peak MLX allocation was **44,336,956 bytes** (42.3 MiB); peak sampled worker RSS was **80,330,752 bytes**. Minimum sampled system-available memory was **1,159,610,368 bytes**, minimum pressure-free **28%**, maximum swap-used growth **68,681,728 bytes**, maximum cumulative swap-out growth **409,600 bytes**, and minimum disk free **23,352,819,712 bytes**. No guard limit was crossed. The parent sampled at 0.5-second intervals and may miss short system transients; MLX peak allocation is tracked separately. These measurements apply to this tiny model and this system load, not future larger runs.

## Reproduction, checks, and next decision

From the repository root, using the existing isolated scratch environment:

```bash
python3 research/slm/scratch/real_text/expanded_prepare.py --verify
PYTHONPATH=research/slm/scratch research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -v
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/expanded_pilot.py
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/expanded_pilot.py --execute --run-dir research/slm/scratch/runs/NEW_UNIQUE_RUN_NAME
```

The verifier reconstructs source spans and tokenizer merges and checks the v2 hashes without overwriting artifacts. The execute command is **opt-in** and refuses an existing run directory. The worker refuses direct execution without the live watchdog. **26 focused scratch tests passed**, including v2 rights, split/leakage, tokenizer isolation, sealed-test access, and both checkpoint restorations; see [test results](checks/v5-expanded-corpus-unittest.log). The complete versioned output checksum ledger is [artifact_hashes.json](runs/v2-expanded-cpython/artifact_hashes.json).

**Decision:** Keep step 60 as the only checkpoint selected by this validation protocol. Do not use the sealed test set to tune the prompt, tokenizer, optimizer, or stopping point. This result supports the correctness of the expanded corpus and guarded training infrastructure, while exposing generation collapse and optimization instability. A subsequent, separately versioned bounded experiment could predeclare a lower learning rate and document-balanced deterministic minibatch order, then rerun the same validation protocol. No further training or model scaling was started here.
