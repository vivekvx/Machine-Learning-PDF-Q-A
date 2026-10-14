# SLM-A CPython corpus v3 and bounded pilot

## Outcome

The v3 corpus and its rights, split, tokenizer, and resource gates passed. One **100-step**, guarded, fresh-weight pilot completed on the unchanged four-layer, width-128, four-head Transformer: **1,082,496 parameters** with a 1,024-ID tokenizer trained from scratch. Fixed-window training and document-balanced validation losses decreased at every scheduled evaluation. The best validation checkpoint under the frozen rule was step 100, which was also the final checkpoint. Greedy continuations still repeat **“the”**; this is next-token optimization progress, not evidence of coherent technical text or academic question answering. No QASPER or sealed test evaluation was performed.

## Source, rights, and corpus construction

The only source is the official CPython [`v3.12.15` tag](https://github.com/python/cpython/releases/tag/v3.12.15), peeled commit `e848e4b09ca4bc67c0041c9ce7526dae471d4c36`, verified with `git ls-remote`. The exact [official source archive](https://codeload.github.com/python/cpython/tar.gz/e848e4b09ca4bc67c0041c9ce7526dae471d4c36) was retrieved on **2026-10-12 UTC**; SHA-256 `cc32fd4a62ecbc76d60df1d238531dada68d6b57423037222c250ca0f7e41cf2`. The retained [root license](real_text/v3/LICENSE) and [documentation license](real_text/v3/Doc_license.rst) identify **Python Software Foundation License Version 2** for software and documentation. The [rights ledger](real_text/v3/RIGHTS_LEDGER.md) records attribution, redistribution conditions, exclusions, and limits; the [source ledger](real_text/v3/source_ledger.json) records exact URLs, source file hashes, header authors, and decisions. No other source repository was added.

The [versioned manifest](real_text/v3/manifest.json) pins source, transformation, tokenizer, split text, license, and file hashes. A conservative extractor retained unindented technical prose from `Doc/library`, `Doc/tutorial`, and `Doc/reference`; it removed code/examples, tables, figures, external links, quotations, medical-keyword paragraphs, explicit rights-notice paragraphs, and detached references to excluded material. Each retained paragraph records original source lines and cleaned-text SHA-256. A deterministic [12-paragraph sample](real_text/v3/quality_samples.json) was inspected from training and validation only. Most samples are coherent technical prose; some remaining RST-derived navigation or context references remain. This inspection is limited and is **not** a comprehensive human rights or quality review.

Of 348 examined official documentation pages, **112** were excluded before splitting: 105 lacked two eligible prose paragraphs, six matched a conservative external-specification/third-party risk list, and one carried an adaptation statement near its header. Of 236 candidates, six duplicate/near-duplicate paragraphs were removed, and two pages then lacked enough prose, leaving **234 documents**. The source ledger preserves the excluded records. The automated screen found no medical-keyword match in selected titles or retained prose. It does not prove that every paragraph has been independently reviewed for subject matter or contributor rights.

Related module-family pages and near duplicates were grouped before assigning whole documents to splits. The [grouping audit](real_text/v3/grouping_audit.json) records **169 final source groups**; the [leakage audit](real_text/v3/leakage_audit.json) found **zero cross-split exact/near paragraph matches and zero near-document matches** at word-5-gram Jaccard ≥0.6 with at least five shared shingles. Lexical checks cannot rule out paraphrases or shared facts. The old v1/v2 sealed material remains unchanged, and v3 has its own sealed final test partition.

| Split | Documents | Source groups | Retained paragraphs | UTF-8 bytes | Unique cleaned-text BPE tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | **193** | 135 | 2,194 | 542,443 | **249,103** |
| Validation | **21** | 17 | 271 | 67,824 | **31,431** |
| Sealed test | **20** | 17 | 259 | 67,805 | **31,283** |

“Unique” here means tokens counted once in retained, deduplicated source text, **not repeated training-window exposures**. The 100k–300k training target was met without padding, synthesis, or repeated counting. A fresh deterministic byte BPE was fitted on the 193 training documents only, with **1,024 actual IDs**, exact UTF-8 roundtrip, and **0 unknown tokens** by byte fallback. Measured bytes/token: train **2.178**, validation **2.158**, test **2.167** (test aggregate statistic only). Token counts are not directly comparable to v2's because the tokenizer changed. The test text was excluded from tokenizer fitting, training, fixed evaluation, generation, and checkpoint selection.

## Frozen pilot protocol and correctness

Before training, [pilot_config.v1.json](real_text/v3/pilot_config.v1.json) and [pilot_plan.v1.json](real_text/v3/pilot_plan.v1.json) froze seed `20261012`, architecture, 128-target-token windows, batch size 1, constant AdamW learning rate **0.0001**, weight decay `0.01`, 100 updates/12,800 target tokens, fixed evaluation windows, 20-step checkpoints, divergence rules, and the 180-second wall cap. The learning rate was a provisional setting from the prior stability audit, **not a proven optimum**. The existing scratch preflight and live memory/swap/disk watchdog thresholds were reused unchanged. Training remained opt-in.

Training sampled **100 distinct windows without replacement** from 1,852 possible train windows, spanning **58 documents and 44 groups**. Thus only **12,800 target-token positions**, about **5.14%** of the unique training-token count, were exposed to updates. Fixed training evaluation used 64 windows from 64 documents; only **two** of those windows were sampled for optimizer updates, so its curve mainly measures *unseen training-document windows*. Fixed validation evaluation used 38 windows from 20 of the 21 validation documents; one short validation document had no full 129-ID window. Validation checkpoint selection used the **equal-document mean** of those fixed windows, while both equal-window and equal-document losses were recorded. The test file was not opened by the worker.

Next-token targets were `ids[1:]` for inputs `ids[:-1]`; the existing causal mask and model dimensions were unchanged. No learning-rate schedule, gradient accumulation, warm start, or pretrained tokenizer/weights were used. All 100 gradient, model, and AdamW-state checks were finite. Step loss in `steps.jsonl` is a changing minibatch value before its update; the comparable curves in `curve.jsonl` and `curve.csv` use exactly frozen windows after scheduled updates. The lowest document-balanced validation loss at the scheduled checkpoints selected the best checkpoint, with earlier-step tie-breaking. Five checkpoints were saved at steps 20, 40, 60, 80, and 100.

| Step | Fixed train loss | Validation window loss | Validation document-balanced loss | Document-balanced perplexity |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7.0551 | 7.0390 | 7.0428 | 1,144.54 |
| 20 | 4.8468 | 4.8410 | 4.8437 | 126.94 |
| 40 | 4.5420 | 4.5297 | 4.5332 | 93.05 |
| 60 | 4.4325 | 4.4219 | 4.4250 | 83.51 |
| 80 | 4.3713 | 4.3682 | 4.3713 | 79.15 |
| **100 — best/final** | **4.3195** | **4.3165** | **4.3207** | **75.24** |

There was **no scheduled fixed-training or document-balanced validation loss reversal** in this one 100-step pilot. This does not establish behavior beyond step 100 or across other seeds/corpora. The gradient L2 range was **1.006–10.324** (maximum at step 1); parameter L2 stayed **65.647–66.042**. The step-100 checkpoint restored all **28 model tensors and 58 AdamW tensors exactly**, including optimizer step 100; validation probe-logit difference and in-memory/freshly loaded probe difference were **0.0**. The [read-only analysis](runs/v4-cpython-v3-pilot/analysis.json) independently reproduced the frozen sample order, curves, selection, and saved checkpoint losses. Greedy train and validation [generations](runs/v4-cpython-v3-pilot/generations.json) both repeat “the,” so the model remains unsuitable for technical QA or useful prose generation. No claim of general language ability or improvement over v2 follows from losses with different tokenizers.

## Hardware and safeguards

The run used Apple M2 with **8 GiB unified memory**, Python **3.14.6**, MLX/MLX-Metal **0.32.3**, and psutil **7.1.0**. The last preflight sample reported **1.970 GB available memory**, **48% pressure-free**, **22.524 GB free disk**, and **5.210 GB used swap**; allocated swap free was informational, not a fixed macOS swap limit. Preflight passed. No live guard fired.

The parent completed in **11.30 s** wall time. Instrumented mean optimizer-step time was **69.10 ms**, with aggregate **1,852 target tokens/s** (excluding evaluation, checkpoint, startup, and generation time). Peak MLX allocation was **54,785,224 bytes** (52.25 MiB); maximum sampled worker RSS **104,890,368 bytes**, minimum sampled system-available memory **1,699,577,856 bytes**, minimum pressure-free **41%**, maximum swap-used growth **0 bytes**, maximum cumulative swap-out increase **3,244,032 bytes**, and minimum sampled disk free **22,456,983,552 bytes**. The parent sampled every 0.5 s, so brief system transients may be missed. These are measurements of this model and system load, not a guarantee for longer jobs.

## Reproduction and next decision

From the repository root, using the isolated scratch environment:

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/v3_prepare.py --verify
research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/tests -p 'test_corpus_v3.py' -v
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/v3_pilot.py
# Explicit opt-in only; choose a new run directory and require a passing preflight:
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/v3_pilot.py --execute --run-dir research/slm/scratch/runs/NEW_UNIQUE_V3_RUN
research/slm/scratch/.venv/bin/python research/slm/scratch/real_text/v3_analyze.py --run-dir research/slm/scratch/runs/NEW_UNIQUE_V3_RUN --output-json /tmp/slma-v3-analysis.json --output-csv /tmp/slma-v3-curve.csv
```

The original `v3_prepare.py --prepare` command requires the exact pinned archive at the manifest path and refuses existing corpus outputs; it is intended for a new isolated copy, not to overwrite this version. Seven focused v3 tests and six v2 regression tests passed. The v1/v2 pilot and optimization ledgers still match their frozen hashes; the v3 [artifact checksum ledger](real_text/v3/artifact_hashes.json) covers this report, source, data, code, tests, logs, and run outputs.

**Decision:** GO for using this v3 corpus and guarded infrastructure as the basis for a separately preregistered next experiment at the **same model size**. NO-GO for claiming technical QA capability, distributing model weights without further rights review, or extending this run automatically. Before any longer training, review more source samples for residual detached references, define a sufficiently broad validation protocol, and specify how a fuller pass through the 249k-token corpus will be assessed without opening the sealed test split.
