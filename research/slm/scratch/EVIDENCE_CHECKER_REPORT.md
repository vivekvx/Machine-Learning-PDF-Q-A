# Evidence checker v2: candidate classification improved; combined QA remains NO-GO

**Decision, 14 October 2026:** The original 944,195-parameter pointer reader, 960-token train-only tokenizer, selected weights, and answerable-only demo remain frozen. A new checker passed class gates on *supplied labeled candidate spans*, but **failed on spans actually selected by the pointer**. The combined system is **not approved**, and the new sealed test was **not opened**. [`evidence_checker/v2/combined_gate.json`](evidence_checker/v2/combined_gate.json) is the authoritative decision; the v2 `selection.json` records only the gold-candidate pass and must not authorize test evaluation.

## Read-only diagnosis before training

The [frozen v1 checker scores](pointer_qa/answerability/v1/run/validation_predictions.jsonl) were swept at every score boundary, without changing weights. No threshold achieved both preregistered 70% class gates on calibration, assessment, or all validation records. The reproducible sweep, individual confusion matrices for 0.30–0.80, AUROC, input hashes, and best possible maximin thresholds are in [`evidence_checker/v1_threshold_audit.json`](evidence_checker/v1_threshold_audit.json).

| Frozen v1 partition | AUROC | Best possible minimum class accuracy | Any threshold passes both? |
| --- | ---: | ---: | --- |
| 20 calibration documents | 0.674 | 62.5% | No |
| 40 assessment documents | 0.714 | 69.4% | No |
| All 60 validation documents | 0.700 | 66.7% | No |

At the originally selected **0.70** threshold, the 40-document assessment confusion matrix was **TP 129, FN 31, TN 80, FP 80**: 80.6% supported acceptance, 50.0% unsupported rejection, and 65.3% balanced accuracy. Category AUROC on all validation records was 0.932 for wrong entity, **0.506 for wrong field**, **0.586 for missing fact**, and 0.764 for contradiction. Thus threshold calibration alone could not fix v1. The 900-dimensional pooled feature head also had worsening validation BCE in the prior report. A read-only frozen-pointer feature check found much stronger candidate-margin separation (overall AUROC approximately 0.953 on those same v1 records), motivating a compact candidate-specific calibrator. No target, offset, gradient, checkpoint, or source-label defect was confirmed by focused tests; the old data supplied only one negative per source question and tied negative type to question index, which limited hard-negative exposure.

## Preregistered v2 checker experiment

The pointer model and tokenizer were loaded from their exact prior SHA-256-bound checkpoint. **No pointer parameter was updated.** The separate MLX checker is a **9-parameter learned logistic head** over eight source-candidate score features: candidate score, best competing score, margin, candidate probability, top probability, score entropy, candidate score relative to mean, and score spread. It uses no answer text, label, candidate position, template category, or rule-produced support decision as an input. It is a trained calibrator over a trained neural reader, not an independent semantic language model.

The deterministic v2 synthetic corpus contains **7,200 training records across 300 documents** and **1,440 validation records across 60 documents**. Each source question contributes a supported span, wrong-entity span, wrong-field span, current corrected value, and obsolete conflicting value; each source document also contributes absent-fact questions with plausible candidate values. There are 2,400 supported and 4,800 unsupported train records. All candidate offsets, support labels, unique IDs, train/validation document separation, and exact candidate-context non-overlap were checked. Original synthetic technical text has no third-party license dependency. The new sealed test was not read during construction or tuning.

Before execution, [`config.json`](evidence_checker/v2/config.json) and [`plan.json`](evidence_checker/v2/plan.json) fixed seed 20261015, a **1,200-step maximum**, batch size 1, AdamW learning rate 0.01, 120-second wall cap, 150-step checkpoint interval, and train-only feature scaling. The actual sampled order balanced **600 supported and 600 unsupported** examples, stratified over all six categories. The 60 validation documents were separated into 20 for threshold calibration, 20 for checkpoint selection, and a further 20 for the final gate. The last 20 labels were not used for the new head's gradient updates, threshold, or checkpoint choice. This validation corpus had been inspected in the earlier v1 checker work, so it is not a pristine study-wide test set.

The run stopped by its registered patience rule after **1,050 optimizer steps**, selecting the step-450 checkpoint and calibration threshold **0.68**. Checkpoint reload reproduced all two parameter tensors, optimizer step, threshold, and development BCE exactly. The pointer checkpoint SHA-256 stayed `75d4ca2edaa7ea243017e58d8950e7ad2d1ed0942a695d3c5ca156f19c9e2c3d`.

## Final validation, separate conditions

| Reserved 20-document condition | Supported acceptance | Unsupported rejection | AUROC | Confusion (TP/FN/TN/FP) |
| --- | ---: | ---: | ---: | --- |
| Supplied labeled candidate spans, 480 records | **136/160 = 85.0%** | **278/320 = 86.9%** | 0.931 | 136 / 24 / 278 / 42 |
| Actual frozen-pointer selected spans, 200 scenarios | **127/137 = 92.7%** | **40/63 = 63.5%** | 0.852 | 127 / 10 / 40 / 23 |

The first row passes 70%/70%; the second **fails the unsupported rejection gate**. Among 63 selected unsupported spans, **23 were answered** (36.5% unsupported-answer rate). Error grouping attributes **14/23** false answers to obsolete conflicting values, 5 to ordinary wrong spans, and 4 to absent facts. In one representative held-out validation record, the passage explicitly says the earlier owner `team iris` is obsolete and the current owner is `team kelp`; the frozen pointer selected `team iris`, and the checker accepted it at probability **0.924**. This shows the score-only features cannot reliably resolve cross-sentence supersession. Supplied obsolete-value candidates were rejected only **50/80 = 62.5%**, below the other negative categories.

On the 80 ordinary answerable questions in this gate partition, the pointer alone selected **72/80 = 90.0%** correct spans. The combined system returned **66/80 = 82.5%** exact answers and answered **71/80 = 88.8%** of questions. Its current-value exact match on 80 correction questions was **61/80 = 76.3%**. It abstained correctly on **36/40 = 90.0%** absent-fact scenarios, but overall coverage was **150/200 = 75.0%**. These are validation measurements, not sealed-test metrics. The combined system both produces too many unsupported answers and suppresses some correct pointer answers; the answer selector itself remains unchanged.

## Safety, integrity, and next boundary

The unchanged Mac memory preflight passed. The guarded run took **24.59 seconds**, peaked at **17.76 MB MLX allocation** and **140.18 MB worker RSS**, with minimum **1.712 GB system-available memory**, minimum **41% free memory pressure**, **0 MB maximum swap-used growth**, **1.31 MB system-wide swap-out counter growth**, and at least **56.99 GB free disk**. No resource guard fired. Swap-out activity is system-wide and cannot be attributed entirely to the worker.

Nine new focused tests passed for split/offset and label integrity, deterministic balanced sampling, question-specific candidate scores, gradients and parameter updates, threshold accounting, resource guards, frozen pointer and checkpoint hashes, and the combined NO-GO decision. The prior pointer ledger still verifies **117** files and the earlier QA ledger **70** files. New run curves, step records, predictions, resource snapshots, source hashes, and checkpoints live under [`evidence_checker/v2/`](evidence_checker/v2/). The original answerable-only demo is unchanged. No test run, app integration, Qwen training, or external model/API use occurred.

**Blocker:** a score-only checker cannot reliably tell that a highly ranked candidate was superseded by later evidence. A future experiment would need a separately preregistered contextual checker that compares the candidate to the full passage and explicitly models corrections, then passes both gates on actual pointer-selected candidates. This run is stopped at the validation boundary. Do not infer approval from the gold-only `selection.json`; only a passing combined gate can permit opening the sealed test once.

### Reproduction and verification

Run from the repository root with the existing `research/slm/scratch/.venv`:

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/evidence_checker/v2/prepare.py --verify
research/slm/scratch/.venv/bin/python -m unittest research/slm/scratch/evidence_checker/v2/test_v2.py research/slm/scratch/evidence_checker/test_gate_artifacts.py -v
research/slm/scratch/.venv/bin/python research/slm/scratch/evidence_checker/hash_artifacts.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/hash_artifacts.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/hash_artifacts.py --verify
```

The frozen build and training entrypoints reject overwrite. To replay the experiment, use a new versioned workspace with the same pinned inputs and explicitly run `prepare.py --build`, `train.py --plan`, and `train.py --execute`; each hash and the fixed order must be retained. The current files are an immutable record of this bounded attempt.
