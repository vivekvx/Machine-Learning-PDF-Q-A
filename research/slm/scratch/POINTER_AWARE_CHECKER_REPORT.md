# Pointer-aware evidence checker: numeric pass, scientific NO-GO

**Decision (14 October 2026):** The single bounded v1 experiment met its preregistered numeric validation thresholds on fresh synthetic documents, but its engineered features reproduce **every** generated label with a two-condition Boolean rule. This is a decisive label-shortcut finding. The learned 14-parameter head cannot be credited with independently learning evidence sufficiency. The combined pipeline is **not approved**, the sealed test is **unopened**, and the answerable-only pointer demo remains unchanged. The authoritative decision is [`pointer_aware_checker/v1/decision.json`](pointer_aware_checker/v1/decision.json).

## Diagnosis before the new training run

The frozen 944,195-parameter pointer, its tokenizer, and the previous checker were not changed. The v2 checker scores each candidate from a question paired with that candidate's **single source sentence**. It has no view of a different sentence saying an earlier value is obsolete. On the previously evaluated 200 pointer-selected validation scenarios, supported candidates had median probability **0.946**, missing-fact candidates **0.362**, and obsolete values **0.929**. The 15 obsolete scores substantially overlapped the 65 current-value scores (median **0.908**). Wrong-entity and wrong-field selected candidates numbered only four each, so their distribution estimates are weak. The frozen 0.68 threshold gave TP/FN/TN/FP **127/10/40/23** and AUROC **0.852**. A read-only sweep found that 0.80 would give **112/137 = 81.8%** supported acceptance and **46/63 = 73.0%** unsupported rejection, but ordinary exact answers would fall from **66/80 to 63/80** and only **4/15** obsolete values would be rejected. The best maximin threshold, 0.83, likewise retained only **62/80** ordinary exact answers. These thresholds were inspected on an already evaluated gate and were **not** promoted to a new validated selection. Full distributions, confusion matrices at every 0.01 threshold, input hashes, and AUROC are in [`readonly_diagnosis.json`](pointer_aware_checker/v1/readonly_diagnosis.json).

## One preregistered checker experiment

Before execution, [`config.json`](pointer_aware_checker/v1/config.json) and [`plan.json`](pointer_aware_checker/v1/plan.json) froze seed **20261016**, **800 maximum optimizer steps**, batch size 1, AdamW learning rate **0.01**, a **180-second** worker and parent wall cap, 100-step checkpoints, early stopping, 0.01 calibration grid, source hashes, split IDs, and memory safeguards. This was the only new checker training run. The frozen pointer checkpoint SHA-256 remained `75d4ca2edaa7ea243017e58d8950e7ad2d1ed0942a695d3c5ca156f19c9e2c3d`.

The new, original synthetic corpus has **80 training documents** and **20 each** for threshold calibration, checkpoint selection, and final validation. Its **1,120/280/280/280** scenarios respectively include original questions, matching corrections, corrections to an unrelated fact, and missing-fact questions. Every source document is new relative to the pointer's frozen training and validation documents. Documents, exact question-context strings, and IDs are disjoint across the four new partitions. The checker trains on frozen-pointer-scored candidates from these new documents, including the spans the pointer actually chose. The same value has both supported and unsupported labels under changed question or correction contexts: **632** document-candidate value pairs in training and **159** in the final validation partition. No sealed test predictions or answers were used.

The new checker is a **14-parameter learned logistic head** over eight frozen pointer score features plus five explicit question-relative features. The latter parse an entity and field from the **full question**, inspect the candidate sentence, and look across the **full passage** for a correction to that same entity and field. It does not train or rerank the pointer, use source fact annotations as inputs, or substitute an extractor's answer. A candidate is accepted or rejected at the development-selected threshold **0.49**; its copied span remains the frozen pointer's choice. This feature design is limited to the controlled synthetic vocabulary and correction syntax. The correct answer can still be lost when the pointer chooses a wrong span.

The guarded worker completed **800** steps, chose the **step-700** checkpoint from the separate selection documents, and reproduced its tensors, optimizer step, threshold, and development metrics after reload. It used training-only feature scaling. The final 20-document gate was evaluated once after checkpoint and threshold selection. New and old checkers were then compared on identical candidate IDs, with the old checkpoint and its original 0.68 threshold unchanged. Those comparisons are in [`comparison.json`](pointer_aware_checker/v1/run/comparison.json); individual scores and predictions are saved in `run/`.

| Fresh 20-document final validation, actual pointer selections | Pointer only | Old checker | New checker |
| --- | ---: | ---: | ---: |
| Supported acceptance (207 selected supported) | 207/207 by always answering | 185/207 = 89.4% | **204/207 = 98.6%** |
| Unsupported rejection (73 selected unsupported) | 0/73 | 46/73 = 63.0% | **73/73 = 100%** |
| Ordinary answerable exact match (80) | **73/80 = 91.3%** | 66/80 = 82.5% | **72/80 = 90.0%** |
| Missing-fact abstention (40) | 0/40 | 35/40 = 87.5% | **40/40 = 100%** |
| Obsolete-value rejection (16 selected obsolete) | 0/16 | 0/16 | **16/16** |
| Unsupported-answer rate | 73/73 | 27/73 = 37.0% | **0/73** |
| Answer coverage across 280 scenarios | 280/280 | 212/280 = 75.7% | **204/280 = 72.9%** |

The new checker accepted **72/73 = 98.6%** of the pointer's correct ordinary answers; it lost one. It accepted **64/64** correctly selected current correction values, versus **56/64** for the old checker. On all **1,280** supplied candidate spans in the final validation partition, it accepted **231/240** supported spans and rejected **1,040/1,040** unsupported spans. These are synthetic-template measurements, not PDF or QASPER performance.

## Shortcut audit overrides the numeric pass

The independent read-only audit found that `joint_question_entity_field_match AND NOT superseded_candidate_for_question` equals the generated support label for **5,120/5,120** training candidates, **1,280/1,280** calibration candidates, **1,280/1,280** selection candidates, and **1,280/1,280** final validation candidates. It also labels **all 280** pointer-selected final cases correctly without the learned head. The feature constructor encodes most of the synthetic generator's decision rule. This does **not** mean answers were silently replaced by a rule; the reported model predictions came from the trained head. It means their success does not isolate learning from hard-coded task knowledge. The shortcut was discovered after the preregistered run and is recorded separately in [`shortcut_audit.json`](pointer_aware_checker/v1/run/shortcut_audit.json). Consequently, numeric gate success does not satisfy the request to prevent label shortcuts. Opening the sealed test would yield a misleading measure of learned evidence reasoning, so it remains closed. No combined demo was published.

## Resources, integrity, and reproducibility

The unchanged macOS/MLX preflight passed. On the Apple M2 MacBook Air with **8 GiB** memory, Python **3.14.6**, MLX/MLX Metal **0.32.3**, and psutil **7.1.0**, the run took **24.61 s** wall time. Worker mean optimizer-step time was **1.05 ms** after feature extraction; peak MLX allocation was **18.47 MB** and peak worker RSS **77.81 MB**. Minimum system-available memory was **1.40 GB**, minimum free-memory pressure **34%**, maximum swap-used growth **0 MB**, system-wide swap-out counter growth **5.73 MB**, and minimum free disk **56.98 GB**. No guard fired. Swap-out counters are system-wide, not attributable solely to this worker.

Seven focused tests passed for split and offset integrity, contextual label flips, unrelated corrections, balanced authentic-pointer sample order, finite gradients, exact checkpoint restoration, and the scientific NO-GO decision. The previous pointer, checker, and earlier QA ledgers still verified **117**, **46**, and **70** files respectively. All new code, data, scores, checkpoints, resource snapshots, and hashes are under [`pointer_aware_checker/v1/`](pointer_aware_checker/v1/); old frozen trees and the FastAPI app were not edited.

Reproduce the read-only checks from the repository root:

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_aware_checker/v1/prepare.py --verify
research/slm/scratch/.venv/bin/python -m unittest research/slm/scratch/pointer_aware_checker/v1/test_v1.py -v
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/hash_artifacts.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/evidence_checker/hash_artifacts.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/hash_artifacts.py --verify
```

The versioned `--build`, `--plan`, and `--execute` entrypoints refuse to overwrite this completed experiment. A fresh version must be preregistered for any future run. The next scientifically useful step is an independently designed challenge with less predictable evidence wording and labels not recoverable from the input feature parser, while keeping the pointer frozen and the current sealed test closed. Do not infer unrestricted PDF QA readiness from this bounded synthetic result.
