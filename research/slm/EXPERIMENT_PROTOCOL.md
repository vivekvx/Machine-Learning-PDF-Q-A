# Evidence-sufficiency-aware grounded QA: frozen experiment protocol

## Research questions and scope

1. At matched compute and data exposure, does an evidence-sufficiency-aware LoRA improve grounded answer accuracy, citation validity, and abstention over the untuned Qwen3-0.6B and ordinary LoRA?
2. Can the resulting quantized inference model answer document questions within an 8 GB Apple M2 memory budget, even if controlled fine-tuning runs on a larger GPU?

This protocol is a design, not a fine-tuning result or a novelty claim. QASPER supplies source documents and human QA annotations; the current 352/80 train/validation pilot is intentionally extractive, short-context, and one-document-per-question. Keep QASPER's official document-disjoint train/validation/test splits. Never use held-out test answers for data design, prompt tuning, early stopping, retrieval selection, or manual error analysis before the final locked evaluation. The dataset card's CC BY 4.0 license does not establish the redistribution rights of every underlying S2ORC paper.

## Fixed experimental inputs

- Base model: the pinned untuned `mlx-community/Qwen3-0.6B-4bit` conversion for local inference; controlled cloud training must pin and verify its upstream `Qwen/Qwen3-0.6B` weights, tokenizer, and conversion lineage. Do not treat MLX and PEFT adapters as interchangeable.
- Training source: only included non-medical QASPER **train** documents. Validation is the existing fixed 80 question IDs from 80 disjoint documents (40 answerable, 40 unanswerable). Freeze the IDs, exact evidence passages, references, prompts, tokenizer, and retrieval indices/configuration before training.
- Retrieval conditions: oracle source-matched paragraph (a diagnostic upper-context condition), per-document TF-IDF+FAISS, and per-document MiniLM+FAISS. The latter two use identical source paragraphs, top 3, 512-token sequence budget, 64-token generation reserve, and exact section/paragraph evidence matching. Oracle context is not a deployable retrieval result.
- All answers come from the provided passages. For an unsupported question, the target is exactly `UNANSWERABLE`. For supported answers, require a real `[section:i:paragraph:j]` citation. Do not invent PDF page numbers.

## Evidence conditions

| Condition | Operational definition | Training target for an otherwise answerable question |
| --- | --- | --- |
| Sufficient | Every fact needed for the reference answer is present in the packed source passages; the answer and citation can be checked against source text. | Grounded answer and source ID |
| Partial | Some relevant facts remain, but an indispensable answer-bearing fact or relation is absent. Requires a documented removal and independent support review; a mere shortened passage is not assumed insufficient. | `UNANSWERABLE` for the **provided context** |
| Irrelevant | Retrieved passages come from the permitted source document but do not support the answer; exclude accidental answer leakage by audit. | `UNANSWERABLE` |
| Absent | No evidence passage is supplied. | `UNANSWERABLE` |

QASPER's globally unanswerable label is distinct from context insufficiency. Record both the original QASPER label and the constructed context-sufficiency label; preserve every original annotation. The current single-paragraph extractive pilot cannot automatically yield trustworthy partial examples. Quarantine any proposed partial/irrelevant example whose support status is uncertain. Include realistic rank and position variation, including answer-bearing passages near the beginning, middle, and end of a packed context; avoid using validation references to choose placement or distractors.

## Controlled model arms

1. **Untuned**: the pinned base model, no adapter, fixed greedy generation.
2. **Standard LoRA**: same train question IDs with sufficient evidence only, assistant-only loss, same answer/citation format.
3. **RAFT-style distractor LoRA**: same train questions and verified answer-bearing evidence mixed with source-document distractors; supervise the concise answer and citation, not chain-of-thought. Include controlled answerability examples.
4. **Evidence-sufficiency-aware LoRA**: same train questions and exposures across the four verified conditions; supervise a grounded answer only when the *provided* context is sufficient and exact abstention otherwise. Record condition labels and document IDs.

Before training, construct all arms from the same permitted train-document pool. Match total optimizer steps, supervised assistant tokens (or report unavoidable differences), token budget, effective batch size, LoRA rank/target modules, optimizer/schedule, quantization, seed list, and number of document/question exposures. To isolate the effect of condition labels, include a matched distractor arm with the same context lengths and exposure counts. Use at least three prespecified seeds if resources permit; report every run, including failures. Tune only on the fixed validation split and prespecify selection metric and tie breaking. No extended training should begin until cloud preflight, source-rights review, evidence audit, and frozen configurations pass.

## Evaluation and analysis

- Run all four arms on the same 80 validation question IDs and each frozen retrieval condition. Keep gold-evidence and retrieved-evidence results separate. Score answerable exact match/token F1 and unanswerable exact-abstention precision/recall separately. Report citation-source validity, invented-reference rate, per-question retrieval Recall@1/3, MRR@3, packed evidence coverage, generation truncation/format errors, answer latency, process/accelerator peak memory, and swap pressure.
- Judge evidence support on a deterministic, blinded manual sample stratified by answerability, retrieval condition, evidence sufficiency, and model arm. Distinguish lexical overlap from semantic entailment. Report reviewer rubric, disagreements, and adjudication; do not claim lexical substring checks are support judgments.
- Add stress slices for partial/irrelevant/absent context and evidence position only after independent evidence-label review. Compare same question/context pair across model arms; cluster uncertainty intervals by document rather than treating multiple questions from one paper as independent.
- Register the final model-selection rule and freeze all scripts, hashes, prompts, retrieval indices, and evaluation metrics. Then run the held-out **test** split once, with no subsequent tuning. If changes are necessary after seeing test outcomes, label them exploratory and use a new held-out cohort for confirmation. Preserve test answer isolation up to that final step.
- For resource efficiency, benchmark the final quantized inference artifact on the 8 GB M2 with the same memory/swap guards, batch 1, measured latency and task utility. A cloud-trained adapter requires a validated export/quantization path and a fresh local benchmark.

## Methodological lineage

The distractor arm is informed by [RAFT (2024)](https://arxiv.org/abs/2403.10131); graded support and evidence-use auditing by [RAGBench](https://arxiv.org/abs/2407.11005) and evidence-sufficiency work; position controls by [Lost in the Middle](https://arxiv.org/abs/2307.03172); and explicit treatment of retrieval failure by [CRAG](https://arxiv.org/abs/2401.15884). These sources motivate controls and measurement, not an implementation copy or a claim of novelty.
