# Retrieval validation and untuned Qwen3-0.6B baseline

**Run:** 2026-10-11, immutable `research/slm/retrieval_baseline/v1/` artifacts. **Status:** Both retrievers and all three 80-question untuned inference conditions completed; no fine-tuning or deployment occurred. The original FastAPI application, earlier baseline predictions, and five-step training records were not changed.

## Fixed cohort and provenance

The 80 QASPER validation questions comprise 40 answerable and 40 consensus-unanswerable questions from 80 distinct included, non-medical validation documents. They are disjoint from the 352 selected train documents. The source JSONL SHA-256 is `54ef6bbffda3072918d51e0ce37cdf24c87be9e844b64b23b02742794eeedba0`. QASPER is pinned at source revision `fdc9d8214fbab5dd782958601db4d678e6934a54` and Parquet revision `06806e4608976fc2fac0a090ac425d5b2b29caf4`. The output validator checked all 80 IDs, their order, document provenance, source paragraph text/hashes, and the 512-token prompt plus 64-token generation reserve. No held-out test answers were read by the new retrieval, inference, training audit, or output-validation scripts.

The inherited curation excluded 35 train, 21 validation, and 14 test medical/biomedical/healthcare documents and quarantined 27/12/8 uncertain documents, respectively. This run did not reclassify them. QASPER's dataset card states CC BY 4.0 and credits Dasigi et al. (2021); it does **not** independently establish redistribution rights for every underlying S2ORC paper. See `DATASET_REPORT.md` for the filter and annotation audit.

## Retrieval environment and measured comparison

The existing project `venv` had a broken PyTorch installation (`libtorch_cpu.dylib` missing). An isolated Python 3.11.15 environment at `research/slm/.retrieval-venv` now has pinned Torch 2.7.1, Sentence Transformers 5.1.2, Transformers 4.56.2, FAISS 1.12.0, and scikit-learn 1.7.2; the complete transitive lock is `retrieval_baseline/requirements.lock.txt` (SHA-256 `2fe021695fd9f87c77621778a59f19a249d5c1dd0708ed829ad59e894d4706c3`). The pinned MiniLM `model.safetensors` blob was absent despite a cached snapshot link, so only that file was fetched. Its SHA-256 is `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`. **Real CPU inference was verified**: two test strings produced finite `float32` vectors of shape `(2, 384)` with unit norms. The working MLX environment was not modified.

Both methods ranked the same whole non-empty paragraphs from each question's source document, with identical paragraph counts, no cross-document search, top 3, exact FAISS inner-product search, and the same Qwen token-budget packer. TF-IDF used per-document word/bigram vectors (up to 8,192 features, L2 normalized); MiniLM used normalized dense embeddings. Gold matching uses exact `section:i:paragraph:j` IDs. Retrieval labels and validation references were attached **after** ranking and were not used as queries.

| Metric, 40 answerable questions | TF-IDF+FAISS | MiniLM+FAISS |
| --- | ---: | ---: |
| Recall@1 | 8/40 = 20.0% | 11/40 = 27.5% |
| Recall@3 | 15/40 = 37.5% | 21/40 = 52.5% |
| MRR@3 | 0.275 | 0.375 |
| Gold paragraph actually packed | 12/40 = 30.0% | 15/40 = 37.5% |
| Top-3 candidates rejected by context budget, all 80 queries | 66 | 45 |
| Queries with no fitting passage | 0 | 0 |
| Warm per-question retrieval mean / p95 | 6.1 / 8.9 ms | 416.8 / 885.4 ms |

The warm figures exclude the first query in each fresh process because it imported FAISS/scikit-learn and initialized backend machinery inside the timer. The raw 80-query means, including that first import, were 23.4 ms and 439.8 ms; MiniLM model load was measured separately at 0.687 s. These are single-run local CPU measurements, not benchmark confidence intervals. MiniLM ranked more gold paragraphs, but the 512-token context budget removed six of its top-3 gold hits (versus three TF-IDF hits). Recall/MRR are undefined for the 40 globally unanswerable questions because they have no gold evidence paragraph ID.

## Untuned Qwen inference and output audit

The existing local `mlx-community/Qwen3-0.6B-4bit` revision `73e3e38d981303bc594367cd910ea6eb48349da8` was used without an adapter; model weight SHA-256 `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`. MLX-LM 0.32.0 / MLX 0.32.3 ran greedy batch-1 generation on the Apple M2. The pinned Qwen chat template used `enable_thinking=False`, which renders a closed empty `<think>` block in the assistant prefix. Streaming generation recorded stop reason and token count; the tokenizer stops on `<|im_end|>` (ID 151645), with a 64-token limit. The identical system instruction, question, answer contract, and generation settings were used for gold and both retrieved conditions. Only system/user messages were sent to the model; references and assistant targets were used afterward for scoring. A one-question smoke test passed before the 80-question runs.

| Metric | Gold passage | TF-IDF retrieval | MiniLM retrieval |
| --- | ---: | ---: | ---: |
| Completed questions | 80/80 | 80/80 | 80/80 |
| Answerable exact match | 0/40 | 0/40 | 0/40 |
| Answerable mean token F1 | 0.274 | 0.132 | 0.150 |
| Unanswerable exact `UNANSWERABLE` recall | 0/40 | 0/40 | 0/40 |
| Abstention precision | undefined (no exact abstentions) | undefined | undefined |
| Answerable reference citation present | 22/40 | 5/40 | 7/40 |
| Generation stopped by 64-token cap | 7 | 8 | 14 |
| Format: cited / uncited / malformed citation | 23 / 52 / 5 | 16 / 48 / 16 | 16 / 48 / 16 |
| Mean / p95 answer latency | 0.327 / 0.558 s | 0.432 / 0.658 s | 0.457 / 0.667 s |
| Peak MLX allocation | 749 MB | 773 MB | 772 MB |
| Minimum system available memory | 1.73 GB | 1.66 GB | 1.68 GB |
| Minimum free swap | 1.32 GB | 1.40 GB | 1.42 GB |

Exact match strips valid citation markers and compares normalized word sequences; it remains zero when a semantically related response is verbose or incomplete. Example: the gold prediction for “What does KBQA abbreviate for” was “KBQA stands for Knowledge Base Question Answering. [section:0:paragraph:0]”, supported by that passage, but not an exact match to the shorter reference. Conversely, with TF-IDF the model gave the correct expansion while the packed passages did **not** contain it: that answer was not grounded in supplied evidence. The model often wrote prose such as “The passage does not establish an answer...” instead of the required exact `UNANSWERABLE`. Such text is recorded as an output-format/scoring failure, never silently converted into an abstention. The parser also flags empty, control/thinking-token, malformed-citation, unknown-stop, and truncated responses. No generation exceptions were recorded in the completed 240 predictions. Every parseable `[section:i:paragraph:j]` citation referred to a selected passage, but that syntactic check does not prove entailment; no invented page numbers were detected.

An assistant qualitative spot-check reviewed the **first three answerable and first three unanswerable rows per condition** (18 response/context pairs). It found supported and unsupported answers, context-appropriate prose refusals that fail the exact format, and false refusals despite sufficient gold context. Detailed judgments and reasons are in `retrieval_baseline/v1/support_spotcheck.jsonl`. This is not an independent, blinded human support audit, and full-set semantic evidence support remains **unmeasured**. Lexical substring proxies in the prediction JSON must not be interpreted as entailment rates.

The 8 GB M2 passed the existing 1.2 GB available / 0.75 GB free-swap preflight for each run and never crossed the 0.9 GB / 0.4 GB runtime guard. Before the one-question smoke test, free swap was only about 0.76 GB, barely above preflight; the machine had already used several GB of swap. These runs establish feasibility for bounded **inference under the measured conditions**, not comfortable headroom for longer training or simultaneous services. The five-step QLoRA feasibility report already recorded more severe swap pressure.

## Training-data readiness and unresolved issues

The existing training pilot has **352 examples from 352 distinct included train documents**: 256 answerable rows with exactly one source-matched QASPER paragraph and answer string found there, plus 96 consensus-unanswerable rows with a labeled document excerpt that is **not gold evidence**. Re-auditing all 2,311 curated train questions against the same rules found 752 eligible questions across 474 documents. This leaves **122 additional eligible train documents** outside the selected 352 (61 answerable and 61 unanswerable under first eligible source-order selection); no expansion was performed. Rejection counts across all train questions: 633 reference answer absent from matched gold paragraph, 473 multiple distinct gold paragraphs, 362 missing/unmatched evidence, 76 source paragraph mismatches, and 15 over token budget. These strict rules enrich for short extractive questions; they do not verify that every answer is semantically supported or that every unanswerable excerpt is truly insufficient. This run's 18-row spot-check and the pending human evidence queue underscore that limit.

For a controlled evidence-sufficiency experiment, the 352 rows are a reproducible **pilot**, not yet an audited full training cohort. Partial and irrelevant evidence labels have not been built or independently reviewed. Upstream Qwen-to-MLX conversion lineage and a compatible cloud GPU/PEFT runtime remain unverified. The protocol in `EXPERIMENT_PROTOCOL.md` defines matched arms and a frozen test policy; it is a plan, not a training result.

**Recommendation: NO-GO for controlled cloud LoRA training now.** Proceed with independent human support/answerability review, explicit partial/irrelevant construction audits, underlying-paper rights review, and pinned cloud model/runtime preflight. Then freeze the comparison arms and validation selection rule before any training. Provisioning a cloud environment for those read-only checks is reasonable. Keep the held-out test answers sealed until the final locked evaluation.

## Reproduction and checks

The versioned retrieval inputs, individual predictions, stop/error/format fields, per-question passage IDs and hashes, model/config hashes, metrics, manifests, and QASPER annotation/evidence-ID sidecar are under `retrieval_baseline/v1/`. `artifact_manifest.json` validates the core output SHA-256 hashes; `prediction_provenance.manifest.json` and `support_spotcheck.manifest.json` validate the two sidecars. See `retrieval_baseline/README.md` for the isolated environment and opt-in rebuild commands. A rebuild can reproduce IDs, ranks, and deterministic greedy outputs, but runtime fields make whole-file hashes host/run-specific.

The final second pass succeeded: six new focused tests; eight existing portable tests (including hashes of previous baseline and five-step records); five curation, three dataset, and three feasibility tests in their respective existing environments; curated-data validation; artifact validation of all 240 predictions; source paragraph and prompt-budget checks; real MiniLM inference; and compilation of the new scripts. No test split answers were accessed, and no training was started.
