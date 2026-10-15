# Hybrid PDF assistant: paired development evaluation

**2026-10-15.** This is a local development measurement, not a blind test or an estimate of general research-paper accuracy. The original PDF fixture, questions, and short references were fixed before the `official_v2` run. No sealed test partition was opened.

## Fixed conditions and files

The [manifest](scratch/pdf_qa/hybrid_eval/official_v2/manifest.json) pins the exact four-page original synthetic PDF (`d2d8a4587097070c0d8da9d3e8720186df4848fd98766653509971ed4e047719` SHA-256), four question IDs, order, and [v8 configuration](scratch/pdf_qa/hybrid_config.v8.json) SHA-256 `809c4d6fdc69036a243056b6c16e1eb22b0a5607c54b9572922ceae4d57c0142`. The same server-side index and retrieved chunks were used in both modes for each question. Qwen-only and hybrid used the same pinned official Qwen weights, chat template, context and decoding limits. Hybrid added only optional frozen-pointer guidance on two eligible factual questions. [Individual HTTP responses](scratch/pdf_qa/hybrid_eval/official_v2/http_results.jsonl), [metrics](scratch/pdf_qa/hybrid_eval/official_v2/metrics.json), [source-offset checks](scratch/pdf_qa/hybrid_eval/official_v2/verification.json), and [content review](scratch/pdf_qa/hybrid_eval/official_v2/ASSISTANT_CONTENT_REVIEW.md) are retained. The content review was performed by the development assistant, not an independent human.

| ID | Task and reference | Qwen-only result | Hybrid result | Pointer |
| --- | --- | --- | --- | --- |
| F01 | Larch cache method; `prefix scan` | “The method used for larch cache is prefix scan. [E1]” | “The method used for the larch cache is prefix scan. [E1]” | selected the right span |
| F02 | Larch cache metric; `21 ms` | “The metric recorded for larch cache is 21 ms. [E1]” | same sentence | selected the right span |
| F03 | Missing accuracy result; insufficient | insufficient-evidence response | same | bypassed |
| F04 | Whole-paper summary | problem, method, four metrics and limits; `[E1] [E2] [E3] [E4]` pooled at end | identical | bypassed |

All eight requests returned HTTP 200. For the two factual questions, each mode contained **2/2 correct reference phrases**. Strict normalized exact match against short reference strings was **0/2** in both modes because the model returned sentences; mean token F1 was 0.3333 Qwen-only and 0.3205 hybrid. The missing-accuracy question was rejected **1/1 by deterministic measured-field logic before model inference**. It cannot measure Qwen's semantic abstention. The three generated responses per mode used existing citation IDs, but the summary's pooled IDs do not locate support for each sentence. The automatic support check flags that condition. “Answer coverage” was 1.0 for the two answerable facts and summary; it does not count a fabricated answer for F03.

The source-offset verifier checked **32 returned excerpt instances** against text extracted from the preserved fixture bytes: all offsets and page references matched, all cited IDs existed, and the retrieved evidence lists were identical between paired modes. This establishes *provenance of displayed excerpts*, not entailment of model claims. There were **0 pointer-helped, 0 pointer-harmed, and 2 unchanged** factual outcomes. The pointer's 2/2 span selection on this deliberately simple PDF is not evidence of transfer to general technical documents. There are no gold retrieval annotations adequate for a Recall@K estimate here.

## Speed and resources

| Measured quantity | Local Qwen RAG | Hybrid Scratch + Qwen RAG |
| --- | ---: | ---: |
| Median HTTP latency, four questions | 3,108 ms | 4,067 ms |
| Maximum HTTP latency | 4,526 ms | 4,781 ms |
| Peak MLX allocation seen | 806,455,564 bytes | 806,455,564 bytes |
| Minimum sampled available RAM | 1,108,115,456 bytes | 1,149,337,600 bytes |
| Largest per-request swap-used growth | 221,118,464 bytes | 0 bytes |
| Largest per-request swap-out growth | 4,456,448 bytes | 114,688 bytes |

The short circuit on F03 takes only a few milliseconds; median latency therefore should not be interpreted as a pure model-speed comparison. Pointer calls add work to F01/F02. The worker reloads Qwen for each generated answer under current 8 GB memory conditions. Resource samples are local observations, not guarantees for another workload. No unsafe guard was bypassed.

## Real-text and behavior checks

An earlier **separate development spot check**, using the already cached `mlx-community` conversion rather than the official Qwen weights, converted pinned CPython documentation training text into a PDF. Its answer about `2to3` transformations cited the wrong page excerpt even though the relevant statement existed on another page. A question about location abstained despite the source mentioning `Tools/scripts`. See [`real_cpython_v1`](scratch/pdf_qa/hybrid_eval/real_cpython_v1/) and the [main report](HYBRID_OFFLINE_RAG_REPORT.md). This is a concrete citation and recall weakness; it is not a score for the official model. Earlier community-model development responses included missing or invented citation IDs and truncation. The current application marks these as unverified instead of silently accepting them.

The actual localhost browser was tested with upload, a factual question, a broad summary, mode state, and expandable citations. Corrupt and text-empty PDFs returned 422. The tests also exercise repeated use of one uploaded index, separate document IDs, intent routing, context budgets, local-only worker instrumentation, and memory guard failures. A long 40+ page PDF, a physically disconnected network test, and independently reviewed unseen real papers remain untested.

The [full browser capture](scratch/pdf_qa/hybrid_eval/browser_official_v8_full.png) records the official-model summary and page excerpts. A later browser check also verified that Send stays disabled until indexing completes. The [run hash ledger](scratch/pdf_qa/hybrid_eval/OFFICIAL_RUN_HASHES.json) covers the frozen official-model evaluation, its original PDF, and the post-restart smoke record.

**Interpretation:** A useful supervised local demonstration is justified. Neither the tiny paired sample nor the assistant's content review establishes a hybrid improvement or reliable citation factuality. The practical default is Qwen-only until a rights-cleared, independently checked real-PDF benchmark shows a benefit from pointer guidance.

## Repeatability

From the repository root, verify the **existing saved responses without rerunning inference or rewriting the fixed scores**:

```sh
research/slm/scratch/pdf_qa/.venv/bin/python research/slm/scratch/pdf_qa/hybrid_eval/check_official_hashes.py
```

The runner `official_v2/evaluate.py` performs fresh model inference and must be used only as a new experiment with a new output directory and preregistered protocol, never to overwrite these fixed observations. The preserved fixture is `scratch/pdf_qa/hybrid_eval/final_v1/larch-study.pdf`.
