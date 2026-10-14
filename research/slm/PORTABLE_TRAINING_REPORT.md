# Portable SLM experiment preparation (2026-10-11)

**Status:** The datasets, versioned configs, opt-in entrypoints, retrieval
condition definitions, and 80-question evaluation harness are ready for
review. No further training or model inference was run for this step. The
existing FastAPI code, untuned baseline files, and five-step feasibility
records are unchanged.

## Fixed data and audit

The v1 dataset uses only included, non-medical QASPER train and validation
documents from the existing curated cohort. It preserves document-level
splits; no test question or answer is read by the new preparation, retrieval,
training, or evaluation scripts. QASPER is pinned at source revision
`fdc9d8214fbab5dd782958601db4d678e6934a54` and published Parquet
revision `06806e4608976fc2fac0a090ac425d5b2b29caf4`.

| File | Answerable | Unanswerable | Documents | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `portable/data/v1/train.jsonl` | 256 | 96 | 352 train | `34ee127615ebbdf800066257b6d06022be37ecc6a0b25edb3cac0422d35cda9a` |
| `portable/data/v1/validation_gold.jsonl` | 40 | 40 | 80 validation | `54ef6bbffda3072918d51e0ce37cdf24c87be9e844b64b23b02742794eeedba0` |

Answerable selection requires exactly one distinct source-matched QASPER
paragraph, no unmatched evidence, and a reference answer found in that exact
paragraph after normalization. All annotation evidence IDs for that paragraph
are retained. This deliberately rejects multi-passage questions rather than
quietly dropping potentially necessary evidence. The target is the selected
answer followed by its real section/paragraph ID, such as
`[section:4:paragraph:1]`. No PDF page numbers are invented. Consensus
unanswerable rows target exactly `UNANSWERABLE`; their bounded document excerpt
is labeled **not gold evidence**. A single excerpt does not prove full-document
unanswerability. The pilot is biased toward short, extractive, single-passage
questions and must not be described as representative of all QASPER QA.

Every selected row and retrieval candidate is kept whole. Source paragraphs
that exceed the 512-token sequence or leave fewer than 64 generation tokens
are rejected or logged, never silently clipped. In the examined source-order
candidates, 9 train and 3 validation questions were rejected for token budget;
the exact per-reason decisions are in `selection_audit.jsonl` and
`data/v1/manifest.json`. The maximum training full length was 468 tokens;
the maximum validation gold full length was 441. The ten-row
`evidence_review_sample.jsonl` is a **pending** human review queue.

The local MLX tokenizer configuration and pinned upstream Qwen tokenizer
configuration have different full chat-template hashes because the upstream
template handles additional message forms. For **all 432 selected rows**, their
`enable_thinking=False` prompt and complete message token IDs matched exactly.
The training code checks this prefix and masks every prompt token with `-100`
(PEFT) or the framework's completion offset (MLX), leaving only assistant
completion tokens as targets. Future tokenizer drift fails preflight.

## Separate retrieval conditions

`portable/v1/retrieval.json` records three conditions:

| Condition | Representation and index | Verified status |
| --- | --- | --- |
| Gold | Exact QASPER source paragraph; labeled excerpt for unanswerable | 80 rows built and validated |
| TF-IDF+FAISS | Per-document word/bigram TF-IDF, 8,192 features, L2 normalization, FAISS `IndexFlatIP`, top 3 | 80 rows built; 12/40 answerable gold paragraphs retrieved after context packing |
| MiniLM+FAISS | `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized dense vectors, FAISS `IndexFlatIP`, top 3 | Config and offline builder prepared; runtime not verified |

TF-IDF's 80-row output hash is
`28bd63a8c9825c2220b044414022b8266e8d15dae33c9211d4b98e808cace38f`.
It records 66 top-three candidate passages excluded by the token budget and
zero rows with no fitting passage. Those are retrieval coverage measurements,
not model answer scores. Retrieval ranks from question text and the source
document before attaching gold labels. The application's own RAG code uses
MiniLM sentence-transformer embeddings plus FAISS; the earlier research FAISS
baseline and this new TF-IDF condition use **TF-IDF** plus FAISS. This local
machine has a cached MiniLM revision, but the available PyTorch build lacks
`libtorch_cpu.dylib`, so MiniLM inference was not executed.

## Backend configs and readiness

The MLX config pins `mlx-community/Qwen3-0.6B-4bit` at
`73e3e38d981303bc594367cd910ea6eb48349da8` and verifies the existing
model-file hash. Its installed and previously exercised versions are
`mlx-lm==0.32.0`, `mlx==0.32.3`, Python 3.14.6 on the Apple M2. The proposed
training config is batch 1, max sequence 512, seed 20261011, 200 steps,
rank 4, the last 2 layers, seven attention/MLP projections, learning rate
`1e-5`, and gradient checkpointing. MLX's `scale=2` is a direct multiplier;
the GPU PEFT proposal uses `lora_alpha=8` with rank 4, giving a nominal
`alpha/r=2` multiplier. These settings align the intended low-rank update
scale, but do **not** make adapters or training numerics interchangeable.
The MLX config uses Adam and the GPU proposal uses AdamW; the backend
comparison must record this optimizer difference.

The GPU config pins upstream `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`. It proposes Linux CUDA,
PyTorch/PEFT, NF4 4-bit quantization with double quantization and bfloat16
compute. Direct dependency versions in `portable/requirements-gpu.txt` were
checked to exist on PyPI; their combined runtime and the GPU LoRA path were
**not tested**. The upstream base snapshot's weights are not present locally,
and the MLX conversion card does not give its exact original base commit.
Therefore the two pinned model artifacts must be treated as separate baselines
until conversion fidelity is checked. A PEFT adapter must not be loaded as an
MLX adapter without a validated conversion.

Both training entrypoints run preflight by default and require `--execute`
plus a new output directory to train. They never download weights. MLX
preflight currently **refuses** training because measured free swap was about
1.20 GB, below the configured 2 GB minimum; earlier five-step training had
left as little as 463 MB free swap. The GPU preflight correctly reports no
CUDA device, no pinned upstream weights, missing GPU packages, and less than
the configured 10 GB free storage on this Mac. No longer run was attempted.
The older five-step entrypoint was also made opt-in; its saved run records,
adapter, and report were left intact.

## Evaluation protocol and limitations

`portable/evaluate.py` fixes the same 80 validation question IDs across gold,
TF-IDF, and future MiniLM conditions. It is opt-in and keeps answerable and
unanswerable metrics separate: answerable exact match and token F1 (citations
stripped for answer scoring), strict `UNANSWERABLE` accuracy, false
abstention, reference-citation rate, unsupported citations, gold-passage
retrieval recall, invented page-reference flags, lexical evidence-support
proxies, latency, process RSS,
model peak allocation, and system memory. It writes per-question records and
ten manual support/citation review forms. Lexical overlap is not a semantic
support judgment. **No 80-question model metrics or quality improvements are
claimed:** evaluation inference and manual review have not run.

This study can now ask separately whether task-specific LoRA changes grounded
answer quality, citations, and abstention on the fixed validation conditions,
and whether a resulting model can perform useful inference on an 8 GB laptop.
Fine-tuning may use a larger GPU machine; laptop deployment would require a
validated export/quantization path and a fresh memory/latency measurement.
The held-out test split remains reserved for a final locked assessment.

QASPER's dataset card states CC BY 4.0 and credits Dasigi et al. (2021).
This does not independently establish redistribution rights for every
underlying S2ORC paper. Title/abstract filtering and the unresolved medical
quarantine still have the limits documented in `DATASET_REPORT.md`.

## Checks and next steps

Portable tests passed **8/8**: document isolation, exact source passages,
context budget, cross-tokenizer serialization, assistant-only labels,
retrieval provenance, scoring, and unchanged prior artifacts. The existing
curation tests passed **5/5**, legacy dataset tests **3/3**, and
the prior feasibility tests passed **3/3**. `validate_curated.py` passed
with the repository environment's `pyarrow`.
MLX training and evaluation preflight-only commands ran; no training or
inference was started. See `portable/README.md` for reproducible commands.

Before real fine-tuning: review the evidence queue and annotation bias; fix
the MiniLM/PyTorch environment and build its separate retrieval condition;
obtain and verify the pinned upstream weights on a suitable CUDA machine;
run GPU dependency and tokenizer preflight; establish free-memory/storage
headroom; then execute a bounded training run and compare an untuned and
tuned model on the **same** 80 validation rows. Validate any GPU-to-MLX
conversion before measuring 8 GB laptop inference. Do not use test answers
for model selection.
