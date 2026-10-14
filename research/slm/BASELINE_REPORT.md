# Qwen3-0.6B untuned baseline

**Status (2026-10-11):** The pinned 4-bit MLX model loaded and completed a one-question
smoke test and two eight-question validation runs on this Apple M2 MacBook Air.
No fine-tuning was started. The FastAPI application was unchanged.

## Dataset

| Official split | Included docs | Excluded medical docs | Quarantined uncertain docs | Usable questions | Quarantined questions in included docs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 826 | 35 | 27 | 2,311 | 76 |
| Validation | 248 | 21 | 12 | 823 | 82 |
| Test | 394 | 14 | 8 | 1,208 | 160 |

The revised cohort contains 219 conflicting answerability cases and 99 answerable
cases without supplied evidence; all are quarantined with raw annotations. The
prior counts (147 and 73) came from a smaller cohort filtered by keywords in
full text and answers. Document decisions now use only titles and abstracts;
QASPER's Parquet has no subject field. All 1,585 source documents retain their
official split, and the exact/near-duplicate audit found no cross-split pairs
under the recorded thresholds. See [DATASET_REPORT.md](DATASET_REPORT.md),
`curated_manifest.json`, and `curated/document_decisions.jsonl` for rules,
counts, provenance, and SHA-256 hashes.

The QASPER dataset card states CC BY 4.0 and credits Dasigi et al. (2021).
That license does not independently establish rights for each S2ORC paper.
Ambiguous medical-topic decisions still need manual review.

## Model and machine

- Model: `mlx-community/Qwen3-0.6B-4bit` at commit
  `73e3e38d981303bc594367cd910ea6eb48349da8`; published model-file
  SHA-256 `392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`
  was verified before loading. The conversion card identifies `Qwen/Qwen3-0.6B`
  as its base and lists Apache-2.0, but does not specify the exact source-base commit.
- Runtime: `mlx-lm==0.32.0`, `mlx==0.32.3`, Python 3.14, macOS 26.6.2,
  Apple M2, 8,589,934,592 bytes unified memory. Exact installed versions are
  in `requirements-baseline.lock.txt`.
- Boundaries: batch size 1, maximum sequence length 1,024 tokens, maximum
  generation 64 tokens. The runner checks available memory before loading and
  between questions, stops below its configured guard, and writes partial
  results on a guarded failure.

## Measured validation results

The deterministic subset has four answerable and four unanswerable questions
from eight distinct **validation** papers. Test examples were not used for
prompt selection or model selection. Gold-evidence inputs and FAISS-retrieved
inputs are separate JSONL files. The FAISS condition uses per-document TF-IDF
word/bigram vectors with `IndexFlatIP`; it is not the existing RAG embedding
pipeline, whose local PyTorch import failed because `libtorch_cpu.dylib` was
missing. Metrics are exact-match and token-F1 against any preserved answer
variant, strict `UNANSWERABLE` marker accuracy, and wall-clock generation time.

| Condition | N | Exact match | Token F1 | Strict abstention accuracy | Mean generation latency | Peak MLX memory | Minimum observed available memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold supplied evidence / labeled document excerpt | 8 | 0.000 | 0.208 | 0.500 | 0.490 s | 832 MB | 1,291 MB |
| FAISS TF-IDF retrieved paragraphs | 8 | 0.000 | 0.108 | 0.500 | 0.516 s | 1,001 MB | 1,299 MB |

On the four **unanswerable** examples, strict abstention accuracy was **0/4**
in both conditions. The model sometimes described uncertainty in prose but did
not emit the required machine-readable marker. The single-question cold smoke
took 2.43 s; the validation latency is warm-run latency on this machine.

Example gold-evidence predictions:

- Question: “what ner models were evaluated?” Reference: “Stanford NER;
  spaCy 2.0; recurrent model with a CRF top layer.” Prediction: “The ner
  models evaluated in the document are Stanford NER, spaCy 2.0, and a
  recurrent model...” (F1 0.438; exact match 0).
- Unanswerable question: “What is the interannotator agreement for the human
  evaluation?” Prediction: “The question cannot be answered with the provided
  information.” This conveys uncertainty but fails the strict
  `UNANSWERABLE` output requirement.

Four answerable evidence-support samples were manually reviewed in
`MANUAL_EVIDENCE_AUDIT.md`: three were supported by the selected passage,
and one supported an answer variant only partially. Nonempty evidence is not
proof of complete annotation support.

## Memory and LoRA readiness

The model ran without breaching the 900 MB runtime available-memory guard.
Across the full runs, peak MLX allocations were 0.83 GB (gold) and 1.00 GB
(FAISS); peak process RSS was 0.28 GB. The lowest observed system-available
memory was about **1.29 GB**. That number fluctuates with other applications
and is **not** a guaranteed allocation for LoRA. Gradient activations and
optimizer state were not measured. Inference compatibility is verified;
comfortable 8 GB LoRA training is **not yet verified**. The next safe step is a
separate, short rank/context-length memory profiling plan before any optimizer
step, with other applications' memory pressure recorded.

## Reproduce

From the repository root, with the preparation and retrieval requirements
installed in the project Python environment:

```sh
python research/slm/curate_dataset.py
python research/slm/build_eval.py
python research/slm/build_faiss_eval.py
python -m unittest discover -s research/slm/tests -v
python research/slm/validate_curated.py
```

Create an isolated MLX environment and install
`research/slm/requirements-baseline.lock.txt`. The model downloader uses only
the pinned conversion commit and checks the model-file SHA-256:

```sh
python research/slm/download_model.py
research/slm/.venv/bin/python research/slm/run_baseline.py --smoke --output research/slm/baseline_smoke.json
research/slm/.venv/bin/python research/slm/run_baseline.py --output research/slm/baseline_validation.json
research/slm/.venv/bin/python research/slm/run_baseline.py --input research/slm/validation_faiss_eval.jsonl --output research/slm/baseline_faiss_validation.json
```

Results and per-question predictions are saved in the three JSON files. The
sample is intentionally tiny, oracle gold evidence is not deployable retrieval,
question/answer annotation disagreements remain quarantined, and no
held-out test performance or fine-tuning gain is claimed.
