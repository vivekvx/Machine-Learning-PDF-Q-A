# Qwen3-0.6B QLoRA training feasibility (2026-10-11)

**Result:** A guarded, five-optimizer-step QLoRA run completed on the Apple M2
MacBook Air with 8 GB unified memory. The isolated adapter loaded and generated
two validation responses. **NO-GO for longer training under the measured memory
conditions:** swap free fell to 463 MB during this short run. Five steps establish
API and checkpoint compatibility, not sustained memory safety or answer quality.

## Inputs and provenance

The existing curated QASPER cohort and untuned baseline were read before this
experiment. Their files were not regenerated. The curated split counts remain
2,311 train, 823 validation, and 1,208 test usable questions, after title/abstract
domain screening and annotation quarantine. QASPER source revision is
`fdc9d8214fbab5dd782958601db4d678e6934a54`; published Parquet revision is
`06806e4608976fc2fac0a090ac425d5b2b29caf4`. Its dataset card states CC BY
4.0 and credits Dasigi et al. (2021), but does not independently establish the
rights to every underlying S2ORC paper. See `DATASET_REPORT.md` and
`curated_manifest.json` for exclusions and checksums.

`prepare_feasibility_sample.py` selects **32 training questions** from 32
train documents (24 answerable with a matched source paragraph containing the
reference answer; 8 consensus unanswerable) and **8 validation questions**
from 8 different validation documents (4 answerable, 4 unanswerable). The
validation documents also differ from the baseline evaluation documents. There
are no test records in these JSONL files. The selected train/validation examples
are from included, non-medical documents. For unanswerable questions QASPER
provides no gold supporting evidence; the input contains an explicitly labeled
document excerpt, not fabricated gold evidence. This is a bounded-context
training feasibility sample, not a comprehensive audit of whether every paper
passage could answer each question. Answerable support is checked by exact
normalized answer text in a QASPER matched paragraph; six sample rows were
also inspected manually for obvious context/target mismatch, without exhaustive
manual validation.

Manual spot checks: answerable IDs `753990d0b621d390ed58f20c4d9e4f065f0dc672`,
`938cf30c4f1d14fa182e82919e16072fdbcf2a82`, and
`e07df8f613dbd567a35318cd6f6f4cb959f5c82d` each had an answer phrase
in the displayed passage with question-relevant context. Unanswerable IDs
`31b6544346e9a31d656e197ad01756813ee89422`,
`10091275f777e0c2890c3ac0fd0a7d8e266b57cf`, and
`519db0922376ce1e87fcdedaa626d665d9f3e8ce1` were checked for the
literal `UNANSWERABLE` target and correctly labeled excerpts; those excerpts
alone cannot establish absence of an answer in the full document.

The training/validation SHA-256 values are
`faecb86f9ef6bbf40a0e0966bf0127aca33c929b6b82b9b3fb5900217f0f16e5`
and `35819c88fb25d13eb6ca1a5fe904e410ca3a8bb306f8a6678e50c3b57f0a35d3`.
`feasibility_data/manifest.json` records source-manifest and model revisions.

## Runtime and method

The already cached `mlx-community/Qwen3-0.6B-4bit` revision
`73e3e38d981303bc594367cd910ea6eb48349da8` was used; its local
`model.safetensors` SHA-256 was verified as
`392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2`.
The conversion card names `Qwen/Qwen3-0.6B` as its source but does not specify
the precise source-base commit. No weights or packages were downloaded or
upgraded. Installed versions: Python 3.14.6, `mlx-lm==0.32.0`, `mlx==0.32.3`,
`transformers==5.19.0`, `psutil==7.1.0`; macOS 26.6.2, arm64 Apple M2,
8,589,934,592 bytes unified memory.

The installed MLX-LM `train_model`/`TrainingArgs` API supports LoRA on the
quantized model, `mask_prompt`, a maximum sequence length, layer count,
gradient checkpointing, and adapter output. A local `QwenMaskedDataset`
applies the pinned Qwen tokenizer chat template with `enable_thinking=False`
and returns the assistant-start offset, so the supervised loss targets the
assistant answer and end marker. The expected target is exactly `UNANSWERABLE`
for consensus unanswerable records. This uses the installed framework's
completion mask; tokenizer prefix/offset checks run over all 40 rows.

Configuration: batch 1, maximum sequence 384 tokens, 5 optimizer steps,
last 2 transformer layers, LoRA rank 4/scale 8/dropout 0, Adam learning rate
`1e-5`, gradient checkpointing enabled, seed 0, one-step loss reporting.
Trainable parameters were 0.180 million (0.030% of 596.050 million). The
MLX-LM trainer's automatic recommended wired-limit call was disabled inside
the worker; no macOS-wide GPU memory limit was changed. The training log's
initial `WikiSQL` dataset line came from the installed API's default display
field; the actual input was the local `QwenMaskedDataset`. The saved adapter
configuration's display field was corrected to the local data path after the
run; adapter weights were not edited.

The parent process verified model/sample hashes and checked available memory,
swap, memory pressure, and disk space before launch. Its watchdog sampled
about twice per second and would terminate the worker below 700 MB available
memory, below 400 MB free swap, above 600 MB swap growth, above 2.5 GB worker
RSS, below 10% system memory-pressure free percentage, or after 240 seconds.
These guards were not triggered. The complete thresholds and time series are
in `training_feasibility_run.json`.

## Measurements

| Measurement | Observed |
| --- | ---: |
| Completed optimizer steps | 5/5 |
| Wall time, worker | 11.76 s |
| Step durations | 5.115, 0.340, 0.596, 0.255, 0.295 s |
| Training losses | 2.475, 2.680, 3.295, 6.688, 5.616 |
| Supervised tokens across steps | 70 |
| Peak MLX allocation reported by trainer | 750,599,806 bytes |
| Peak worker RSS sampled by parent | 347,684,864 bytes |
| Lowest sampled system-available memory | 1,145,487,360 bytes |
| Lowest sampled free swap | 463,405,056 bytes |
| Maximum swap growth versus preflight | 514,850,816 bytes |
| Lowest sampled system memory-pressure free value | 27% |

Preflight had 1,582,776,320 bytes available memory and 978,255,872 bytes
free swap. These are whole-system measurements; other processes may contribute
to changes. MLX allocation is not total system memory. Gradient checkpointing
was supported and enabled, but its benefit was not isolated by an A/B run.
The five losses are noisy and offer no evidence of improvement.

The adapter is `feasibility_adapter/adapters.safetensors` (707 KB; SHA-256
`fe742fc4a562aa63163f6f6f79d59a2b0bf064b7485cc789b98a4220448810a3`).
`mlx_lm.load(..., adapter_path=...)` succeeded without changing the base model.
Two validation prompts were generated, each capped at 32 tokens:

| Label | Reference | Adapter response |
| --- | --- | --- |
| Answerable | “Knowledge Base Question Answering” | “KBQA stands for Knowledge Base Question Answering.” |
| Unanswerable | `UNANSWERABLE` | “The question cannot be answered with the provided information.” |

These are compatibility checks, not a quality benchmark. The second response
conveys abstention but fails the strict required marker. The original baseline
predictions and metrics remain untouched. `baseline_format_audit.json` records
that, among four original unanswerable validation questions, strict marker
accuracy was 0/4 for each evidence condition; marker anywhere occurred in 1/4
gold and 2/4 FAISS responses, while a conservative abstention-language scan
flagged 4/4 in each. This separates answer-format/parsing misses from the
original strict metric, without treating every prose response as a valid
`UNANSWERABLE` generation. Baseline SHA-256 values remain:

| Original file | SHA-256 |
| --- | --- |
| `baseline_smoke.json` | `5505f2950d121d697b8a04f5f3eb3ce6b65dc0f56d84cd979f14c87a5153a60e` |
| `baseline_validation.json` | `bd4acc9b9a82804fcf0714d362dff5fc59ca95407330676c3fc260af7c07eb7f` |
| `baseline_faiss_validation.json` | `c9abb0c7a86a3aed804f39a5bd37e378c9d18bd34e567b04a64131b76e6f98a4` |

## Retrieval distinction

The application code in `src/embeddings.py` uses
`sentence-transformers/all-MiniLM-L6-v2`, and `src/vector_store.py` indexes
its dense vectors with FAISS `IndexFlatIP`. The research evaluation code in
`build_faiss_eval.py` instead builds per-document word/bigram **TF-IDF** vectors
and searches them with FAISS `IndexFlatIP`. Thus FAISS is common to both;
the embedding methods differ. This is verified from code, not a runtime
comparison: the application's sentence-transformer import failed in the
existing environment because PyTorch's `libtorch_cpu.dylib` was missing.

## Reproduce and checks

From the repository root with the existing isolated MLX environment and cached
model/data:

```sh
research/slm/.venv/bin/python research/slm/prepare_feasibility_sample.py
research/slm/.venv/bin/python -m unittest discover -s research/slm/tests -p test_feasibility.py -v
research/slm/.venv/bin/python research/slm/run_feasibility.py
research/slm/.venv/bin/python research/slm/verify_adapter.py
```

The runner refuses to overwrite an existing log, adapter, or run result; move
those artifacts before a new run. The feasibility tests passed (3/3). The
broader existing unittest suite and `validate_curated.py` could not import
because this MLX-only environment lacks `pyarrow`; their results were not
claimed as passes in this run. Baseline hashes were rechecked after inference.

**Next safe step:** Free sufficient swap and memory headroom, then repeat a
guarded profiling run at the intended sequence length with a larger, manually
reviewed training subset and strict output-format validation. Do not start
longer LoRA training on the present 8 GB machine state until sustained memory
behavior, annotation quality, and the unanswerable output format are resolved.
