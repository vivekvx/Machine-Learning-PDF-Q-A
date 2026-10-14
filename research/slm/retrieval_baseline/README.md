# Frozen retrieval and untuned Qwen baseline

The `v1/` artifacts are the fixed 80-question QASPER **validation** comparison. Each condition uses the same 80 document IDs, whole source paragraphs, question text, top-3 ranking, 512-token Qwen prompt limit, 64-token generation reserve, and exact section/paragraph gold matching. Gold evidence is an oracle diagnostic condition. No script here trains or reads held-out test answers.

## Isolated retrieval environment

Use Python 3.11.15. The repository `venv` had `torch` installed without `libtorch_cpu.dylib`; the independent environment below fixes retrieval without touching `research/slm/.venv` (MLX).

```sh
uv venv research/slm/.retrieval-venv --python /opt/homebrew/bin/python3.11
uv pip sync --python research/slm/.retrieval-venv/bin/python research/slm/retrieval_baseline/requirements.lock.txt
```

The pinned MiniLM revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Only `model.safetensors` was missing from the local snapshot. If absent on another host, fetch that file explicitly before the offline retrieval commands:

```sh
research/slm/.retrieval-venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', filename='model.safetensors', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'))"
```

The full local Qwen MLX snapshot must already be present at `research/slm/.cache/qwen3-0.6b-4bit` with the weight/config hashes in `portable/v1/mlx.json`. The inference runner never downloads it.

## Rebuild without overwriting v1

Run from the repository root. Use a new empty output directory; the tools refuse to overwrite versioned results. The source dataset and model caches must already be prepared by the earlier `research/slm` pipeline.

```sh
export SLM_RETRIEVAL_BASELINE_OUT="$PWD/research/slm/retrieval_baseline/rebuild-v2"
mkdir -p "$SLM_RETRIEVAL_BASELINE_OUT"
cp research/slm/retrieval_baseline/v1/config.json "$SLM_RETRIEVAL_BASELINE_OUT/config.json"
research/slm/.venv/bin/python research/slm/retrieval_baseline/audit_training.py --output "$SLM_RETRIEVAL_BASELINE_OUT/training_audit.json"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 research/slm/.retrieval-venv/bin/python research/slm/retrieval_baseline/run_retrieval.py --condition tfidf_faiss
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 research/slm/.retrieval-venv/bin/python research/slm/retrieval_baseline/run_retrieval.py --condition minilm_faiss
research/slm/.venv/bin/python research/slm/retrieval_baseline/evaluate.py --condition gold
```

The final command is preflight only. It must report `ready: true` before inference. A one-question smoke run precedes the full runs:

```sh
research/slm/.venv/bin/python research/slm/retrieval_baseline/evaluate.py --condition gold --execute --smoke --output "$SLM_RETRIEVAL_BASELINE_OUT/smoke_gold.json"
research/slm/.venv/bin/python research/slm/retrieval_baseline/evaluate.py --condition gold --execute --output "$SLM_RETRIEVAL_BASELINE_OUT/predictions_gold.json"
research/slm/.venv/bin/python research/slm/retrieval_baseline/evaluate.py --condition tfidf_faiss --execute --output "$SLM_RETRIEVAL_BASELINE_OUT/predictions_tfidf_faiss.json"
research/slm/.venv/bin/python research/slm/retrieval_baseline/evaluate.py --condition minilm_faiss --execute --output "$SLM_RETRIEVAL_BASELINE_OUT/predictions_minilm_faiss.json"
research/slm/.venv/bin/python research/slm/retrieval_baseline/export_provenance.py
```

Each 80-question inference run checks at least 1.2 GB available memory and 0.75 GB free swap before loading; it stops below 0.9 GB available or 0.4 GB free swap. It records `stopped_or_failed` with the last completed question if the guard trips. Do not weaken the guard to force a run. The machine's swap use varies with unrelated processes, so another run may legitimately fail preflight. The outputs include runtime latencies and memory, so whole-file hashes are expected to differ on a rebuild; compare question order, rankings, generated responses, and metrics explicitly.

`v1/support_spotcheck.jsonl` is an assistant qualitative review of a prespecified 18-row sample, not an independent human judgment. Repeat that review on rebuilt predictions before using it. The artifact validator additionally requires that review file and its checksum manifest:

```sh
research/slm/.venv/bin/python research/slm/retrieval_baseline/validate.py
research/slm/.venv/bin/python -m unittest discover -s research/slm/retrieval_baseline/tests -v
research/slm/.venv/bin/python -m unittest discover -s research/slm/portable/tests -v
```

Unset `SLM_RETRIEVAL_BASELINE_OUT` to validate the saved `v1/` artifacts. The validator checks their hashes, and the portable tests check the prior-result hashes. The `v1/artifact_manifest.json` and sidecar provenance/support manifests hold SHA-256 checksums. Do not overwrite previous baseline, feasibility, or app files.
