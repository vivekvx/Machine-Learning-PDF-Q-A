# Train-only audit v1

The saved v1 directory records the fixed 352 train IDs, 80 validation IDs, full source annotations, automated evidence flags, 122 train-only candidate documents, a licensing ledger, and a blocked three-arm variant design. It contains no reviewed training variants and no test answers. The original v1 outputs are never overwritten.

Run from the repository root with the existing tokenizer-capable `research/slm/.venv`:

```sh
research/slm/.venv/bin/python research/slm/training_data_v2/validate.py
research/slm/.venv/bin/python -m unittest discover -s research/slm/training_data_v2/tests -v
```

To independently rebuild without touching v1, use an empty temporary directory:

```sh
audit_tmp=$(mktemp -d)
research/slm/.venv/bin/python research/slm/training_data_v2/build.py \
  --output-dir "$audit_tmp/v1" --review-csv "$audit_tmp/EVIDENCE_REVIEW_SAMPLE.csv"
```

Compare the rebuilt `manifest.json` and each `file_sha256` against `training_data_v2/v1/manifest.json`; the current code reproduced all eight artifact hashes and the entire manifest on 2026-10-11. The build reads only curated train questions/documents, document filter decisions, the fixed train records, and the fixed validation records for ID isolation; it does not open test question or answer files, download weights, or run a model. It uses the locally cached pinned tokenizer. Source hashes, config hash, and generation-tool hashes are recorded in the manifest; if any source changes, the build stops.

Manual reviewers should work from copies of `EVIDENCE_REVIEW_SAMPLE.csv`, leave the original unchanged, save each independent review separately, and adjudicate before a new versioned training release. See `TRAINING_DATA_AUDIT.md`, `LICENSE_PROVENANCE.md`, and `EXPERIMENT_PROTOCOL_V2.md` for the rubric and release gates.
