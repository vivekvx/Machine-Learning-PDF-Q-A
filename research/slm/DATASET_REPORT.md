# QASPER non-medical curation audit

This report supersedes the first keyword-only pilot audit. QASPER is pinned at source
revision `fdc9d8214fbab5dd782958601db4d678e6934a54` and published Parquet
conversion revision `06806e4608976fc2fac0a090ac425d5b2b29caf4`.
The older `prepare_dataset.py`, `pilot/`, and `manifest.json` are retained as a
legacy record and are not the curated research dataset.

| Official split | Source docs | Included docs | Excluded docs | Quarantined docs | Usable questions | Quarantined questions in included docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 888 | 826 | 35 | 27 | 2,311 | 76 |
| Validation | 281 | 248 | 21 | 12 | 823 | 82 |
| Test | 416 | 394 | 14 | 8 | 1,208 | 160 |

Included questions comprise 2,056/770/1,134 answerable and 255/53/74
unanswerable examples in train/validation/test. The **219 conflicting**
answerability cases (5/67/147) and **99 answerable cases lacking any supplied
evidence** (71/15/13) are quarantined with raw annotations. The earlier totals
of 147 and 73 applied to the much smaller, overfiltered cohort; the revised
title/abstract curation exposes more source annotations. No answer or evidence
was fabricated.

The document filter reads **title and abstract only**. The published Parquet
schema has no subject metadata. A domain term in the title or two distinct
domain terms in the abstract causes exclusion; a single abstract domain signal,
ambiguous term, or missing abstract causes quarantine. The full decision and
reason for every paper is in `curated/document_decisions.jsonl`. Neither QA
answers nor full-text keyword hits influence the document decision. This rule
can still miss indirect medical topics and quarantine benign NLP papers;
manual review of those cases is needed before a final study.

The original document-level train/validation/test assignments are preserved.
Exact document-ID and full-text hashes plus high-similarity title/abstract
comparisons found **zero cross-split duplicate pairs** under the documented
thresholds. This is a detection rule, not a proof that semantic duplicates do
not exist. Source and output SHA-256 hashes and all rules are recorded in
`curated_manifest.json`.

QASPER's dataset card states **CC BY 4.0** and credits Dasigi et al. (2021),
*A Dataset of Information-Seeking Questions and Answers Anchored in Research
Papers*. Its papers derive from S2ORC. That dataset license does **not**
independently establish redistribution rights for every underlying paper.

For grounded evaluation, `validation_gold_eval.jsonl` has four answerable and
four unanswerable questions from eight distinct validation documents. Source
evidence IDs and section/paragraph indices replace unavailable PDF page numbers.
Unanswerable examples use a labeled deterministic document excerpt because
there is no gold evidence for them. `validation_faiss_eval.jsonl` is a separate
FAISS TF-IDF retrieval condition on the same questions. No test question or
answer was used to select baseline prompts or examples. Four answerable gold
examples were manually spot-checked in `MANUAL_EVIDENCE_AUDIT.md`.

## Rebuild

From the repository root, in a Python environment with
`research/slm/requirements.txt` and `requirements-retrieval.txt` installed:

```sh
python research/slm/curate_dataset.py
python research/slm/build_eval.py
python research/slm/build_faiss_eval.py
python -m unittest discover -s research/slm/tests -v
python research/slm/validate_curated.py
```

The pinned Parquet shards are cached in ignored `research/slm/.cache/`.
`curated/` contains complete included document and question JSONL files,
quarantine records, per-document decisions, and duplicate audit records.
Validation inputs are bounded and are not a representative test set.
