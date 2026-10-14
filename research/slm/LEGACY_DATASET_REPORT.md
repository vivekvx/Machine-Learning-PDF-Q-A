# QASPER dataset preparation and audit

Source: [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) at pinned source revision
`fdc9d8214fbab5dd782958601db4d678e6934a54`; published Parquet conversion at `06806e4608976fc2fac0a090ac425d5b2b29caf4`.
The dataset card states **CC BY 4.0**. Attribution: Dasigi et al. (2021), A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers; Allen Institute for AI; source papers from S2ORC.
The source papers originate in S2ORC; the dataset card does not establish the
individual license of every underlying paper. Review paper rights before broader redistribution.

| Official split | Source docs | Excluded docs | Eligible docs | Eligible questions | Pilot questions | Answerable | Unanswerable | Answerable missing evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 888 | 330 | 558 | 1614 | 8 | 4 | 4 | 54 |
| validation | 281 | 115 | 166 | 617 | 8 | 4 | 4 | 10 |
| test | 416 | 167 | 249 | 849 | 8 | 4 | 4 | 9 |

| Split | Answerable questions | Unanswerable questions | Mixed | Unlabeled | Repeated question text within document | Document chars min/median/max | Estimated tokens min/median/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| train | 1428 | 184 | 2 | 0 | 1 | 0/22082/168260 | 0/4178/29932 |
| validation | 532 | 38 | 47 | 0 | 0 | 5612/18230/43544 | 1112/3438/8499 |
| test | 694 | 57 | 98 | 0 | 0 | 3824/17488/62317 | 728/3426/13532 |

Duplicate document IDs/texts across all source documents: **0**.
Excluded document counts by matched term and complete source/output hashes are in
`manifest.json`; document-level exclusions are in `pilot/exclusions.jsonl`.

Pilot files preserve each source document's split, original document/question/annotation IDs,
all answer annotations, section names and paragraphs, and supplied evidence. `exclusions.jsonl`
lists document IDs and literal terms matched by the conservative full-document filter.
Mixed answerability annotations are counted in the manifest and omitted from the pilot.
No missing evidence is filled in. QASPER does not provide PDF page numbers here.

The manifest records source and output SHA-256 hashes, answerability counts, duplicate checks,
document character lengths, and estimated token lengths. Token estimates use regex word/punctuation
units and are **not** Qwen3 tokenizer counts. The pilot contains one question per selected document,
up to four per answerability label per official split. It is a small pipeline fixture, not a
representative statistical sample. The document-level split is preserved.

## Rebuild and validate

From the repository root, install `research/slm/requirements.txt` into a Python environment,
then run:

```sh
python research/slm/prepare_dataset.py
python -m unittest discover -s research/slm/tests -v
python research/slm/validate_dataset.py
```

The builder downloads only three pinned Parquet shards into ignored `research/slm/.cache/`.
Delete that cache to force a fresh download. Validation checks output hashes and source split
integrity. The scripts do not download model weights or train a model.

## Limitations

The literal exclusion filter is deliberately conservative and may reject benign NLP papers
that merely mention medical terms. It cannot guarantee detection of every indirect biomedical
reference; inspect `exclusions.jsonl` and spot-check eligible papers before final experiments.
Answer annotations may disagree. Figure/table evidence may be represented by source markers
instead of printable passage text. The source card's split table omits test statistics; actual
Parquet counts are recorded here. The source card also leaves some curation and annotation
details as “More Information Needed.”
