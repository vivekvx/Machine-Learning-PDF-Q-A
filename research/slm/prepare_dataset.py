"""Legacy keyword-only QASPER pilot; superseded by curate_dataset.py.

Run from any directory: python research/slm/prepare_dataset.py
Only the pinned QASPER Parquet shards are downloaded; they are cached locally.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://huggingface.co/datasets/allenai/qasper"
SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
SPLITS = ("train", "validation", "test")
LICENSE = "CC BY 4.0"
PILOT_PER_LABEL = 4

# Conservative, literal biomedical/healthcare vocabulary. Applied to the entire
# paper AND its annotations, then logged by matched term and document ID.
EXCLUSION_TERMS = (
    "biomed", "bio-med", "biomedical", "medical", "medicine", "medicines", "medication",
    "clinical", "clinician", "patient", "patients", "health", "healthcare",
    "hospital", "disease", "diagnosis", "diagnostic", "therapy", "therapeutic",
    "treatment", "pharmaceutical", "pharmacology", "drug", "drugs",
    "cancer", "tumor", "tumour", "epidemiology", "pathology", "radiology",
    "genomic", "genome", "genetics", "gene", "genes", "protein", "proteins",
    "cell", "cells", "virus", "viral", "covid", "coronavirus", "sars",
    "vaccine", "vaccination", "surgery", "surgical", "psychiatric", "mental health",
)
EXCLUSION_RE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(EXCLUSION_TERMS, key=len, reverse=True)) + r")(?!\w)", re.I)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(split: str, cache: Path) -> Path:
    if split not in SPLITS:
        raise ValueError(split)
    path = cache / f"{split}.parquet"
    if not path.exists():
        url = f"{SOURCE_URL}/resolve/{PARQUET_REVISION}/qasper/{split}/0000.parquet"
        temp = path.with_suffix(".partial")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def iter_documents(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8):
        yield from batch.to_pylist()


def normalized(text: str) -> str:
    return " ".join((text or "").casefold().split())


def exclusion_matches(doc: dict) -> list[str]:
    parts = [doc.get("title") or "", doc.get("abstract") or ""]
    full_text = doc.get("full_text") or {}
    parts.extend(full_text.get("section_name") or [])
    for paragraph_list in full_text.get("paragraphs") or []:
        parts.extend(paragraph_list or [])
    parts.extend((doc.get("figures_and_tables") or {}).get("caption") or [])
    qas = doc.get("qas") or {}
    parts.extend(qas.get("question") or [])
    for group in qas.get("answers") or []:
        for answer in group.get("answer") or []:
            parts.append(answer.get("free_form_answer") or "")
            for key in ("extractive_spans", "evidence", "highlighted_evidence"):
                parts.extend(answer.get(key) or [])
    return sorted({m.group().casefold() for m in EXCLUSION_RE.finditer("\n".join(p or "" for p in parts))})


def sections(doc: dict) -> list[dict]:
    full = doc["full_text"]
    names, paragraphs = full["section_name"], full["paragraphs"]
    if len(names) != len(paragraphs):
        raise ValueError(f"section alignment failed: {doc['id']}")
    return [{"section_index": i, "section_name": name, "paragraphs": paras}
            for i, (name, paras) in enumerate(zip(names, paragraphs))]


def question_records(doc: dict, split: str) -> list[dict]:
    qas = doc["qas"]
    count = len(qas["question"])
    for key in ("question_id", "answers"):
        if len(qas[key]) != count:
            raise ValueError(f"question alignment failed: {doc['id']} {key}")
    result = []
    for index in range(count):
        group = qas["answers"][index]
        values, ids, workers = group["answer"], group["annotation_id"], group["worker_id"]
        if len(values) != len(ids) or len(values) != len(workers):
            raise ValueError(f"annotation alignment failed: {doc['id']} {index}")
        annotations = [{"annotation_id": aid, "worker_id": worker, **answer}
                       for aid, worker, answer in zip(ids, workers, values)]
        labels = {a["unanswerable"] for a in annotations}
        label = ("mixed" if len(labels) != 1 else
                 "unanswerable" if True in labels else "answerable") if labels else "unlabeled"
        result.append({"source_dataset": "allenai/qasper", "source_revision": SOURCE_REVISION,
                       "parquet_revision": PARQUET_REVISION, "split": split,
                       "document_id": doc["id"], "question_id": qas["question_id"][index],
                       "question": qas["question"][index], "answerability": label,
                       "annotations": annotations})
    return result


def evidence_present(record: dict) -> bool:
    return any(any(str(e).strip() for e in a.get("evidence") or [])
               for a in record["annotations"] if not a["unanswerable"])


def build(cache: Path, output: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source_files = {split: source_path(split, cache) for split in SPLITS}
    stats = {}
    selected = {}
    seen_ids, seen_texts, seen_questions = {}, {}, {}
    duplicates = []
    exclusions = []
    term_counts = collections.Counter()
    for split in SPLITS:
        counts = collections.Counter()
        lengths, token_lengths = [], []
        examples, docs = [], []
        label_docs = collections.Counter()
        for doc in iter_documents(source_files[split]):
            doc_id = doc["id"]
            counts["source_documents"] += 1
            section_list = sections(doc)
            qs = question_records(doc, split)
            counts["source_questions"] += len(qs)
            counts["source_annotations"] += sum(len(q["annotations"]) for q in qs)
            body = "\n".join(p or "" for s in section_list for p in s["paragraphs"])
            for kind, key in (("document_id", doc_id), ("document_text", hashlib.sha256(normalized(body).encode()).hexdigest())):
                seen = seen_ids if kind == "document_id" else seen_texts
                prior = seen.setdefault(key, (split, doc_id))
                if prior != (split, doc_id):
                    duplicates.append({"kind": kind, "split": split, "other_split": prior[0],
                                       "document_id": doc_id, "other_document_id": prior[1]})
            matches = exclusion_matches(doc)
            if matches:
                counts["excluded_documents"] += 1
                counts["excluded_questions"] += len(qs)
                exclusions.append({"split": split, "document_id": doc_id, "matched_terms": matches})
                term_counts.update(matches)
                continue
            counts["eligible_documents"] += 1
            counts["eligible_questions"] += len(qs)
            counts["eligible_annotations"] += sum(len(q["annotations"]) for q in qs)
            lengths.append(len(body))
            token_lengths.append(len(TOKEN_RE.findall(body)))
            for q in qs:
                counts[f"label_{q['answerability']}"] += 1
                for annotation in q["annotations"]:
                    counts["annotation_unanswerable" if annotation["unanswerable"] else "annotation_answerable"] += 1
                if q["answerability"] == "answerable" and not evidence_present(q):
                    counts["answerable_without_evidence"] += 1
                qkey = normalized(q["question"])
                old = seen_questions.setdefault((doc_id, qkey), q["question_id"])
                if old != q["question_id"]:
                    counts["duplicate_question_text_within_document"] += 1
            # Select one question per document, four per label. A document never
            # moves splits; original source order makes selection deterministic.
            for q in qs:
                label = q["answerability"]
                if label not in ("answerable", "unanswerable") or label_docs[label] >= PILOT_PER_LABEL:
                    continue
                if label == "answerable" and not evidence_present(q):
                    continue
                label_docs[label] += 1
                docs.append({"source_dataset": "allenai/qasper", "source_revision": SOURCE_REVISION,
                             "split": split, "document_id": doc_id, "title": doc["title"],
                             "abstract": doc["abstract"], "sections": section_list,
                             "figures_and_tables": doc.get("figures_and_tables")})
                examples.append(q)
                break
        for name, rows in (("documents", docs), ("questions", examples)):
            path = output / f"{split}_{name}.jsonl"
            with path.open("w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts["pilot_documents"] = len(docs)
        counts["pilot_questions"] = len(examples)
        counts["pilot_answerable"] = label_docs["answerable"]
        counts["pilot_unanswerable"] = label_docs["unanswerable"]
        stats[split] = {**dict(counts), "eligible_document_char_lengths": summarize(lengths),
                        "eligible_document_estimated_tokens": summarize(token_lengths)}
        selected[split] = [d["document_id"] for d in docs]
    exclusion_path = output / "exclusions.jsonl"
    with exclusion_path.open("w", encoding="utf-8") as out:
        for row in exclusions:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "legacy_superseded_by_curated_manifest.json",
        "source_url": SOURCE_URL, "source_revision": SOURCE_REVISION,
        "parquet_revision": PARQUET_REVISION, "license": LICENSE,
        "attribution": "Dasigi et al. (2021), A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers; Allen Institute for AI; source papers from S2ORC",
        "processing": {"split_policy": "official document-level train/validation/test; no shuffle",
                       "exclusion_scope": "entire title, abstract, section paragraphs, questions and answer annotations",
                       "exclusion_terms": list(EXCLUSION_TERMS),
                       "pilot_policy": "first four eligible documents with an evidenced answerable question and first four with a labeled unanswerable question per split, one question per document, original source order",
                       "estimated_tokens": "regex count of Unicode word groups or punctuation; not a Qwen tokenizer", "page_numbers": "not present in source; none generated"},
        "splits": stats, "pilot_document_ids": selected,
        "audit": {"exclusion_term_document_counts": dict(term_counts),
                  "duplicate_documents": duplicates},
        "source_file_sha256": {split: sha256(path) for split, path in source_files.items()},
        "pilot_file_sha256": {path.name: sha256(path) for path in sorted(output.glob("*.jsonl"))},
    }
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(manifest, ROOT / "LEGACY_DATASET_REPORT.md")
    return manifest


def summarize(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}


def write_report(manifest: dict, path: Path) -> None:
    rows = []
    for split, c in manifest["splits"].items():
        rows.append(f"| {split} | {c.get('source_documents',0)} | {c.get('excluded_documents',0)} | {c.get('eligible_documents',0)} | {c.get('eligible_questions',0)} | {c.get('pilot_questions',0)} | {c.get('pilot_answerable',0)} | {c.get('pilot_unanswerable',0)} | {c.get('answerable_without_evidence',0)} |")
    detail_rows = []
    for split, c in manifest["splits"].items():
        lengths = c["eligible_document_char_lengths"]
        tokens = c["eligible_document_estimated_tokens"]
        detail_rows.append(f"| {split} | {c.get('label_answerable',0)} | {c.get('label_unanswerable',0)} | {c.get('label_mixed',0)} | {c.get('label_unlabeled',0)} | {c.get('duplicate_question_text_within_document',0)} | {lengths['min']}/{lengths['median']}/{lengths['max']} | {tokens['min']}/{tokens['median']}/{tokens['max']} |")
    path.write_text("""# QASPER dataset preparation and audit

Source: [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) at pinned source revision
`{source_revision}`; published Parquet conversion at `{parquet_revision}`.
The dataset card states **CC BY 4.0**. Attribution: {attribution}.
The source papers originate in S2ORC; the dataset card does not establish the
individual license of every underlying paper. Review paper rights before broader redistribution.

| Official split | Source docs | Excluded docs | Eligible docs | Eligible questions | Pilot questions | Answerable | Unanswerable | Answerable missing evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

| Split | Answerable questions | Unanswerable questions | Mixed | Unlabeled | Repeated question text within document | Document chars min/median/max | Estimated tokens min/median/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{details}

Duplicate document IDs/texts across all source documents: **{duplicate_count}**.
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
""".format(source_revision=manifest["source_revision"], parquet_revision=manifest["parquet_revision"],
           attribution=manifest["attribution"], table="\n".join(rows), details="\n".join(detail_rows),
           duplicate_count=len(manifest["audit"]["duplicate_documents"])), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    print(json.dumps(build(args.cache, args.output)["splits"], indent=2))
