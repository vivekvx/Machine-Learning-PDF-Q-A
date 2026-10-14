"""Deterministic QASPER curation. Filtering never reads QA fields."""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

from prepare_dataset import (ROOT, SOURCE_REVISION, PARQUET_REVISION, SOURCE_URL,
                             SPLITS, iter_documents, question_records, sections,
                             sha256, source_path, normalized)

OUT = ROOT / "curated"
DOMAIN = re.compile(r"\b(?:biomed(?:ical)?|medical|medicine|clinical|clinician|health(?:care)?|hospital|patients?|diseases?|cancer|tumou?rs?|pharmac(?:eutical|ology)|drugs?|psychiatr(?:y|ic)|epidem(?:ic|iology)|covid|coronavirus|sars|vaccines?|surgery|radiolog(?:y|ical)|genom(?:e|ic)|genetic|proteins?|diabetes|dementia|psychosis|pubmed)\b", re.I)
AMBIGUOUS = re.compile(r"\b(?:cell|cells|virus|viral|treatment|therapy|diagnosis|mental health|stress disorder|depression)\b", re.I)
WORD = re.compile(r"\w+", re.UNICODE)


def classify_document(doc: dict) -> tuple[str, list[str]]:
    """Use only title/abstract. QASPER has no subject metadata in its Parquet schema."""
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    title_domain = sorted({m.group().casefold() for m in DOMAIN.finditer(title)})
    abstract_domain = sorted({m.group().casefold() for m in DOMAIN.finditer(abstract)})
    ambiguous = sorted({m.group().casefold() for m in AMBIGUOUS.finditer(title + " " + abstract)})
    if title_domain:
        return "exclude", ["title_domain:" + x for x in title_domain]
    if len(abstract_domain) >= 2:
        return "exclude", ["abstract_multi_domain:" + x for x in abstract_domain]
    if abstract_domain or ambiguous or not abstract.strip():
        return "quarantine", ["abstract_domain:" + x for x in abstract_domain] + ["ambiguous:" + x for x in ambiguous] + (["missing_abstract"] if not abstract.strip() else [])
    return "include", ["no_title_abstract_domain_signal"]


def answer_text(annotation: dict) -> str | None:
    free = (annotation.get("free_form_answer") or "").strip()
    spans = [s.strip() for s in annotation.get("extractive_spans") or [] if s and s.strip()]
    if free:
        return free
    if spans:
        return "; ".join(spans)
    if annotation.get("yes_no") is True:
        return "Yes"
    if annotation.get("yes_no") is False:
        return "No"
    return None


def resolve_question(q: dict) -> tuple[str, list[str], list[str]]:
    """Return status, reason codes, and all available reference answer variants."""
    annotations = q["annotations"]
    labels = {a.get("unanswerable") for a in annotations}
    if not annotations or None in labels:
        return "quarantine", ["missing_answerability"], []
    if len(labels) > 1:
        return "quarantine", ["conflicting_answerability"], []
    if True in labels:
        return "use", ["unanswerable_consensus"], ["UNANSWERABLE"]
    answers = list(dict.fromkeys(t for a in annotations if (t := answer_text(a))))
    if not answers:
        return "quarantine", ["missing_answer_text"], []
    if not any(any(str(e).strip() for e in a.get("evidence") or []) for a in annotations):
        return "quarantine", ["missing_evidence"], answers
    return "use", ["answerable_with_supplied_evidence"], answers


def text_fingerprint(doc: dict) -> str:
    body = " ".join(p or "" for s in sections(doc) for p in s["paragraphs"])
    return hashlib.sha256(normalized(body).encode()).hexdigest()


def title_grams(title: str) -> set[str]:
    t = normalized(title)
    return {t[i:i + 5] for i in range(max(0, len(t) - 4))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def duplicate_pairs(metadata: list[dict]) -> list[dict]:
    """Exact ID/body plus high-similarity title+abstract across official splits."""
    pairs = []
    for i, a in enumerate(metadata):
        for b in metadata[i + 1:]:
            if a["split"] == b["split"]:
                continue
            reason = None
            if a["document_id"] == b["document_id"]:
                reason = "same_id"
            elif a["body_sha256"] == b["body_sha256"]:
                reason = "same_body"
            elif a["title_grams"] and b["title_grams"] and jaccard(a["title_grams"], b["title_grams"]) >= .88:
                if jaccard(a["abstract_words"], b["abstract_words"]) >= .80:
                    reason = "near_duplicate_title_abstract"
            if reason:
                pairs.append({"left": [a["split"], a["document_id"]],
                              "right": [b["split"], b["document_id"]], "reason": reason})
    return pairs


def evidence_passages(q: dict, doc: dict) -> list[dict]:
    lookup = collections.defaultdict(list)
    for section in sections(doc):
        for pi, paragraph in enumerate(section["paragraphs"]):
            lookup[normalized(paragraph)].append((section["section_index"], pi))
    passages = []
    seen = set()
    for annotation in q["annotations"]:
        if annotation["unanswerable"]:
            continue
        for ei, evidence in enumerate(annotation.get("evidence") or []):
            evidence = (evidence or "").strip()
            if not evidence or normalized(evidence) in seen:
                continue
            seen.add(normalized(evidence))
            locations = lookup.get(normalized(evidence), [])
            passages.append({"evidence_id": f"{annotation['annotation_id']}:{ei}",
                             "text": evidence, "section_index": locations[0][0] if locations else None,
                             "paragraph_index": locations[0][1] if locations else None,
                             "source_match": bool(locations), "source_type": "paragraph" if locations else "unmatched_or_figure_table"})
    return passages


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build() -> dict:
    OUT.mkdir(exist_ok=True)
    cache = ROOT / ".cache"
    sources = {s: source_path(s, cache) for s in SPLITS}
    source_hashes = {s: sha256(p) for s, p in sources.items()}
    # Pass one sees metadata only. Never inspect held-out QA fields for filtering.
    decisions, meta = {}, []
    for split in SPLITS:
        for doc in iter_documents(sources[split]):
            status, reasons = classify_document(doc)
            key = (split, doc["id"])
            decisions[key] = {"split": split, "document_id": doc["id"], "title": doc["title"],
                              "subject_metadata": None, "status": status, "reasons": reasons}
            meta.append({"split": split, "document_id": doc["id"], "body_sha256": text_fingerprint(doc),
                         "title_grams": title_grams(doc["title"]),
                         "abstract_words": set(WORD.findall(normalized(doc["abstract"] or "")))})
    duplicates = duplicate_pairs(meta)
    for pair in duplicates:
        for split, did in (pair["left"], pair["right"]):
            d = decisions[(split, did)]
            if d["status"] == "include":
                d["status"] = "quarantine"
            d["reasons"].append("cross_split_" + pair["reason"])
    write_jsonl(OUT / "document_decisions.jsonl", list(decisions.values()))
    write_jsonl(OUT / "cross_split_duplicates.jsonl", duplicates)
    counts = {}
    for split in SPLITS:
        counter = collections.Counter()
        docs, questions, quarantine = [], [], []
        for doc in iter_documents(sources[split]):
            decision = decisions[(split, doc["id"])]
            counter["source_documents"] += 1
            counter["source_questions"] += len(doc["qas"]["question"])
            counter["documents_" + decision["status"]] += 1
            if decision["status"] != "include":
                counter["questions_document_" + decision["status"]] += len(doc["qas"]["question"])
                continue
            doc_record = {"split": split, "document_id": doc["id"], "title": doc["title"],
                          "abstract": doc["abstract"], "sections": sections(doc),
                          "figures_and_tables": doc.get("figures_and_tables"),
                          "source_revision": SOURCE_REVISION}
            docs.append(doc_record)
            for q in question_records(doc, split):
                status, reasons, answers = resolve_question(q)
                counter["questions_" + status] += 1
                for reason in reasons:
                    counter["reason_" + reason] += 1
                if status == "quarantine":
                    quarantine.append({"question": q, "reasons": reasons})
                    continue
                passages = evidence_passages(q, doc)
                if q["answerability"] == "answerable" and not passages:
                    quarantine.append({"question": q, "reasons": ["empty_evidence_after_normalization"]})
                    counter["questions_use"] -= 1
                    counter["questions_quarantine"] += 1
                    continue
                counter["usable_" + q["answerability"]] += 1
                counter["unmatched_evidence_passages"] += sum(not p["source_match"] for p in passages)
                questions.append({**q, "expected_answers": answers, "evidence_passages": passages,
                                  "curation_rule": reasons[0]})
        for name, rows in (("documents", docs), ("questions", questions), ("question_quarantine", quarantine)):
            write_jsonl(OUT / f"{split}_{name}.jsonl", rows)
        counts[split] = dict(counter)
    manifest = {"source_url": SOURCE_URL, "source_revision": SOURCE_REVISION,
                "parquet_revision": PARQUET_REVISION, "source_sha256": source_hashes,
                "license": {"dataset": "CC BY 4.0", "source": f"{SOURCE_URL}/blob/{SOURCE_REVISION}/README.md",
                            "underlying_paper_rights": "not individually established"},
                "rules": {"document_filter": "title/abstract only; no subject metadata in published Parquet",
                          "medical_domain_regex": DOMAIN.pattern, "ambiguous_regex": AMBIGUOUS.pattern,
                          "title_domain": "exclude", "two_abstract_domain_terms": "exclude",
                          "one_abstract_domain_or_ambiguous_or_missing_abstract": "quarantine",
                          "answerability_disagreement": "quarantine", "missing_supplied_evidence": "quarantine",
                          "unanswerable": "all annotations agree; expected UNANSWERABLE",
                          "answerable": "all annotations agree; retain all nonempty answer variants and original evidence"},
                "splits": counts, "cross_split_duplicates": duplicates,
                "output_sha256": {p.name: sha256(p) for p in sorted(OUT.glob("*.jsonl"))}}
    (ROOT / "curated_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build()["splits"], indent=2))
