"""Build bounded, deterministic validation inputs; test is never used here."""
import json
from pathlib import Path

from curate_dataset import ROOT, OUT, write_jsonl

LIMIT_PER_LABEL = 4
MAX_PASSAGES = 3
MAX_CHARS_PER_PASSAGE = 900


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream)


def build():
    docs = {d["document_id"]: d for d in rows(OUT / "validation_documents.jsonl")}
    selected = []
    counts = {"answerable": 0, "unanswerable": 0}
    chosen_documents = set()
    for q in rows(OUT / "validation_questions.jsonl"):
        label = q["answerability"]
        if counts[label] >= LIMIT_PER_LABEL or q["document_id"] in chosen_documents:
            continue
        doc = docs[q["document_id"]]
        if label == "answerable":
            passages = [p for p in q["evidence_passages"] if p["source_match"] and len(p["text"]) <= MAX_CHARS_PER_PASSAGE]
            if not passages:
                continue
            condition = "gold_evidence"
        else:
            # No gold evidence exists for a source-labeled unanswerable question.
            # Supply a deterministic document excerpt, clearly marked as such.
            passages = []
            for section in doc["sections"]:
                for pi, paragraph in enumerate(section["paragraphs"]):
                    if paragraph and paragraph.strip():
                        passages.append({"evidence_id": f"excerpt:{section['section_index']}:{pi}",
                                         "text": paragraph, "section_index": section["section_index"],
                                         "paragraph_index": pi, "source_match": True,
                                         "source_type": "document_excerpt_not_gold"})
                    if len(passages) >= 2:
                        break
                if len(passages) >= 2:
                    break
            condition = "document_excerpt_for_unanswerable"
        bounded = [{**p, "text": p["text"][:MAX_CHARS_PER_PASSAGE],
                    "truncated": len(p["text"]) > MAX_CHARS_PER_PASSAGE}
                   for p in passages[:MAX_PASSAGES]]
        if not bounded:
            continue
        selected.append({"split": "validation", "condition": condition,
                         "question_id": q["question_id"], "question": q["question"],
                         "document_id": q["document_id"], "title": doc["title"],
                         "selected_passages": bounded,
                         "expected_answers": q["expected_answers"],
                         "answerability": label,
                         "provenance": {"source_dataset": "allenai/qasper",
                                        "source_revision": q["source_revision"],
                                        "annotation_ids": [a["annotation_id"] for a in q["annotations"]],
                                        "no_page_numbers": True}})
        counts[label] += 1
        chosen_documents.add(q["document_id"])
    if counts != {"answerable": LIMIT_PER_LABEL, "unanswerable": LIMIT_PER_LABEL}:
        raise RuntimeError(f"insufficient validation examples: {counts}")
    write_jsonl(ROOT / "validation_gold_eval.jsonl", selected)
    return selected


if __name__ == "__main__":
    print(f"Wrote {len(build())} validation examples")
