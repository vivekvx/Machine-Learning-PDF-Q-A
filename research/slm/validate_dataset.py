"""Validate the generated QASPER pilot and its manifest."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import ROOT, SPLITS, exclusion_matches, sha256  # noqa: E402


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for split, expected in manifest["source_file_sha256"].items():
        cached = root / ".cache" / f"{split}.parquet"
        if cached.exists():
            assert sha256(cached) == expected, f"source hash mismatch: {split}"
    for filename, expected in manifest["pilot_file_sha256"].items():
        assert sha256(root / "pilot" / filename) == expected, f"hash mismatch: {filename}"
    all_ids = set()
    all_qids = set()
    for split in SPLITS:
        docs = read_jsonl(root / "pilot" / f"{split}_documents.jsonl")
        questions = read_jsonl(root / "pilot" / f"{split}_questions.jsonl")
        ids = {doc["document_id"] for doc in docs}
        assert len(ids) == len(docs) == len(questions)
        assert not (all_ids & ids), f"cross-split document leak: {split}"
        all_ids.update(ids)
        assert ids == set(manifest["pilot_document_ids"][split])
        assert len(docs) == manifest["splits"][split]["pilot_documents"]
        assert len(questions) == manifest["splits"][split]["pilot_questions"]
        for doc in docs:
            assert doc["split"] == split and not exclusion_matches({
                "title": doc["title"], "abstract": doc["abstract"],
                "full_text": {"section_name": [s["section_name"] for s in doc["sections"]],
                              "paragraphs": [s["paragraphs"] for s in doc["sections"]]},
                "figures_and_tables": doc["figures_and_tables"],
                "qas": {"question": [], "answers": []}})
        for q in questions:
            assert q["split"] == split and q["document_id"] in ids
            assert q["question_id"] not in all_qids
            all_qids.add(q["question_id"])
            assert q["answerability"] in ("answerable", "unanswerable")
            assert q["annotations"] and all(a["annotation_id"] for a in q["annotations"])
            assert all(a["unanswerable"] == (q["answerability"] == "unanswerable") for a in q["annotations"])
            if q["answerability"] == "answerable":
                assert any(a["evidence"] for a in q["annotations"])
    assert not manifest["audit"]["duplicate_documents"]
    return manifest


if __name__ == "__main__":
    result = validate()
    print("Validated pilot:", {s: result["splits"][s]["pilot_questions"] for s in SPLITS})
