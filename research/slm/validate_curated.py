"""Validate curated counts, split isolation, output hashes, and eval provenance."""
import json
from pathlib import Path

from curate_dataset import ROOT, OUT, SPLITS, classify_document, sha256
from build_eval import rows


def validate():
    m = json.loads((ROOT / "curated_manifest.json").read_text())
    for filename, expected in m["output_sha256"].items():
        assert sha256(OUT / filename) == expected, filename
    for split, expected in m["source_sha256"].items():
        assert sha256(ROOT / ".cache" / f"{split}.parquet") == expected, split
    decisions = {(r["split"], r["document_id"]): r for r in rows(OUT / "document_decisions.jsonl")}
    ids, qids = set(), set()
    for split in SPLITS:
        docs = list(rows(OUT / f"{split}_documents.jsonl"))
        questions = list(rows(OUT / f"{split}_questions.jsonl"))
        quarantined = list(rows(OUT / f"{split}_question_quarantine.jsonl"))
        local = {d["document_id"] for d in docs}
        assert len(local) == len(docs) == m["splits"][split]["documents_include"]
        assert not local & ids
        ids.update(local)
        assert len(questions) == m["splits"][split]["questions_use"]
        assert len(quarantined) == m["splits"][split]["questions_quarantine"]
        for d in docs:
            assert decisions[(split, d["document_id"])]["status"] == "include"
            assert classify_document(d)[0] == "include"
        for q in questions:
            assert q["document_id"] in local and q["split"] == split
            assert q["question_id"] not in qids
            qids.add(q["question_id"])
            assert q["annotations"] and q["expected_answers"]
            assert q["answerability"] in ("answerable", "unanswerable")
            if q["answerability"] == "answerable":
                assert q["evidence_passages"]
        for record in quarantined:
            assert record["question"]["question_id"] not in qids
            assert record["reasons"]
    assert not m["cross_split_duplicates"]
    validation_docs = {d["document_id"] for d in rows(OUT / "validation_documents.jsonl")}
    eval_rows = list(rows(ROOT / "validation_gold_eval.jsonl"))
    retrieval_rows = list(rows(ROOT / "validation_faiss_eval.jsonl"))
    assert len(eval_rows) == 8
    assert {r["answerability"] for r in eval_rows} == {"answerable", "unanswerable"}
    assert len({r["document_id"] for r in eval_rows}) == 8
    assert all(r["split"] == "validation" and r["document_id"] in validation_docs for r in eval_rows)
    assert [r["question_id"] for r in retrieval_rows] == [r["question_id"] for r in eval_rows]
    assert all(r["condition"] == "faiss_retrieved_evidence" for r in retrieval_rows)
    assert all(all(p["evidence_id"].startswith("retrieved:") for p in r["selected_passages"]) for r in retrieval_rows)
    return m


if __name__ == "__main__":
    result = validate()
    print("Validated curated examples:", {s: result["splits"][s]["questions_use"] for s in SPLITS})
