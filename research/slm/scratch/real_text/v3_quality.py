"""Deterministic train/validation quality samples; never reads sealed test."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "v3"


def sample():
    result = {"schema_version": 1, "selection": "SHA-256 of split:document:source-lines; first eight train and four validation paragraphs",
              "automated_sample_only": True, "human_review_decisions": None,
              "sealed_test_opened_or_displayed": False, "samples": []}
    for split, count in (("train", 8), ("validation", 4)):
        rows = [json.loads(line) for line in (V3 / f"{split}.jsonl").read_text().splitlines()]
        candidates = [(hashlib.sha256(f"{split}:{row['document_id']}:{p['source_lines']}".encode()).hexdigest(), row, p)
                      for row in rows for p in row["paragraphs"]]
        for _, row, p in sorted(candidates)[:count]:
            result["samples"].append({"split": split, "document_id": row["document_id"],
                                      "source_url": row["source_url"], "source_lines": p["source_lines"],
                                      "text": p["text"], "text_sha256": p["sha256"]})
    return result


if __name__ == "__main__":
    output = V3 / "quality_samples.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(sample(), indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(sample()['samples'])} train/validation samples")
