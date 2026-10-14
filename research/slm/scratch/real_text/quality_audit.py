"""Deterministic checks and representative text samples from frozen v1 rows."""
import json
import re
from pathlib import Path

V1 = Path(__file__).resolve().parent / "v1"
MEDICAL = re.compile(r"\b(?:medical|biomedical|clinical|patient|pharmaceutical|healthcare|medicine|disease|diagnosis|drug)\b", re.I)
FURNITURE = re.compile(r"\.\. |>>>|https?://|\|[^\n]*\||:(?:py:)?\w+:`|\*\*Source code")


def audit():
    rows = {split: [json.loads(x) for x in (V1 / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation")}
    paragraphs = [(split, row, p) for split, docs in rows.items() for row in docs for p in row["paragraphs"]]
    normalized = [re.sub(r"\W+", " ", p["text"].casefold()).strip() for _, _, p in paragraphs]
    duplicate_count = len(normalized) - len(set(normalized))
    medical_hits = [{"document_id": row["document_id"], "source_lines": p["source_lines"], "term": m.group(0)}
                    for _, row, p in paragraphs for m in MEDICAL.finditer(p["text"])]
    furniture_hits = [{"document_id": row["document_id"], "source_lines": p["source_lines"], "match": m.group(0)}
                      for _, row, p in paragraphs for m in FURNITURE.finditer(p["text"])]
    samples = [{"split": split, "document_id": row["document_id"], "source_lines": p["source_lines"],
                "text": p["text"]}
               for split, docs in rows.items() for row in docs for p in (row["paragraphs"][0], row["paragraphs"][-1])]
    result = {"schema_version": 1, "document_count": sum(map(len, rows.values())),
              "paragraph_count": len(paragraphs), "normalized_exact_duplicate_paragraph_count": duplicate_count,
              "medical_keyword_hits": medical_hits, "furniture_regex_hits": furniture_hits,
              "sample_selection": "First and last retained prose span of each source document",
              "samples": samples, "limits": "Keyword and syntax checks are automated and do not establish independent authorship or semantic quality."}
    if duplicate_count or medical_hits or furniture_hits:
        raise ValueError("Quality audit found an excluded-content signal")
    return result


if __name__ == "__main__":
    with (V1 / "quality_audit.json").open("x") as stream:
        json.dump(audit(), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
