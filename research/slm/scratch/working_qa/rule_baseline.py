"""Transparent template-specific reference extractor; never used by neural QA."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from working_qa.prepare import COLORS, DEVICES, FIELDS, SYNONYMS, V1, enumerate_candidates


def predict(passage, question):
    entities = [f"{color} {device}" for color in COLORS for device in DEVICES]
    mentioned = [name for name in entities if re.search(rf"\b{re.escape(name)}\b", question)]
    fields = [field for field in FIELDS if any(re.search(rf"\b{re.escape(word)}\b", question)
                                                 for word in SYNONYMS[field])]
    if len(mentioned) != 1 or len(fields) != 1:
        return "UNANSWERABLE"
    matches = []
    for candidate in enumerate_candidates(passage):
        sentence = passage[candidate["sentence_start"]:candidate["sentence_end"]]
        if (re.search(rf"\b{re.escape(mentioned[0])}\b", sentence) and
            re.search(rf"\b{re.escape(fields[0])}\b", sentence)):
            matches.append(candidate["value"])
    return matches[0] if len(matches) == 1 else "UNANSWERABLE"


def evaluate_validation():
    rows = [json.loads(line) for line in (V1 / "validation.jsonl").read_text().splitlines()]
    result = {"model_type": "deterministic synthetic-template baseline; not a neural model",
              "count": len(rows), "answerable": sum(r["answerable"] for r in rows),
              "answerable_exact": sum(predict(r["passage"], r["question"]) == r["expected_answer"]
                                      for r in rows if r["answerable"]),
              "unanswerable": sum(not r["answerable"] for r in rows),
              "unanswerable_correct": sum(predict(r["passage"], r["question"]) == "UNANSWERABLE"
                                          for r in rows if not r["answerable"])}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Baseline output already exists")
    report = evaluate_validation()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
