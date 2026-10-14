"""Read-only train/validation shortcut audit; the sealed test is never read."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from working_qa.prepare import V1, sha, verify  # noqa: E402
from working_qa.train import rows  # noqa: E402


def styles(row):
    out = []
    for item in row["candidates"]:
        text = row["passage"][item["sentence_start"]:item["sentence_end"]]
        if text.startswith("For "):
            out.append("for_lists")
        elif " says " in text:
            out.append("says_has")
        elif ": " in text:
            out.append("colon_is")
        else:
            raise ValueError("Unexpected sentence shape")
    return tuple(out)


def audit():
    verify()
    train, validation = rows("train"), rows("validation")
    train_shapes = {(row["tier"], row["answerable"], styles(row)) for row in train}
    val_shapes = {(row["tier"], row["answerable"], styles(row)) for row in validation}
    return {"train_count": len(train), "validation_count": len(validation),
            "train_unique_fact_ids": len({r["fact_id"] for r in train}),
            "validation_unique_fact_ids": len({r["fact_id"] for r in validation}),
            "shared_fact_ids": len({r["fact_id"] for r in train} & {r["fact_id"] for r in validation}),
            "shared_entity_names": len({r["target_entity"] for r in train} & {r["target_entity"] for r in validation}),
            "shared_exact_passage_question": len({(r["passage"], r["question"]) for r in train} &
                                                {(r["passage"], r["question"]) for r in validation}),
            "train_structural_signatures": len(train_shapes),
            "validation_structural_signatures": len(val_shapes),
            "validation_signatures_seen_in_train": len(train_shapes & val_shapes),
            "validation_rows_with_seen_structure": sum((r["tier"], r["answerable"], styles(r)) in train_shapes
                                                       for r in validation),
            "answerability_by_candidate_count_train": {str(n): dict(Counter(str(r["answerable"]) for r in train
                                                                            if len(r["candidates"]) == n))
                                                       for n in (1, 2, 4)},
            "train_sha256": sha(V1 / "train.jsonl"),
            "validation_sha256": sha(V1 / "validation.jsonl"),
            "test_file_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit artifact already exists")
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
