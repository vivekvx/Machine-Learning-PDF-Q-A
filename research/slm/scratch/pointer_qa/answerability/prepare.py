"""Balanced original sufficiency cases from train/validation source documents."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1"
POINTER = HERE.parent
sys.path.insert(0, str(POINTER.parent))
from working_qa.prepare import enumerate_candidates  # noqa: E402
from pointer_qa.prepare import FIELDS, SYNONYMS, V1 as SOURCE, sha, value  # noqa: E402


def source_rows(split):
    if split not in ("train", "validation"):
        raise ValueError("Classifier preparation cannot open sealed test")
    return [json.loads(line) for line in (SOURCE / f"{split}.jsonl").read_text().splitlines()]


def absent_question(row):
    present = {fact["field"] for fact in row["facts"]}
    missing = next(field for field in FIELDS if field not in present)
    return f"What is the {SYNONYMS[missing][0]} for {row['target_entity']}?"


def corrected_passage(row, seed):
    old = row["facts"][row["gold_candidate_index"]]
    used = {fact["value"] for fact in row["facts"]}
    rng = random.Random(f"{seed}:correction:{row['id']}")
    replacement = value(old["field"], rng, used)
    sentence = (f"Correction: For {old['entity']}, the current {old['field']} is "
                f"[ {replacement} ]; the earlier value is obsolete.")
    original = [row["passage"][fact["sentence_start"]:fact["sentence_end"]] for fact in row["facts"]]
    insertion = rng.randrange(5)
    original.insert(insertion, sentence)
    passage = " ".join(original)
    candidates = enumerate_candidates(passage)
    if len(candidates) != 5:
        raise ValueError("Correction candidates missing")
    new_indices = [i for i, item in enumerate(candidates) if item["value"] == replacement]
    old_indices = [i for i, item in enumerate(candidates) if item["value"] == old["value"]]
    if len(new_indices) != 1 or len(old_indices) != 1:
        raise ValueError("Correction candidate ambiguity")
    return passage, new_indices[0], old_indices[0]


def scenario_records(row, seed):
    index = row["gold_candidate_index"]
    facts = row["facts"]
    category = ("wrong_entity", "wrong_field", "missing_fact", "contradiction")[index]
    base = {"source_question_id": row["id"], "document_id": row["document_id"],
            "split": row["split"], "family": row["family"], "category": category,
            "source": "original_deterministic_synthetic_sufficiency_v1"}
    if category in ("wrong_entity", "wrong_field"):
        gold = facts[index]
        if category == "wrong_entity":
            negative = next(i for i, item in enumerate(facts)
                            if item["field"] == gold["field"] and item["entity"] != gold["entity"])
        else:
            negative = next(i for i, item in enumerate(facts)
                            if item["entity"] == gold["entity"] and item["field"] != gold["field"])
        scenarios = [("original", row["passage"], row["question"], [index],
                      [(index, True), (negative, False)])]
    elif category == "missing_fact":
        missing = absent_question(row)
        proposed = next(i for i, item in enumerate(facts) if item["entity"] == row["target_entity"])
        scenarios = [("original", row["passage"], row["question"], [index], [(index, True)]),
                     ("missing", row["passage"], missing, [], [(proposed, False)])]
    else:
        passage, new_index, old_index = corrected_passage(row, seed)
        scenarios = [("corrected", passage, row["question"], [new_index],
                      [(new_index, True), (old_index, False)])]
    result = []
    for kind, passage, question, supported, proposals in scenarios:
        candidates = enumerate_candidates(passage)
        scenario_id = f"{row['id']}:{kind}"
        for candidate_index, label in proposals:
            candidate = candidates[candidate_index]
            result.append({**base, "id": f"{scenario_id}:{candidate_index}",
                           "scenario_id": scenario_id, "scenario_kind": kind,
                           "passage": passage, "question": question,
                           "proposed_candidate_index": candidate_index,
                           "candidate_start": candidate["start"], "candidate_end": candidate["end"],
                           "candidate_value": candidate["value"],
                           "supported_candidate_indices": supported,
                           "label_supported": label})
    return result


def build():
    if (V1 / "manifest.json").exists():
        raise FileExistsError("Sufficiency corpus already frozen")
    c = json.loads((V1 / "config.json").read_text())
    selection_path = POINTER / "v2" / "validation" / "pointer_selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["pointer_gate_passed"]:
        raise RuntimeError("Pointer gate did not pass")
    datasets = {split: [item for row in source_rows(split)
                        for item in scenario_records(row, c["seed"])]
                for split in c["source_splits"]}
    for split, records in datasets.items():
        labels = Counter(row["label_supported"] for row in records)
        if labels[True] != labels[False]:
            raise ValueError("Sufficiency labels unbalanced")
        for row in records:
            candidates = enumerate_candidates(row["passage"])
            idx = row["proposed_candidate_index"]
            if (row["passage"][row["candidate_start"]:row["candidate_end"]] != row["candidate_value"] or
                candidates[idx]["value"] != row["candidate_value"] or
                row["label_supported"] != (idx in row["supported_candidate_indices"])):
                raise ValueError("Sufficiency annotation invalid")
        with (V1 / f"{split}.jsonl").open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    files = ("config.json", "train.jsonl", "validation.jsonl")
    manifest = {"schema_version": 1, "rights": "original generated synthetic technical text",
                "pointer_selection_sha256": sha(selection_path),
                "source_pointer_manifest_sha256": sha(SOURCE / "manifest.json"),
                "test_sealed": True,
                "stats": {split: {"records": len(records),
                                  "documents": len({r["document_id"] for r in records}),
                                  "supported": sum(r["label_supported"] for r in records),
                                  "unsupported": sum(not r["label_supported"] for r in records),
                                  "categories": dict(Counter(r["category"] for r in records))}
                          for split, records in datasets.items()},
                "files_sha256": {name: sha(V1 / name) for name in files}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["stats"]


def verify():
    manifest = json.loads((V1 / "manifest.json").read_text())
    if manifest["pointer_selection_sha256"] != sha(POINTER / "v2" / "validation" / "pointer_selection.json"):
        raise ValueError("Frozen pointer selection changed")
    for name, expected in manifest["files_sha256"].items():
        if sha(V1 / name) != expected:
            raise ValueError(f"Frozen sufficiency artifact changed: {name}")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--build", action="store_true")
    actions.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build() if args.build else verify(), indent=2))
