"""Candidate-specific support records from document-disjoint synthetic sources."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from pointer_qa.v2.train import rows as pointer_rows  # noqa: E402
from pointer_qa.answerability.prepare import absent_question, corrected_passage  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config():
    return json.loads((HERE / "config.json").read_text())


def record(source, passage, question, supported, candidate_index, scenario_kind, category):
    candidates = enumerate_candidates(passage)
    item = candidates[candidate_index]
    return {"id": f"{source['id']}:{scenario_kind}:{candidate_index}",
            "scenario_id": f"{source['id']}:{scenario_kind}",
            "source_question_id": source["id"], "document_id": source["document_id"],
            "split": source["split"], "family": source["family"],
            "scenario_kind": scenario_kind, "category": category,
            "question": question, "passage": passage,
            "candidate_index": candidate_index, "candidate_value": item["value"],
            "candidate_start": item["start"], "candidate_end": item["end"],
            "supported_indices": supported, "label_supported": candidate_index in supported,
            "source": "original_synthetic_checker_v2"}


def records_for_question(source, seed):
    i = source["gold_candidate_index"]
    facts = source["facts"]
    gold = facts[i]
    wrong_entity = next(j for j, fact in enumerate(facts)
                        if fact["field"] == gold["field"] and fact["entity"] != gold["entity"])
    wrong_field = next(j for j, fact in enumerate(facts)
                       if fact["entity"] == gold["entity"] and fact["field"] != gold["field"])
    original = [record(source, source["passage"], source["question"], [i], i,
                       "original", "supported_original"),
                record(source, source["passage"], source["question"], [i], wrong_entity,
                       "original", "wrong_entity_same_field"),
                record(source, source["passage"], source["question"], [i], wrong_field,
                       "original", "wrong_field_same_entity")]
    corrected, new_i, old_i = corrected_passage(source, seed)
    correction = [record(source, corrected, source["question"], [new_i], new_i,
                         "correction", "supported_current_value"),
                  record(source, corrected, source["question"], [new_i], old_i,
                         "correction", "obsolete_conflicting_result")]
    return original + correction


def missing_records_for_document(group):
    source = group[0]
    by_entity = {}
    for row in group:
        by_entity.setdefault(row["target_entity"], row)
    output = []
    for entity, row in sorted(by_entity.items()):
        question = absent_question(row)
        candidates = [fact["candidate_index"] for fact in row["facts"] if fact["entity"] == entity]
        if len(candidates) != 2:
            raise ValueError("Expected two plausible missing-fact candidates per entity")
        for index in candidates:
            item = record(row, row["passage"], question, [], index,
                          f"missing-{entity.replace(' ', '_')}", "required_fact_absent")
            output.append(item)
    return output


def build_split(split, seed):
    if split not in ("train", "validation"):
        raise ValueError("Sealed test unavailable before validation gate")
    sources = pointer_rows(split)
    grouped = {}
    for source in sources:
        grouped.setdefault(source["document_id"], []).append(source)
    result = []
    for group in grouped.values():
        if len(group) != 4 or {r["gold_candidate_index"] for r in group} != set(range(4)):
            raise ValueError("Source document does not have four distinct target questions")
        for source in group:
            result.extend(records_for_question(source, seed))
        result.extend(missing_records_for_document(group))
    ids = [r["id"] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate checker record IDs")
    for row in result:
        candidates = enumerate_candidates(row["passage"])
        i = row["candidate_index"]
        if (candidates[i]["value"] != row["candidate_value"] or
            row["passage"][row["candidate_start"]:row["candidate_end"]] != row["candidate_value"] or
            row["label_supported"] != (i in row["supported_indices"])):
            raise ValueError("Candidate support/offset mismatch")
    return result


def build():
    if (HERE / "manifest.json").exists():
        raise FileExistsError("Checker v2 data already frozen")
    c = config()
    datasets = {split: build_split(split, c["seed"]) for split in c["data"]["source_splits"]}
    train_docs = {r["document_id"] for r in datasets["train"]}
    val_docs = {r["document_id"] for r in datasets["validation"]}
    if train_docs & val_docs:
        raise ValueError("Document leakage")
    train_keys = {(r["passage"], r["question"], r["candidate_value"]) for r in datasets["train"]}
    val_keys = {(r["passage"], r["question"], r["candidate_value"]) for r in datasets["validation"]}
    if train_keys & val_keys:
        raise ValueError("Exact candidate-context leakage")
    for split, rows in datasets.items():
        with (HERE / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    files = ("config.json", "train.jsonl", "validation.jsonl")
    stats = {split: {"records": len(rows), "documents": len({r["document_id"] for r in rows}),
                     "scenarios": len({r["scenario_id"] for r in rows}),
                     "supported": sum(r["label_supported"] for r in rows),
                     "unsupported": sum(not r["label_supported"] for r in rows),
                     "categories": dict(Counter(r["category"] for r in rows))}
             for split, rows in datasets.items()}
    manifest = {"schema_version": 2, "rights": "original synthetic non-medical technical text",
                "source_pointer_manifest_sha256": sha(SCRATCH / "pointer_qa" / "v1" / "manifest.json"),
                "source_pointer_selection_sha256": sha(SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json"),
                "source_correction_generator_sha256": sha(SCRATCH / "pointer_qa" / "answerability" / "prepare.py"),
                "test_sealed": True, "stats": stats,
                "files_sha256": {name: sha(HERE / name) for name in files}}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return stats


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["source_pointer_manifest_sha256"] != sha(SCRATCH / "pointer_qa" / "v1" / "manifest.json"):
        raise ValueError("Source pointer corpus changed")
    if manifest["source_pointer_selection_sha256"] != sha(SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json"):
        raise ValueError("Frozen pointer selection changed")
    if manifest["source_correction_generator_sha256"] != sha(SCRATCH / "pointer_qa" / "answerability" / "prepare.py"):
        raise ValueError("Correction generator changed")
    for name, digest in manifest["files_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"Checker v2 file changed: {name}")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build() if args.build else verify(), indent=2))
