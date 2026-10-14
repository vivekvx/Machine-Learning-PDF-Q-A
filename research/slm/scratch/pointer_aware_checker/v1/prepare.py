"""Fresh document-disjoint synthetic sources for the frozen pointer checker.

This creates no test data. The pre-existing sealed test file is never read here.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1]
sys.path.insert(0, str(SCRATCH))
from pointer_qa.prepare import document, pools, FAMILIES, sha  # noqa: E402
from pointer_qa.answerability.prepare import absent_question, corrected_passage  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402


def config():
    return json.loads((HERE / "config.json").read_text())


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_scenarios(source, group, seed):
    """All labels derive from generated facts; same value has positive/negative pairs."""
    old = source["facts"][source["gold_candidate_index"]]
    original = {"passage": source["passage"], "question": source["question"],
                "supported_value": old["value"], "kind": "original"}
    corrected, new_index, _ = corrected_passage(source, seed)
    current = enumerate_candidates(corrected)[new_index]["value"]
    correction = {"passage": corrected, "question": source["question"],
                  "supported_value": current, "kind": "correction"}
    unrelated_source = group[(group.index(source) + 1) % len(group)]
    unrelated, _, _ = corrected_passage(unrelated_source, seed)
    distractor = {"passage": unrelated, "question": source["question"],
                  "supported_value": old["value"], "kind": "unrelated_correction"}
    for item in (original, correction, distractor):
        candidates = enumerate_candidates(item["passage"])
        matches = [i for i, x in enumerate(candidates) if x["value"] == item["supported_value"]]
        if len(matches) != 1:
            raise ValueError("Unique supported span required")
        item["supported_indices"] = matches
        item["scenario_id"] = source["id"] + ":" + item.pop("kind")
        item["document_id"] = source["document_id"]
        item["source_question_id"] = source["id"]
        item["condition"] = item["scenario_id"].rsplit(":", 1)[1]
        del item["supported_value"]
    return [original, correction, distractor]


def generate():
    c = config()
    names = pools(json.loads((SCRATCH / "pointer_qa" / "v1" / "config.json").read_text())["seed"])
    groups = {}
    for family in ("manual", "deployment", "logbook", "specification"):
        for i in range(c["train_documents_per_family"]):
            rows = document("train", family, c["source_index_offset"] + i,
                            names["train"], c["source_seed"])
            groups[rows[0]["document_id"]] = rows
    nval = sum(c["validation_documents"].values())
    for i in range(nval):
        rows = document("validation", "bulletin", c["source_index_offset"] + i,
                        names["validation"], c["source_seed"])
        groups[rows[0]["document_id"]] = rows
    train_ids = {k for k in groups if k.startswith("train-")}
    val_ids = sorted(set(groups) - train_ids)
    a, b, g = c["validation_documents"].values()
    allocation = {"train": sorted(train_ids), "calibration": val_ids[:a],
                  "selection": val_ids[a:a+b], "final_gate": val_ids[a+b:a+b+g]}
    result = {key: [] for key in allocation}
    for partition, docs in allocation.items():
        for did in docs:
            group = groups[did]
            for row in group:
                result[partition].extend(make_scenarios(row, group, c["seed"]))
            for entity in sorted({r["target_entity"] for r in group}):
                row = next(r for r in group if r["target_entity"] == entity)
                result[partition].append({"scenario_id": row["id"] + ":missing",
                    "document_id": did, "source_question_id": row["id"],
                    "passage": row["passage"], "question": absent_question(row),
                    "supported_indices": [], "condition": "missing"})
    for key, scenarios in result.items():
        ids = [x["scenario_id"] for x in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scenario IDs")
        for scenario in scenarios:
            if not enumerate_candidates(scenario["passage"]):
                raise ValueError("No source candidates")
        if len({r["document_id"] for r in scenarios}) != len(allocation[key]):
            raise ValueError("Document count changed")
    texts = {key: {digest_text(r["passage"] + "\n" + r["question"]) for r in rows}
             for key, rows in result.items()}
    for first in texts:
        for second in texts:
            if first < second and texts[first] & texts[second]:
                raise ValueError("Exact question-context leakage")
    return result, allocation


def build():
    if (HERE / "manifest.json").exists():
        raise FileExistsError("Version 1 checker data already frozen")
    result, allocation = generate()
    for key, rows in result.items():
        with (HERE / f"{key}.jsonl").open("x") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
    files = ["config.json", *[f"{key}.jsonl" for key in result]]
    manifest = {"schema_version": 1, "rights": "original generated synthetic technical text",
        "test_sealed": True, "source_pointer_selection_sha256": sha(SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json"),
        "source_pointer_tokenizer_sha256": sha(SCRATCH / "pointer_qa" / "v1" / "tokenizer.json"),
        "source_generator_sha256": sha(SCRATCH / "pointer_qa" / "prepare.py"),
        "correction_generator_sha256": sha(SCRATCH / "pointer_qa" / "answerability" / "prepare.py"),
        "document_ids": allocation,
        "stats": {key: {"documents": len(allocation[key]), "scenarios": len(rows),
                         "conditions": dict(Counter(r["condition"] for r in rows))}
                  for key, rows in result.items()},
        "files_sha256": {name: sha(HERE / name) for name in files}}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["stats"]


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError(f"Input changed: {name}")
    if manifest["source_pointer_selection_sha256"] != sha(SCRATCH / "pointer_qa" / "v2" / "validation" / "pointer_selection.json"):
        raise ValueError("Frozen pointer selection changed")
    return manifest["stats"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build() if args.build else verify(), indent=2))
