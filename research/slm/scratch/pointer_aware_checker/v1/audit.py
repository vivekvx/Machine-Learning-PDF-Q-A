"""Read-only shortcut, pairing, and leakage audit for the bounded run."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    out = HERE / "run" / "shortcut_audit.json"
    if out.exists():
        raise FileExistsError("Shortcut audit already frozen")
    datasets = {name: [json.loads(x) for x in (HERE / f"{name}.jsonl").read_text().splitlines()]
                for name in ("train", "calibration", "selection", "final_gate")}
    document_ids = {name: {x["document_id"] for x in rows} for name, rows in datasets.items()}
    exact_contexts = {name: {hashlib.sha256((x["passage"]+"\n"+x["question"]).encode()).hexdigest()
                             for x in rows} for name, rows in datasets.items()}
    document_overlap = {f"{a}/{b}": len(document_ids[a] & document_ids[b])
                        for a in datasets for b in datasets if a < b}
    context_overlap = {f"{a}/{b}": len(exact_contexts[a] & exact_contexts[b])
                       for a in datasets for b in datasets if a < b}
    if any(document_overlap.values()) or any(context_overlap.values()):
        raise ValueError("Split leakage detected")
    features = {name: [json.loads(x) for x in (HERE / "run" / f"{name}_features.jsonl").read_text().splitlines()]
                for name in datasets}
    shortcut = {}
    paired = {}
    for name, rows in features.items():
        selected = [r for r in rows if r["pointer_selected"]]
        simple = lambda r: bool(r["features"][10] == 1.0 and r["features"][12] == 0.0)
        shortcut[name] = {"selected_count": len(selected),
            "selected_correct": sum(simple(r) == r["label_supported"] for r in selected),
            "all_candidates_count": len(rows),
            "all_candidates_correct": sum(simple(r) == r["label_supported"] for r in rows)}
        by_value = defaultdict(set)
        for r in rows:
            by_value[(r["document_id"], r["candidate_value"])].add(r["label_supported"])
        paired[name] = {"same_document_candidate_value_with_both_labels":
                        sum(values == {True, False} for values in by_value.values()),
                        "unique_document_candidate_values": len(by_value)}
    result = {"document_overlap": document_overlap, "exact_question_context_overlap": context_overlap,
        "paired_label_flips": paired, "simple_boolean_shortcut": shortcut,
        "shortcut_definition": "joint_question_entity_field_match AND NOT superseded_candidate_for_question",
        "interpretation": "A hand-coded feature combination directly reproduces generated labels; validation cannot establish independently learned sufficiency.",
        "test_sealed": True,
        "feature_files_sha256": {name: sha(HERE / "run" / f"{name}_features.jsonl") for name in features}}
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
