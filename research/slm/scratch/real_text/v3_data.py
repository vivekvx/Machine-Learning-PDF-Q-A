"""Training-visible v3 loader. The sealed test file is never opened."""
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "v3"
sys.path.insert(0, str(ROOT.parent))

from checkpoint import digest  # noqa: E402
from v3_tokenizer import V3ByteBPE, SPECIAL  # noqa: E402


def identity(window):
    return {"document_id": window["document_id"], "offset": window["offset"],
            "ids_sha256": hashlib.sha256(json.dumps(window["ids"], separators=(",", ":")).encode()).hexdigest()}


def window_sort_key(window, seed):
    return hashlib.sha256(f"{seed}:{window['document_id']}:{window['offset']}".encode()).hexdigest()


def all_windows(rows, tokenizer, sequence_length):
    result = []
    for row in rows:
        ids = tokenizer.encode(row["text"], bos=True, eos=True)
        for offset in range(0, len(ids) - sequence_length, sequence_length):
            chunk = ids[offset:offset + sequence_length + 1]
            if len(chunk) == sequence_length + 1:
                result.append({"document_id": row["document_id"], "group_id": row["group_id"],
                               "offset": offset, "ids": chunk})
    return result


def fixed_windows(all_, config, split):
    by_doc = defaultdict(list)
    for window in all_:
        by_doc[window["document_id"]].append(window)
    selected = []
    for doc, windows in by_doc.items():
        ordered = sorted(windows, key=lambda w: window_sort_key(w, config["seed"]))
        selected.extend(ordered[:1 if split == "train" else config["fixed_validation_max_windows_per_document"]])
    if split == "train":
        selected = sorted(selected, key=lambda w: window_sort_key(w, config["seed"]))[:config["fixed_train_window_count"]]
    return selected


def plan_from_windows(train_windows, val_windows, config):
    fixed_train = fixed_windows(train_windows, config, "train")
    fixed_val = fixed_windows(val_windows, config, "validation")
    indices = list(range(len(train_windows)))
    random.Random(config["seed"]).shuffle(indices)
    selected = [train_windows[i] for i in indices[:config["optimizer_steps_cap"]]]
    if len(fixed_train) != config["fixed_train_window_count"] or len(selected) != config["optimizer_steps_cap"]:
        raise ValueError("Insufficient train windows")
    if len({w["document_id"] for w in fixed_val}) < 10:
        raise ValueError("Validation lacks independent documents")
    return {"fixed_train": fixed_train, "fixed_validation": fixed_val, "training_sample_order": selected}


def load_verified():
    config = json.loads((V3 / "pilot_config.v1.json").read_text())
    manifest = json.loads((V3 / "manifest.json").read_text())
    corpus_config = json.loads((V3 / "config.json").read_text())
    if manifest["config_sha256"] != digest(V3 / "config.json") or not manifest["test_sealed_for_training_and_model_selection"]:
        raise ValueError("v3 manifest/config or seal differs")
    if digest(V3 / "source/cpython-e848e4b.tar.gz") != manifest["archive_sha256"]:
        raise ValueError("Pinned official archive differs")
    for name in ("train.jsonl", "validation.jsonl", "tokenizer.json", "tokenizer_evaluation.json",
                 "leakage_audit.json", "source_ledger.json", "grouping_audit.json", "LICENSE", "Doc_license.rst"):
        if digest(V3 / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Training-visible corpus artifact changed: {name}")
    if "Python software and documentation are licensed under the\nPython Software Foundation License Version 2." not in (V3 / "Doc_license.rst").read_text():
        raise ValueError("Official documentation rights statement missing")
    leakage = json.loads((V3 / "leakage_audit.json").read_text())
    if leakage["cross_split_paragraph_matches"] or leakage["cross_split_near_documents"]:
        raise ValueError("Unresolved split leakage")
    train = [json.loads(line) for line in (V3 / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (V3 / "validation.jsonl").read_text().splitlines()]
    if [r["document_id"] for r in train] != manifest["split_document_ids"]["train"] or [r["document_id"] for r in val] != manifest["split_document_ids"]["validation"]:
        raise ValueError("Split document IDs changed")
    if {r["document_id"] for r in train} & {r["document_id"] for r in val} or {r["group_id"] for r in train} & {r["group_id"] for r in val}:
        raise ValueError("Training and validation overlap")
    tokdata = json.loads((V3 / "tokenizer.json").read_text())
    if tokdata["type"] != "deterministic_whitespace_piece_byte_bpe_from_v3_train_only" or tokdata["training_document_ids"] != [r["document_id"] for r in train] or tokdata["special_ids"] != SPECIAL:
        raise ValueError("Tokenizer fit provenance differs")
    if tokdata["training_text_sha256"] != {r["document_id"]: r["text_sha256"] for r in train}:
        raise ValueError("Tokenizer was not fitted to current training text")
    tokenizer = V3ByteBPE(tokdata["merges"], tokdata["target_size"])
    if tokenizer.vocab_size != config["tokenizer_vocabulary_size"]:
        raise ValueError("Tokenizer size changed")
    if manifest["split_stats"]["train"]["unique_text_tokens"] < corpus_config["min_unique_train_tokens_for_pilot"]:
        raise ValueError("Corpus below size gate")
    train_windows = all_windows(train, tokenizer, config["sequence_length"])
    val_windows = all_windows(val, tokenizer, config["sequence_length"])
    return config, manifest, tokenizer, train, val, train_windows, val_windows


def load_plan(config, train_windows, val_windows):
    frozen = json.loads((V3 / "pilot_plan.v1.json").read_text())
    if frozen["config_sha256"] != digest(V3 / "pilot_config.v1.json") or frozen["manifest_sha256"] != digest(V3 / "manifest.json"):
        raise ValueError("Frozen pilot plan provenance mismatch")
    plan = plan_from_windows(train_windows, val_windows, config)
    for key in plan:
        if [identity(w) for w in plan[key]] != frozen[key]:
            raise ValueError(f"Frozen pilot window/sample order differs: {key}")
    return plan


def freeze_plan():
    path = V3 / "pilot_plan.v1.json"
    if path.exists():
        raise FileExistsError(path)
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    plan = plan_from_windows(train_windows, val_windows, config)
    record = {"schema_version": 1, "config_sha256": digest(V3 / "pilot_config.v1.json"),
              "manifest_sha256": digest(V3 / "manifest.json"), "tokenizer_sha256": digest(V3 / "tokenizer.json"),
              "all_train_windows": len(train_windows), "all_validation_windows": len(val_windows),
              "training_target_tokens": len(plan["training_sample_order"]) * config["sequence_length"],
              "unique_training_window_ids": len({(w["document_id"], w["offset"]) for w in plan["training_sample_order"]}),
              "sampled_training_documents": len({w["document_id"] for w in plan["training_sample_order"]}),
              "sampled_training_groups": len({w["group_id"] for w in plan["training_sample_order"]}),
              "fixed_train": [identity(w) for w in plan["fixed_train"]],
              "fixed_validation": [identity(w) for w in plan["fixed_validation"]],
              "training_sample_order": [identity(w) for w in plan["training_sample_order"]],
              "sealed_test_opened": False}
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return {k: v for k, v in record.items() if k not in ("fixed_train", "fixed_validation", "training_sample_order")}


if __name__ == "__main__":
    print(json.dumps(freeze_plan(), indent=2))
