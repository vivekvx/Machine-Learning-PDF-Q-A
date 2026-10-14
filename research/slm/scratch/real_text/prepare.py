"""Offline, reviewed-span CPython documentation corpus and train-only tokenizer."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "v1"
sys.path.insert(0, str(ROOT.parent))
from tokenizer import ByteBPE  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_span(lines, start, end):
    raw = " ".join(line.strip() for line in lines[start - 1:end] if line.strip())
    if ".. " in raw or "::" in raw or "http" in raw or "|" in raw or ">>>" in raw:
        raise ValueError(f"Source span {start}-{end} contains a directive, code, link, or table")
    text = re.sub(r":(?:py:)?[a-z][a-z0-9_-]*:`([^`]+)`", lambda m: m.group(1).lstrip("!~"), raw)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*", r"\1", text)
    text = text.replace("\\ ", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 25 or re.search(r":\w+:`|`|\.\. |\|", text):
        raise ValueError(f"Source span {start}-{end} did not clean to plain prose")
    return text


def build_rows(config):
    rows, seen = [], set()
    for path, selection in config["source_documents"].items():
        lines = (V1 / "raw" / path).read_text(encoding="utf-8").splitlines()
        paragraphs = []
        for first, last in selection["spans"]:
            if not (1 <= first <= last <= len(lines)):
                raise ValueError("Span outside source")
            prose = clean_span(lines, first, last)
            normalized = re.sub(r"\W+", " ", prose.casefold()).strip()
            if normalized in seen:
                raise ValueError("Duplicate prose span")
            seen.add(normalized)
            paragraphs.append({"source_lines": [first, last], "text": prose, "sha256": hashlib.sha256(prose.encode()).hexdigest()})
        text = "\n\n".join(p["text"] for p in paragraphs) + "\n"
        rows.append({"document_id": path, "split": selection["split"], "source_url":
                     f"https://github.com/python/cpython/blob/{config['source_commit']}/{path}",
                     "title": path.rsplit("/", 1)[-1].removesuffix(".rst"),
                     "paragraphs": paragraphs, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return rows


def rights_gate(config):
    if config["source_tag"] != "v3.12.15" or config["source_commit"] != "e848e4b09ca4bc67c0041c9ce7526dae471d4c36":
        raise ValueError("Pinned official source revision changed")
    for name, expected in config["expected_source_sha256"].items():
        if sha256(V1 / "raw" / name) != expected:
            raise ValueError(f"Pinned source bytes differ: {name}")
    license_text = (V1 / "raw/Doc/license.rst").read_text()
    if "Python software and documentation are licensed under the\nPython Software Foundation License Version 2." not in license_text:
        raise ValueError("Documentation license grant missing")
    if "Copyright © 2001-2023 Python Software Foundation; All Rights" not in license_text:
        raise ValueError("Required PSF copyright notice missing")
    if "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" not in (V1 / "raw/LICENSE").read_text():
        raise ValueError("Root license notice missing")
    return True


def encode_windows(tokenizer, rows, sequence_length, max_windows):
    windows = []
    for row in rows:
        ids = tokenizer.encode(row["text"], bos=True, eos=True)
        # Windows never cross document boundaries; the last short tail is held out.
        for offset in range(0, len(ids) - sequence_length, sequence_length):
            if len(windows) >= max_windows:
                return windows
            chunk = ids[offset:offset + sequence_length + 1]
            if len(chunk) == sequence_length + 1:
                windows.append({"document_id": row["document_id"], "offset": offset, "ids": chunk})
    return windows


def prepare():
    config = json.loads((V1 / "config.json").read_text())
    rights_gate(config)
    rows = build_rows(config)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if len(train) != 3 or len(validation) != 1 or {r["document_id"] for r in train} & {r["document_id"] for r in validation}:
        raise ValueError("Expected document-disjoint 3/1 split")
    tokenizer = ByteBPE.train([r["text"] for r in train], target_size=config["tokenizer_target_size"])
    tokenizer_data = {"schema_version": 1, "type": "byte_bpe_from_cpython_3_12_15_train_only",
                      "target_size": config["tokenizer_target_size"], "vocab_size": tokenizer.vocab_size,
                      "training_document_ids": [r["document_id"] for r in train],
                      "training_text_sha256": {r["document_id"]: r["text_sha256"] for r in train},
                      "special_ids": {"pad": 0, "bos": 1, "eos": 2, "sep": 3}, "merges": tokenizer.merges}
    for split, items in (("train", train), ("validation", validation)):
        with (V1 / f"{split}.jsonl").open("x") as output:
            for row in items:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (V1 / "tokenizer.json").write_text(json.dumps(tokenizer_data, ensure_ascii=False, separators=(",", ":")) + "\n")
    metrics = {}
    for split, items in (("train", train), ("validation", validation)):
        chars = sum(len(r["text"]) for r in items)
        bytes_ = sum(len(r["text"].encode()) for r in items)
        ids = [tokenizer.encode(r["text"]) for r in items]
        tokens = sum(len(x) for x in ids)
        used = {t for seq in ids for t in seq}
        if any(tokenizer.decode(seq) != r["text"] for seq, r in zip(ids, items)):
            raise ValueError("Tokenizer roundtrip failed")
        metrics[split] = {"documents": len(items), "paragraphs": sum(len(r["paragraphs"]) for r in items),
                          "characters": chars, "utf8_bytes": bytes_, "tokens": tokens,
                          "tokens_per_character": tokens / chars, "bytes_per_token": bytes_ / tokens,
                          "unknown_token_rate": 0.0, "used_vocabulary_ids": len(used),
                          "used_learned_merge_ids": len([x for x in used if x >= 260]),
                          "document_ids": [r["document_id"] for r in items]}
    train_windows = encode_windows(tokenizer, train, config["sequence_length"], config["max_train_windows"])
    val_windows = encode_windows(tokenizer, validation, config["sequence_length"], config["max_validation_windows"])
    if len(train_windows) < 4 or len(val_windows) < 2:
        raise ValueError("Too few fixed-sequence train/validation windows")
    (V1 / "tokenizer_evaluation.json").write_text(json.dumps({"vocab_size": tokenizer.vocab_size,
        "byte_fallback_covers_all_utf8_bytes": True, "validation_used_for_fit": False,
        "split_metrics": metrics, "fixed_sequence_windows": {"train": len(train_windows), "validation": len(val_windows)}}, indent=2) + "\n")
    manifest = {"schema_version": 1, "official_source": "https://github.com/python/cpython",
                "tag": config["source_tag"], "commit": config["source_commit"],
                "tag_date_utc": "2026-09-30T22:04:19Z", "retrieved_utc": "2026-10-12T04:55:33Z",
                "license": "Python Software Foundation License Version 2",
                "license_url": f"https://github.com/python/cpython/blob/{config['source_commit']}/Doc/license.rst",
                "rights_status": "official_documentation_prose_only_verified_with_pinned_license",
                "attribution": "Python Software Foundation; page authors recorded in pinned RST source headers",
                "required_notice": "Copyright © 2001-2023 Python Software Foundation; All Rights Reserved",
                "required_terms": "Retain PSF License Version 2 and copyright notice; describe derivative changes when distributed.",
                "third_party_policy": "Reviewed first-party prose spans only; code examples, tables, seealso references, external linked recipes, and figures excluded. No claim about excluded third-party content.",
                "processing": "Explicit audited source-line spans; strip RST inline markup; join paragraphs per document; exact paragraph deduplication; split by source file before tokenizer fit and windowing; drop short final window.",
                "split_document_text_sha256": {s: {r["document_id"]: r["text_sha256"] for r in items} for s, items in (("train", train), ("validation", validation))},
                "source_sha256": config["expected_source_sha256"], "stats": metrics,
                "fixed_sequence_windows": {"train": len(train_windows), "validation": len(val_windows)},
                "file_sha256": {name: sha256(V1 / name) for name in ("config.json", "train.jsonl", "validation.jsonl", "tokenizer.json", "tokenizer_evaluation.json")}}
    (V1 / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"stats": metrics, "windows": manifest["fixed_sequence_windows"], "vocab_size": tokenizer.vocab_size}, indent=2))


def verify():
    config = json.loads((V1 / "config.json").read_text())
    rights_gate(config)
    rows = build_rows(config)
    manifest = json.loads((V1 / "manifest.json").read_text())
    tokdata = json.loads((V1 / "tokenizer.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha256(V1 / name) != expected:
            raise ValueError(f"Corpus artifact differs: {name}")
    for split in ("train", "validation"):
        expected_rows = [row for row in rows if row["split"] == split]
        saved_rows = [json.loads(line) for line in (V1 / f"{split}.jsonl").read_text().splitlines()]
        if expected_rows != saved_rows:
            raise ValueError(f"{split} rows differ from pinned source spans")
    train = [row for row in rows if row["split"] == "train"]
    rebuilt = ByteBPE.train([row["text"] for row in train], config["tokenizer_target_size"])
    if [list(merge) for merge in rebuilt.merges] != tokdata["merges"]:
        raise ValueError("Tokenizer merges differ from training-only rebuild")
    return {"verified": True, "documents": len(rows), "vocab_size": rebuilt.vocab_size,
            "manifest_sha256": sha256(V1 / "manifest.json")}


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify"]:
        print(json.dumps(verify(), indent=2))
    elif len(sys.argv) == 1:
        prepare()
    else:
        raise SystemExit("Usage: prepare.py [--verify]")
