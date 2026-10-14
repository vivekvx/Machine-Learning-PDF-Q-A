"""Pinned CPython v2 corpus: document splits, rights, leakage, train-only BPE."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "v2"
sys.path.insert(0, str(ROOT.parent))
from tokenizer import ByteBPE  # noqa: E402
from prepare import clean_span, sha256  # noqa: E402

MEDICAL = re.compile(r"\b(?:medical|biomedical|clinical|patient|pharmaceutical|healthcare|medicine|disease|diagnosis|drug)\b", re.I)


def clean_v2(lines, first, last):
    text = clean_span(lines, first, last)
    # Sphinx role labels such as `pure paths <pure-paths>` are navigation syntax;
    # keep the visible label and remove the destination identifier.
    text = re.sub(r"\s<[^<>]+>", "", text)
    if "<" in text or ">" in text:
        # Comparison operators in prose/code literals are retained deliberately.
        if not re.search(r"(?:<=|>=| < | > )", text):
            raise ValueError("Unresolved RST role target")
    return text


def rights_gate(config, *, include_test_sources=True):
    if config["source_tag"] != "v3.12.15" or config["source_commit"] != "e848e4b09ca4bc67c0041c9ce7526dae471d4c36":
        raise ValueError("Official CPython revision differs from v2 pin")
    expected_paths = {"LICENSE", "Doc/license.rst", *config["source_documents"]}
    if set(config["expected_source_sha256"]) != expected_paths:
        raise ValueError("A selected source lacks a pinned hash")
    for name, expected in config["expected_source_sha256"].items():
        if not include_test_sources and name in config["source_documents"] and config["source_documents"][name]["split"] == "test":
            continue
        if sha256(V2 / "raw" / name) != expected:
            raise ValueError(f"Pinned source mismatch: {name}")
    license_text = (V2 / "raw/Doc/license.rst").read_text()
    if "Python software and documentation are licensed under the\nPython Software Foundation License Version 2." not in license_text:
        raise ValueError("Official documentation license statement missing")
    if "Copyright © 2001-2023 Python Software Foundation; All Rights" not in license_text:
        raise ValueError("Required copyright notice missing")
    if "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" not in (V2 / "raw/LICENSE").read_text():
        raise ValueError("Complete PSF license missing")


def sealed_spans(lines, selector):
    """Only source lines/structure are used to select held-out prose; no printed text."""
    groups = []
    start = None
    for index, line in enumerate(lines, 1):
        if line.strip():
            if start is None:
                start = index
        elif start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(lines)))
    chosen = []
    for first, last in groups:
        if first < selector["min_line"]:
            continue
        block = lines[first - 1:last]
        # Only top-level paragraphs, never directive bodies, examples, tables, or links.
        if any(line.startswith(" ") or line.startswith("\t") for line in block):
            continue
        raw = " ".join(line.strip() for line in block)
        if not (selector["min_characters"] <= len(raw) <= selector["max_characters"]):
            continue
        if not raw.endswith((".", "?", "!")) or any(mark in raw for mark in (".. ", "::", "http", ">>>", "|", "`_", '"')):
            continue
        try:
            cleaned = clean_v2(lines, first, last)
        except ValueError:
            continue
        if MEDICAL.search(cleaned):
            continue
        chosen.append((first, last))
        if len(chosen) == selector["max_paragraphs_per_document"]:
            break
    return chosen


def source_authors(lines):
    names = []
    for line in lines[:35]:
        if ".. moduleauthor::" in line or ".. sectionauthor::" in line:
            name = line.split("::", 1)[1].strip().split("<", 1)[0].strip()
            if name and name not in names:
                names.append(name)
    return names


def build_rows(config):
    rows = []
    for path, selection in config["source_documents"].items():
        lines = (V2 / "raw" / path).read_text(encoding="utf-8").splitlines()
        if selection["split"] == "test":
            if selection.get("sealed_selector") != "first_clean_paragraphs":
                raise ValueError("Unexpected sealed test selection method")
            spans = sealed_spans(lines, config["sealed_test_selector"])
        else:
            spans = selection["spans"]
        if len(spans) < 2:
            raise ValueError(f"Too few rights-reviewed prose spans in {path}")
        paragraphs = []
        for first, last in spans:
            prose = clean_v2(lines, first, last)
            if MEDICAL.search(prose):
                raise ValueError(f"Medical content signal in {path}:{first}-{last}")
            paragraphs.append({"source_lines": [first, last], "text": prose,
                               "sha256": hashlib.sha256(prose.encode()).hexdigest()})
        content = "\n\n".join(p["text"] for p in paragraphs) + "\n"
        rows.append({"document_id": path, "split": selection["split"], "domain": selection["domain"],
                     "title": path.rsplit("/", 1)[-1].removesuffix(".rst"),
                     "source_url": f"https://github.com/python/cpython/blob/{config['source_commit']}/{path}",
                     "source_sha256": config["expected_source_sha256"][path],
                     "attribution_names_in_header": source_authors(lines),
                     "selection_method": "sealed_first_clean_paragraphs" if selection["split"] == "test" else "explicit_reviewed_line_spans",
                     "paragraphs": paragraphs, "text": content,
                     "text_sha256": hashlib.sha256(content.encode()).hexdigest()})
    if len(rows) != 12 or Counter(r["split"] for r in rows) != {"train": 6, "validation": 3, "test": 3}:
        raise ValueError("Expected 6/3/3 document-level split")
    if len({r["document_id"] for r in rows}) != len(rows):
        raise ValueError("Document IDs overlap")
    return rows


def shingles(text, size):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def leakage_audit(rows, config):
    policy = config["leakage"]
    items = [(row, p) for row in rows for p in row["paragraphs"]]
    exact = []
    near = []
    max_cross_split = {"jaccard": 0.0, "document_ids": None, "source_lines": None}
    for i, (a, pa) in enumerate(items):
        na = re.sub(r"\W+", " ", pa["text"].casefold()).strip()
        sa = shingles(pa["text"], policy["word_shingle_size"])
        for b, pb in items[i + 1:]:
            if a["split"] == b["split"]:
                continue
            nb = re.sub(r"\W+", " ", pb["text"].casefold()).strip()
            if na == nb:
                exact.append([a["document_id"], b["document_id"]])
            sb = shingles(pb["text"], policy["word_shingle_size"])
            shared = len(sa & sb)
            score = shared / len(sa | sb) if sa | sb else 0.0
            if score > max_cross_split["jaccard"]:
                max_cross_split = {"jaccard": score, "document_ids": [a["document_id"], b["document_id"]],
                                   "source_lines": [pa["source_lines"], pb["source_lines"]]}
            if shared >= policy["minimum_shared_shingles"] and score >= policy["near_duplicate_jaccard_threshold"]:
                near.append({"document_ids": [a["document_id"], b["document_id"]], "jaccard": score})
    # Whole-document signature catches full or almost-full copies even if span boundaries differ.
    document_pairs = []
    signatures = [(r, shingles(r["text"], policy["word_shingle_size"])) for r in rows]
    for i, (a, sa) in enumerate(signatures):
        for b, sb in signatures[i + 1:]:
            if a["split"] == b["split"]:
                continue
            score = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
            if score >= policy["near_duplicate_jaccard_threshold"]:
                document_pairs.append({"document_ids": [a["document_id"], b["document_id"]], "jaccard": score})
    result = {"algorithm": "normalized exact paragraphs plus word-5-gram Jaccard for cross-split paragraphs and documents",
              "policy": policy, "cross_split_exact_paragraph_pairs": exact,
              "cross_split_near_paragraph_pairs": near,
              "cross_split_near_document_pairs": document_pairs,
              "max_cross_split_paragraph_similarity": max_cross_split}
    if exact or near or document_pairs:
        raise ValueError("Cross-split exact or near-duplicate leakage detected")
    return result


def windows(tokenizer, rows, sequence_length, cap, *, cap_per_document=False):
    output = []
    for row in rows:
        ids = tokenizer.encode(row["text"], bos=True, eos=True)
        used = 0
        for offset in range(0, len(ids) - sequence_length, sequence_length):
            if used >= cap or (not cap_per_document and len(output) >= cap):
                break
            chunk = ids[offset:offset + sequence_length + 1]
            if len(chunk) == sequence_length + 1:
                output.append({"document_id": row["document_id"], "offset": offset, "ids": chunk})
                used += 1
    return output


def split_stats(rows, tokenizer, config):
    stats = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        encoded = [tokenizer.encode(r["text"]) for r in subset]
        for row, ids in zip(subset, encoded):
            if tokenizer.decode(ids) != row["text"]:
                raise ValueError("Tokenizer roundtrip failed")
        bytes_ = sum(len(r["text"].encode()) for r in subset)
        tokens = sum(map(len, encoded))
        stats[split] = {"documents": len(subset), "paragraphs": sum(len(r["paragraphs"]) for r in subset),
                        "utf8_bytes": bytes_, "tokens": tokens, "bytes_per_token": bytes_ / tokens,
                        "unknown_token_rate": 0.0, "used_vocabulary_ids": len({t for seq in encoded for t in seq}),
                        "domains": dict(Counter(r["domain"] for r in subset)),
                        "tokens_by_domain": {domain: sum(len(ids) for r, ids in zip(subset, encoded) if r["domain"] == domain)
                                             for domain in sorted({r["domain"] for r in subset})},
                        "document_ids": [r["document_id"] for r in subset]}
    return stats


def prepare():
    config = json.loads((V2 / "config.json").read_text())
    rights_gate(config)
    rows = build_rows(config)
    leakage = leakage_audit(rows, config)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    tokenizer = ByteBPE.train([r["text"] for r in train], config["tokenizer_target_size"])
    stats = split_stats(rows, tokenizer, config)
    train_windows = windows(tokenizer, train, config["sequence_length"], config["max_train_windows"])
    val_windows = windows(tokenizer, val, config["sequence_length"], config["max_validation_windows_per_document"], cap_per_document=True)
    val_by_doc = dict(Counter(w["document_id"] for w in val_windows))
    if len(train_windows) < 20 or len(val_by_doc) != 3 or min(val_by_doc.values()) < 2:
        raise ValueError("Insufficient fixed windows across independent documents")
    tok_record = {"schema_version": 2, "type": "byte_bpe_from_cpython_v2_train_documents_only",
                  "target_size": config["tokenizer_target_size"], "vocab_size": tokenizer.vocab_size,
                  "training_document_ids": [r["document_id"] for r in train],
                  "training_text_sha256": {r["document_id"]: r["text_sha256"] for r in train},
                  "special_ids": {"pad": 0, "bos": 1, "eos": 2, "sep": 3}, "merges": tokenizer.merges}
    for split in ("train", "validation", "test"):
        name = "test.sealed.jsonl" if split == "test" else f"{split}.jsonl"
        with (V2 / name).open("x") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (V2 / "tokenizer.json").open("x") as stream:
        json.dump(tok_record, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    quality = {"schema_version": 2, "checks": {"medical_keyword_hits": 0,
               "cross_split_exact_paragraph_pairs": len(leakage["cross_split_exact_paragraph_pairs"]),
               "cross_split_near_paragraph_pairs": len(leakage["cross_split_near_paragraph_pairs"]),
               "cross_split_near_document_pairs": len(leakage["cross_split_near_document_pairs"])},
               "samples": [{"split": r["split"], "document_id": r["document_id"],
                            "source_lines": p["source_lines"], "text": p["text"]}
                           for r in rows if r["split"] != "test"
                           for p in (r["paragraphs"][0], r["paragraphs"][-1])],
               "test_text_displayed": False,
               "limits": "Automated duplicate/keyword checks and selected prose samples do not establish comprehensive human quality or contributor-rights review."}
    (V2 / "quality_samples.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")
    (V2 / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n")
    (V2 / "tokenizer_evaluation.json").write_text(json.dumps({"schema_version": 2, "target_vocab": config["tokenizer_target_size"],
            "actual_vocab": tokenizer.vocab_size, "fit_split": "train", "validation_or_test_used_for_fit": False,
            "byte_fallback_unknown_rate": 0.0, "split_stats": stats}, indent=2) + "\n")
    produced = ["config.json", "train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json",
                "tokenizer_evaluation.json", "quality_samples.json", "leakage_audit.json"]
    manifest = {"schema_version": 2, "source_url": "https://github.com/python/cpython",
                "source_tag": config["source_tag"], "source_commit": config["source_commit"],
                "tag_date_utc": "2026-09-30T22:04:19Z", "retrieval_date_utc": "2026-10-12",
                "license": "Python Software Foundation License Version 2",
                "license_url": f"https://github.com/python/cpython/blob/{config['source_commit']}/Doc/license.rst",
                "required_notice": "Copyright © 2001-2023 Python Software Foundation; All Rights Reserved",
                "required_terms": "Retain full PSF License Version 2 and copyright notice; summarize derivative changes when distributed.",
                "transformation": "Selected reviewed prose spans for train/validation; conservative structure-only selector for sealed test; stripped RST inline markup; rejected examples/tables/links/medical terms; document split before tokenizer fit and windows.",
                "rights_scope": "Pinned official documentation prose only; no independent contributor-chain audit or inference about third-party linked material.",
                "source_sha256": config["expected_source_sha256"],
                "split_document_text_sha256": {s: {r["document_id"]: r["text_sha256"] for r in rows if r["split"] == s}
                                               for s in ("train", "validation", "test")},
                "split_stats": stats, "fixed_windows": {"train": len(train_windows), "validation": len(val_windows),
                                                        "validation_by_document": val_by_doc},
                "leakage_audit_sha256": sha256(V2 / "leakage_audit.json"),
                "test_sealed_for_training_and_model_selection": True,
                "test_access_policy": "Only corpus creation, aggregate statistics, source rights, and automated leakage checks may read test text; training/evaluation worker never opens test.sealed.jsonl or test raw sources.",
                "file_sha256": {name: sha256(V2 / name) for name in produced}}
    with (V2 / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return {"manifest_sha256": sha256(V2 / "manifest.json"), "split_stats": stats,
            "fixed_windows": manifest["fixed_windows"], "max_cross_split_similarity": leakage["max_cross_split_paragraph_similarity"]}


def verify():
    config = json.loads((V2 / "config.json").read_text())
    rights_gate(config)
    manifest = json.loads((V2 / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha256(V2 / name) != expected:
            raise ValueError(f"v2 corpus file hash mismatch: {name}")
    rows = build_rows(config)
    if leakage_audit(rows, config) != json.loads((V2 / "leakage_audit.json").read_text()):
        raise ValueError("Leakage audit differs")
    for split in ("train", "validation", "test"):
        file = V2 / ("test.sealed.jsonl" if split == "test" else f"{split}.jsonl")
        if [r for r in rows if r["split"] == split] != [json.loads(x) for x in file.read_text().splitlines()]:
            raise ValueError(f"{split} source extraction differs")
    train = [r for r in rows if r["split"] == "train"]
    tok = json.loads((V2 / "tokenizer.json").read_text())
    rebuilt = ByteBPE.train([r["text"] for r in train], config["tokenizer_target_size"])
    if [list(x) for x in rebuilt.merges] != tok["merges"]:
        raise ValueError("Train-only tokenizer cannot be reproduced")
    return {"verified": True, "manifest_sha256": sha256(V2 / "manifest.json"),
            "documents": len(rows), "vocabulary": rebuilt.vocab_size}


if __name__ == "__main__":
    if sys.argv[1:] == ["--verify"]:
        print(json.dumps(verify(), indent=2))
    elif len(sys.argv) == 1:
        print(json.dumps(prepare(), indent=2))
    else:
        raise SystemExit("Usage: expanded_prepare.py [--verify]")
