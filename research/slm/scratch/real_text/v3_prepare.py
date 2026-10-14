"""Build a document-grouped, rights-audited CPython v3 corpus from a pinned archive."""
import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "v3"
ARCHIVE = V3 / "source/cpython-e848e4b.tar.gz"
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from expanded_prepare import clean_v2, source_authors  # noqa: E402
from v3_tokenizer import V3ByteBPE, SPECIAL  # noqa: E402

MEDICAL = re.compile(r"\b(?:medical|biomedical|clinical|patient|pharmaceutical|healthcare|medicine|disease|diagnosis|drug)\b", re.I)
RIGHTS_NOTICE = re.compile(r"\b(?:copyright|licen[cs]e|adapted from|reproduced from|quoted from|permission to reproduce|all rights reserved)\b", re.I)
EXTERNAL = re.compile(r"https?://|www\.|\.\.\s|::|>>>|\|", re.I)
DETACHED_REFERENCE = re.compile(r"\b(?:below|above|following|preceding)\b.{0,100}\b(?:table|example|diagram|list)\b|\bhere is an example\b|\b(?:first|second|previous|following|below|above|this|the)\s+examples?\b|\bsee (?:the )?(?:section|table|example|below|above)\b", re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_archive(config):
    if sha(ARCHIVE.read_bytes()) != config["source_archive_sha256"]:
        raise ValueError("Official source archive hash mismatch")
    files = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "/" not in member.name:
                continue
            path = member.name.split("/", 1)[1]
            if path in ("LICENSE", "Doc/license.rst") or (path.startswith("Doc/") and path.endswith(".rst") and
                    path.split("/")[1] in config["allowed_document_sections"]):
                files[path] = archive.extractfile(member).read()
    if not {"LICENSE", "Doc/license.rst"} <= files.keys():
        raise ValueError("Pinned license files missing")
    license_text = files["Doc/license.rst"].decode("utf-8")
    if "Python software and documentation are licensed under the\nPython Software Foundation License Version 2." not in license_text:
        raise ValueError("Documentation license statement missing")
    if "Copyright © 2001-2023 Python Software Foundation; All Rights" not in license_text:
        raise ValueError("Required documentation copyright notice missing")
    if "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2" not in files["LICENSE"].decode("utf-8"):
        raise ValueError("Full PSF License Version 2 missing")
    return files


def paragraphs(lines, config):
    selected = []
    counts = Counter()
    start = None
    for index in range(len(lines) + 1):
        line = lines[index] if index < len(lines) else ""
        if line.strip():
            if start is None:
                start = index + 1
            continue
        if start is None:
            continue
        first, last = start, index
        block = lines[first - 1:last]
        start = None
        if any(item[:1].isspace() for item in block):
            counts["indented_or_code"] += 1
            continue
        raw = " ".join(item.strip() for item in block)
        if not (config["minimum_clean_paragraph_characters"] <= len(raw) <= config["maximum_clean_paragraph_characters"] and
                raw.endswith((".", "?", "!"))):
            counts["not_prose_shape"] += 1
            continue
        if EXTERNAL.search(raw) or '"' in raw:
            counts["external_code_or_quotation"] += 1
            continue
        if RIGHTS_NOTICE.search(raw):
            counts["rights_notice_paragraph"] += 1
            continue
        if DETACHED_REFERENCE.search(raw):
            counts["detached_table_example_or_navigation"] += 1
            continue
        try:
            clean = clean_v2(lines, first, last)
        except ValueError:
            counts["unclean_rst"] += 1
            continue
        if MEDICAL.search(clean):
            counts["medical_keyword_paragraph"] += 1
            continue
        selected.append({"source_lines": [first, last], "text": clean, "sha256": sha(clean.encode("utf-8"))})
    return selected, counts


def page_family(path):
    section = path.split("/")[1]
    if section == "tutorial":
        return "Doc/tutorial/all-chapters"
    suffix = path.split("/", 2)[2]
    if "/" in suffix:
        return f"Doc/{section}/{suffix.split('/')[0]}"
    stem = Path(suffix).stem
    if section == "reference":
        return f"Doc/reference/{stem}"
    return f"Doc/library/{re.split(r'[.\-_]', stem, maxsplit=1)[0]}"


def normalize(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def shingles(text, size):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left, right):
        a, b = self.root(left), self.root(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def similar_pairs(paragraphs_, policy):
    """Candidate pairs share >=5 rare word-5-grams; exact matches are always tested."""
    exact = defaultdict(list)
    inverted = defaultdict(list)
    signatures = []
    for index, item in enumerate(paragraphs_):
        exact[normalize(item["text"])].append(index)
        signature = shingles(item["text"], policy["near_duplicate_word_shingle_size"])
        signatures.append(signature)
        for shingle in signature:
            inverted[shingle].append(index)
    candidates = set()
    for group in exact.values():
        if len(group) > 1:
            candidates.update((group[0], later) for later in group[1:])
    for group in inverted.values():
        if len(group) <= 30:
            candidates.update((a, b) for position, a in enumerate(group) for b in group[position + 1:])
    matches = []
    for a, b in sorted(candidates):
        if paragraphs_[a]["document_id"] == paragraphs_[b]["document_id"]:
            continue
        if normalize(paragraphs_[a]["text"]) == normalize(paragraphs_[b]["text"]):
            matches.append((a, b, 1.0, "exact"))
            continue
        overlap = len(signatures[a] & signatures[b])
        if overlap < policy["near_duplicate_minimum_shared_shingles"]:
            continue
        score = overlap / len(signatures[a] | signatures[b])
        if score >= policy["near_duplicate_jaccard_threshold"]:
            matches.append((a, b, score, "near"))
    return matches


def build_candidates(files, config):
    rows = []
    source_ledger = []
    excluded = Counter()
    paragraph_rejections = Counter()
    for path, raw in sorted(files.items()):
        if path in ("LICENSE", "Doc/license.rst"):
            continue
        source_hash = sha(raw)
        lines = raw.decode("utf-8", errors="strict").splitlines()
        stem = Path(path).stem
        reason = None
        if stem in config["excluded_source_stems"]:
            reason = "conservative external-specification_or_third-party-risk_stem"
        elif MEDICAL.search(stem.replace("-", " ").replace("_", " ")):
            reason = "medical_title"
        elif any(RIGHTS_NOTICE.search(line) for line in lines[:20]):
            reason = "explicit_rights_or_reproduction_notice_in_header"
        if reason:
            excluded[reason] += 1
            source_ledger.append({"path": path, "source_sha256": source_hash, "decision": "excluded", "reason": reason})
            continue
        paras, rejected = paragraphs(lines, config)
        paragraph_rejections.update(rejected)
        if len(paras) < 2:
            excluded["fewer_than_two_eligible_prose_paragraphs"] += 1
            source_ledger.append({"path": path, "source_sha256": source_hash, "decision": "excluded", "reason": "fewer_than_two_eligible_prose_paragraphs"})
            continue
        content = "\n\n".join(p["text"] for p in paras) + "\n"
        row = {"document_id": path, "source_family": page_family(path), "title": stem,
               "source_url": f"https://github.com/python/cpython/blob/{config['source_commit']}/{path}",
               "source_sha256": source_hash, "attribution_names_in_header": source_authors(lines),
               "paragraphs": paras, "text": content, "text_sha256": sha(content.encode("utf-8"))}
        rows.append(row)
        source_ledger.append({"path": path, "source_sha256": source_hash, "source_url": row["source_url"],
                              "decision": "included_candidate", "attribution_names_in_header": row["attribution_names_in_header"],
                              "license": "PSF License Version 2 for official CPython documentation", "source_family": row["source_family"]})
    return rows, source_ledger, excluded, paragraph_rejections


def group_and_split(rows, config):
    rows = sorted(rows, key=lambda row: row["document_id"])
    index_by_doc = {row["document_id"]: i for i, row in enumerate(rows)}
    union = UnionFind(len(rows))
    families = {}
    for i, row in enumerate(rows):
        if row["source_family"] in families:
            union.join(i, families[row["source_family"]])
        else:
            families[row["source_family"]] = i
    items = [{"document_id": row["document_id"], **p} for row in rows for p in row["paragraphs"]]
    matches = similar_pairs(items, config)
    for a, b, score, kind in matches:
        union.join(index_by_doc[items[a]["document_id"]], index_by_doc[items[b]["document_id"]])
    # Group almost-identical full pages even if paragraph boundaries differ.
    doc_signatures = [shingles(row["text"], config["near_duplicate_word_shingle_size"]) for row in rows]
    near_documents = []
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            a, b = doc_signatures[i], doc_signatures[j]
            shared = len(a & b)
            if shared < config["near_duplicate_minimum_shared_shingles"]:
                continue
            score = shared / len(a | b)
            if score >= config["near_duplicate_jaccard_threshold"]:
                union.join(i, j)
                near_documents.append({"document_ids": [left["document_id"], rows[j]["document_id"]], "jaccard": score})
    # Remove exact and near-duplicate prose before tokenization or token counting.
    duplicate_ids = {b for a, b, _, _ in matches}
    removed = Counter()
    cursor = 0
    retained = []
    for row in rows:
        kept = []
        for p in row["paragraphs"]:
            if cursor in duplicate_ids:
                removed["exact_or_near_duplicate_paragraph"] += 1
            else:
                kept.append(p)
            cursor += 1
        if len(kept) < 2:
            removed["document_after_dedup"] += 1
            continue
        row["paragraphs"] = kept
        row["text"] = "\n\n".join(p["text"] for p in kept) + "\n"
        row["text_sha256"] = sha(row["text"].encode("utf-8"))
        row["group_id"] = f"group-{union.root(index_by_doc[row['document_id']]):04d}"
        retained.append(row)
    groups = defaultdict(list)
    for row in retained:
        groups[row["group_id"]].append(row)
    total = sum(len(row["text"]) for row in retained)
    targets = {split: ratio * total for split, ratio in config["target_split_character_ratios"].items()}
    totals = Counter()
    allocation = {}
    # Assign largest related-page groups first, then stable hash tie-breaks.
    ordered = sorted(groups, key=lambda gid: (-sum(len(r["text"]) for r in groups[gid]),
                                              sha(f"{config['split_seed']}:{gid}".encode())))
    for gid in ordered:
        size = sum(len(r["text"]) for r in groups[gid])
        split = min(("train", "validation", "test"),
                    key=lambda s: ((totals[s] + size - targets[s]) ** 2 - (totals[s] - targets[s]) ** 2) / targets[s])
        allocation[gid] = split
        totals[split] += size
    for row in retained:
        row["split"] = allocation[row["group_id"]]
    if len({r["group_id"] for r in retained if r["split"] == "validation"}) < 2 or len({r["group_id"] for r in retained if r["split"] == "test"}) < 2:
        raise ValueError("Validation/test need multiple independent source groups")
    return retained, {"initial_candidate_documents": len(rows), "retained_documents": len(retained),
                      "family_groups": len(families), "joined_near_paragraph_pairs": len(matches),
                      "joined_near_document_pairs": near_documents,
                      "removed": dict(removed), "group_count": len(groups), "split_character_targets": targets,
                      "split_characters_actual": dict(totals), "group_assignment": allocation}


def leakage_audit(rows, config):
    items = [{"document_id": row["document_id"], "split": row["split"], **p}
             for row in rows for p in row["paragraphs"]]
    matches = similar_pairs(items, config)
    cross = [{"document_ids": [items[a]["document_id"], items[b]["document_id"]], "score": score, "kind": kind}
             for a, b, score, kind in matches if items[a]["split"] != items[b]["split"]]
    doc_signatures = [(row, shingles(row["text"], config["near_duplicate_word_shingle_size"])) for row in rows]
    near_docs = []
    for i, (a, sa) in enumerate(doc_signatures):
        for b, sb in doc_signatures[i + 1:]:
            if a["split"] == b["split"]:
                continue
            shared = len(sa & sb)
            if shared >= config["near_duplicate_minimum_shared_shingles"]:
                score = shared / len(sa | sb)
                if score >= config["near_duplicate_jaccard_threshold"]:
                    near_docs.append({"document_ids": [a["document_id"], b["document_id"]], "jaccard": score})
    if cross or near_docs:
        raise ValueError("Cross-split duplicate or near-duplicate leakage")
    return {"cross_split_paragraph_matches": cross, "cross_split_near_documents": near_docs,
            "threshold": config["near_duplicate_jaccard_threshold"],
            "word_shingle_size": config["near_duplicate_word_shingle_size"],
            "limitation": "Lexical overlap checks do not prove absence of paraphrases or shared facts."}


def prepare():
    config = json.loads((V3 / "config.json").read_text())
    if config["source_tag"] != "v3.12.15" or config["source_commit"] != "e848e4b09ca4bc67c0041c9ce7526dae471d4c36":
        raise ValueError("Official source pin changed")
    files = read_archive(config)
    rows, source_ledger, exclusions, paragraph_rejections = build_candidates(files, config)
    rows, grouping = group_and_split(rows, config)
    leakage = leakage_audit(rows, config)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    tokenizer = V3ByteBPE.train([row["text"] for row in train], config["tokenizer_target_size"])
    if tokenizer.vocab_size != 1024:
        raise ValueError("Tokenizer vocabulary size changed")
    stats = {}
    for split, subset in (("train", train), ("validation", val), ("test", test)):
        token_count = 0
        used = set()
        for row in subset:
            encoded = tokenizer.encode(row["text"])
            if tokenizer.decode(encoded) != row["text"]:
                raise ValueError("Tokenizer roundtrip failed")
            token_count += len(encoded)
            used.update(encoded)
        bytes_ = sum(len(row["text"].encode()) for row in subset)
        stats[split] = {"documents": len(subset), "source_groups": len({r["group_id"] for r in subset}),
                        "paragraphs": sum(len(r["paragraphs"]) for r in subset), "utf8_bytes": bytes_,
                        "unique_text_tokens": token_count, "bytes_per_token": bytes_ / token_count,
                        "unknown_token_rate": 0.0, "used_vocabulary_ids": len(used),
                        "sections": dict(Counter(r["document_id"].split("/")[1] for r in subset))}
    if stats["train"]["unique_text_tokens"] < config["min_unique_train_tokens_for_pilot"]:
        raise ValueError("Actual unique training tokens below preregistered 100k gate; training must stop")
    if stats["train"]["unique_text_tokens"] > config["target_max_unique_train_tokens"]:
        raise ValueError("Training corpus exceeds preregistered 300k target; review selection before training")
    outputs = ["train.jsonl", "validation.jsonl", "test.sealed.jsonl", "tokenizer.json", "tokenizer_evaluation.json",
               "leakage_audit.json", "source_ledger.json", "grouping_audit.json", "manifest.json", "LICENSE", "Doc_license.rst"]
    if any((V3 / name).exists() for name in outputs):
        raise FileExistsError("v3 corpus outputs already exist")
    for split, subset in (("train", train), ("validation", val), ("test.sealed", test)):
        with (V3 / f"{split}.jsonl").open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (V3 / "LICENSE").write_bytes(files["LICENSE"])
    (V3 / "Doc_license.rst").write_bytes(files["Doc/license.rst"])
    tokdata = {"schema_version": 3, "type": "deterministic_whitespace_piece_byte_bpe_from_v3_train_only",
               "target_size": config["tokenizer_target_size"], "vocab_size": tokenizer.vocab_size,
               "training_document_ids": [r["document_id"] for r in train],
               "training_text_sha256": {r["document_id"]: r["text_sha256"] for r in train},
               "special_ids": SPECIAL, "merges": tokenizer.merges}
    (V3 / "tokenizer.json").write_text(json.dumps(tokdata, separators=(",", ":"), ensure_ascii=False) + "\n")
    (V3 / "tokenizer_evaluation.json").write_text(json.dumps({"fit_split": "train", "validation_or_test_used_for_fit": False,
                                                       "stats": stats}, indent=2) + "\n")
    (V3 / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n")
    (V3 / "source_ledger.json").write_text(json.dumps({"entries": source_ledger, "excluded_source_counts": dict(exclusions),
                                                    "rejected_paragraph_counts": dict(paragraph_rejections)}, indent=2) + "\n")
    (V3 / "grouping_audit.json").write_text(json.dumps(grouping, indent=2, sort_keys=True) + "\n")
    hashes = {name: sha((V3 / name).read_bytes()) for name in outputs if name != "manifest.json"}
    manifest = {"schema_version": 3, "source_repository": config["source_repository"], "tag": config["source_tag"],
                "tag_object": config["source_tag_object"], "commit": config["source_commit"],
                "archive_url": config["source_archive_url"], "archive_sha256": config["source_archive_sha256"],
                "retrieved_date_utc": datetime.now(timezone.utc).date().isoformat(),
                "license": "Python Software Foundation License Version 2", "license_source_url":
                f"https://github.com/python/cpython/blob/{config['source_commit']}/Doc/license.rst",
                "required_notice": "Copyright © 2001-2023 Python Software Foundation; All Rights Reserved",
                "required_terms": "Retain full PSF License Version 2 and copyright notice; describe derivative changes on redistribution.",
                "attribution": "Python Software Foundation; source-specific header authors in source_ledger.json",
                "processing": "Conservative unindented prose paragraphs only; reject directives, code, examples, links, tables, quotations, medical and explicit rights-notice paragraphs; normalize inline RST; group related and near-duplicate pages; remove duplicate paragraphs; split before train-only byte BPE fitting and fixed-window construction.",
                "rights_limit": "Official repository documentation license verified; no independent chain-of-title audit of each contributor. Linked works, code examples, figures, and third-party notices excluded.",
                "config_sha256": sha((V3 / "config.json").read_bytes()),
                "source_sha256": {entry["path"]: entry["source_sha256"] for entry in source_ledger if entry["decision"] == "included_candidate"},
                "split_document_ids": {split: [r["document_id"] for r in subset] for split, subset in (("train", train), ("validation", val), ("test", test))},
                "split_document_text_sha256": {split: {r["document_id"]: r["text_sha256"] for r in subset}
                                               for split, subset in (("train", train), ("validation", val), ("test", test))},
                "split_stats": stats, "group_count": grouping["group_count"],
                "test_sealed_for_training_and_model_selection": True, "test_text_displayed": False,
                "file_sha256": hashes}
    (V3 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return {"stats": stats, "source_documents": len(rows), "source_groups": grouping["group_count"],
            "excluded_sources": dict(exclusions), "rejected_paragraphs": dict(paragraph_rejections)}


def verify():
    config = json.loads((V3 / "config.json").read_text())
    read_archive(config)
    manifest = json.loads((V3 / "manifest.json").read_text())
    if sha((V3 / "config.json").read_bytes()) != manifest["config_sha256"]:
        raise ValueError("v3 config changed")
    for name, expected in manifest["file_sha256"].items():
        if sha((V3 / name).read_bytes()) != expected:
            raise ValueError(f"v3 artifact hash mismatch: {name}")
    return {"status": "verified", "stats": manifest["split_stats"], "sealed_test_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.verify:
        parser.error("Choose exactly one of --prepare or --verify")
    print(json.dumps(prepare() if args.prepare else verify(), indent=2))
