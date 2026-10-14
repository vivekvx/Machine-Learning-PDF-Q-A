"""Stream the official ACL Anthology BibTeX export; conservative title matching.

The export is read from a caller supplied local .bib.gz file. No full corpus is
retained and no fuzzy match is silently accepted as a publication identity.
"""
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from review_core import SLM, sha256

BIB_URL = "https://aclanthology.org/anthology.bib.gz"
POLICY_URL = "https://aclanthology.org/faq/copyright/"
SELECTED = SLM / "training_data_v2/v1/selected_audit.jsonl"
ACL_PUBLISHER = "Association for Computational Linguistics"
FIELD = re.compile(r"^\s{4}([a-z_]+)\s*=\s*", re.M)
ENTRY = re.compile(r"^@([a-z]+)\{([^,]+),")


def normalize_title(text):
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def field_value(block, name):
    matches = list(FIELD.finditer(block))
    for i, match in enumerate(matches):
        if match.group(1) != name:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else block.rfind("\n}")
        value = block[match.end():end].strip().rstrip(",").strip()
        if len(value) >= 2 and (value[0], value[-1]) in (("\"", "\""), ("{", "}")):
            value = value[1:-1]
        return " ".join(value.split())
    return None


def entries(bib_path):
    with gzip.open(bib_path, "rt", encoding="utf-8", errors="replace") as stream:
        block = []
        for line in stream:
            if line.startswith("@"):
                if block:
                    yield "".join(block)
                block = [line]
            elif block:
                block.append(line)
        if block:
            yield "".join(block)


def parse_entry(block):
    first = ENTRY.match(block)
    if not first or first.group(1) not in ("inproceedings", "article", "incollection", "misc"):
        return None
    entry = {"acl_bib_key": first.group(2), "entry_type": first.group(1)}
    for field in ("title", "author", "year", "publisher", "url", "doi", "booktitle"):
        entry[field] = field_value(block, field)
    if not entry["title"] or not entry["url"] or not entry["url"].startswith("https://aclanthology.org/"):
        return None
    entry["normalized_title"] = normalize_title(entry["title"])
    entry["acl_id"] = entry["url"].rstrip("/").rsplit("/", 1)[-1]
    return entry


def acl_license(entry):
    year = int(entry["year"]) if entry["year"] and entry["year"].isdigit() else None
    if entry["publisher"] != ACL_PUBLISHER or year is None:
        return ("unresolved", None, "ACL policy does not establish rights for this publisher or missing year")
    if year >= 2016:
        return ("verified_permissive_acl_publication", "CC BY 4.0", "ACL policy covers ACL materials from 2016; third-party material and source-version equivalence remain unverified")
    return ("restrictive_acl_publication", "CC BY-NC-SA 3.0", "ACL policy covers pre-2016 ACL materials; noncommercial/share-alike restrictions require separate review")


def resolve(selected, bib_path):
    """Return one ledger row per selected training document, in frozen order."""
    wanted = {normalize_title(r["title"]) for r in selected}
    exact = defaultdict(list)
    # Compact near-match index for unresolved titles; never auto-approve it.
    candidate_titles = defaultdict(list)
    for block in entries(bib_path):
        entry = parse_entry(block)
        if not entry:
            continue
        key = entry["normalized_title"]
        if key in wanted:
            exact[key].append(entry)
        else:
            words = key.split()
            if words:
                candidate_titles[words[0]].append((key, entry["acl_id"], entry["url"]))
    ledger = []
    for row in selected:
        title_key = normalize_title(row["title"])
        matches = exact.get(title_key, [])
        matched = matches[0] if len(matches) == 1 else None
        if matched:
            license_class, license_name, scope = acl_license(matched)
            method, confidence = "unique_normalized_exact_title", "high_title_match_requires_source_version_check"
            nearby = []
        else:
            license_class, license_name, scope = "unresolved", None, "No unique official title match; do not infer publication rights"
            method, confidence = ("ambiguous_exact_title", "ambiguous") if matches else ("no_exact_title_match", "unresolved")
            first = title_key.split()[0] if title_key else ""
            nearby = sorted(((SequenceMatcher(None, title_key, key, autojunk=False).ratio(), acl_id, url)
                             for key, acl_id, url in candidate_titles.get(first, [])
                             if abs(len(title_key) - len(key)) <= max(len(title_key), len(key)) * .12), reverse=True)[:3]
            nearby = [{"acl_id": acl_id, "url": url, "title_similarity": round(score, 4)}
                      for score, acl_id, url in nearby if score >= .88]
        ledger.append({"document_id": row["document_id"], "question_id": row["question_id"], "source_split": "train",
                       "qasper_title": row["title"], "qasper_source_revision": row["provenance"]["source_revision"],
                       "match_method": method, "match_confidence": confidence,
                       "exact_match_count": len(matches), "nearby_unaccepted_candidates": nearby,
                       "acl_id": matched["acl_id"] if matched else None,
                       "acl_bib_key": matched["acl_bib_key"] if matched else None,
                       "source_url": matched["url"] if matched else None,
                       "publication_year": int(matched["year"]) if matched and matched["year"] and matched["year"].isdigit() else None,
                       "publisher": matched["publisher"] if matched else None,
                       "author_attribution": matched["author"] if matched else None,
                       "publication_title": matched["title"] if matched else None,
                       "venue": matched["booktitle"] if matched else None,
                       "doi": matched["doi"] if matched else None,
                       "acl_publication_license_class": license_class,
                       "acl_publication_license": license_name,
                       "license_policy_url": POLICY_URL if matched else None,
                       "license_scope_note": scope,
                       "qasper_text_training_rights": "unresolved_pending_source_version_and_third_party_review",
                       "human_rights_decision": None,
                       "acl_export_url": BIB_URL,
                       "acl_export_sha256": sha256(bib_path)})
    return ledger


def load_selected(path=SELECTED):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 352 or len({row["document_id"] for row in rows}) != 352 or any(row["split"] != "train" for row in rows):
        raise ValueError("Frozen train cohort drift")
    return rows
