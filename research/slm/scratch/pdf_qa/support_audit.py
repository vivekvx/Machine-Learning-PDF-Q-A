"""Conservative lexical citation warnings; never an entailment verifier."""
import re

STOP = {"a", "an", "the", "of", "to", "for", "in", "on", "it", "is", "was", "are",
        "and", "or", "with", "from", "that", "this", "as", "at", "by", "used", "use"}


def words(text):
    return {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?", text)
            if x.lower() not in STOP and len(x) > 1}


def audit(answer: str, sources: list[dict]):
    lookup = {s["id"]: s["text"] for s in sources}
    warnings = []
    for line in answer.splitlines():
        ids = re.findall(r"\[E\d+\]", line)
        if not ids:
            continue
        claim = re.sub(r"\[E\d+\]", "", line).strip(" -:.;")
        claim_words = words(claim)
        if not claim_words:
            continue
        cited_ids = [token[1:-1] for token in ids]
        # A paragraph may cite several passages jointly. Scoring the whole
        # paragraph against each passage separately creates false warnings.
        source_words = set().union(*(words(lookup.get(source_id, "")) for source_id in cited_ids))
        overlap = len(claim_words & source_words) / len(claim_words)
        if overlap < 0.55:
            warnings.append({"source_ids": cited_ids, "claim": claim[:220],
                             "lexical_overlap": round(overlap, 3),
                             "reason": "low_claim_combined_source_word_overlap_manual_check_required"})
        if len(cited_ids) > 1 and re.search(r"[.!?]\s+\S", claim):
            warnings.append({"source_ids": cited_ids, "claim": claim[:220],
                             "reason": "pooled_citations_do_not_identify_per_sentence_support"})
    return {"method": "automatic_lexical_warning_not_semantic_verification",
            "warnings": warnings, "requires_manual_support_review": bool(warnings)}
