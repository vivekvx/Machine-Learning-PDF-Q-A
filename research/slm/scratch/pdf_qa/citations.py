"""Source-ID syntax checks shared by the web and isolated MLX runtimes."""
import re

EVIDENCE_ID = re.compile(r"\[E(\d+)\]")


def validate_citations(answer: str, sources: list[dict]):
    valid = {s["id"] for s in sources}
    cited = {f"E{x}" for x in EVIDENCE_ID.findall(answer)}
    malformed = re.findall(r"\[(?:E|page|p\.)[^\]]*\]", answer, flags=re.I)
    unknown = sorted(cited - valid)
    return {"valid": not unknown and not any(m not in {f"[{x}]" for x in valid} for m in malformed),
            "cited_ids": sorted(cited & valid), "unknown_ids": unknown,
            "has_citation": bool(cited & valid)}
