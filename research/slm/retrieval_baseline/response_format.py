"""Strict output contract for the non-thinking Qwen grounded-QA prompt."""
import re

FINAL_CITATION = re.compile(r"(?s)^\S.*\s\[section:\d+:paragraph:\d+\]$")
CONTROL = re.compile(r"<\|(?:im_start|im_end|endoftext|fim_suffix)\|>|</?think>|`{3}")


def parse_response(raw, finish_reason):
    if not isinstance(raw, str):
        raise TypeError("Generated response must be text")
    normalized = raw.strip()
    truncated = finish_reason == "length"
    if finish_reason not in ("stop", "length"):
        status = "unknown_generation_stop"
    elif not normalized:
        status = "empty_generation"
    elif CONTROL.search(normalized):
        status = "control_or_thinking_token"
    elif normalized == "UNANSWERABLE":
        status = "exact_abstention"
    elif normalized.casefold() == "unanswerable" or normalized.startswith("UNANSWERABLE"):
        status = "malformed_abstention"
    elif FINAL_CITATION.fullmatch(normalized):
        status = "cited_answer"
    elif "[section:" in normalized:
        status = "malformed_citation"
    else:
        status = "uncited_answer"
    return {"normalized_response": normalized, "format_status": status,
            "generation_truncated": truncated, "exact_unanswerable": status == "exact_abstention"}
