"""Local evidence planning for the optional Qwen and scratch-pointer modes.

This module never loads a model. PDF text is untrusted data, and source IDs are
assigned by this application rather than accepted from a generator.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .pipeline import PDFIndex, question_terms
from .citations import validate_citations

INTENTS = ("factual", "explanation", "comparison", "section_summary",
           "whole_paper_summary", "unsupported")
INSUFFICIENT = "The supplied document evidence is insufficient to answer this question."


def classify_intent(question: str) -> str:
    q = question.strip().lower()
    if not q:
        return "unsupported"
    if re.search(r"\b(?:tell me a joke|write (?:me )?a poem|current weather|stock price|translate this)\b", q):
        return "unsupported"
    if re.search(r"\b(?:summari[sz]e|summary|overview|main points)\b", q):
        if re.search(r"\b(?:methods?|methodology|results?|findings?|limitations?|discussion|introduction|abstract|conclusions?)\b", q):
            return "section_summary"
        return "whole_paper_summary"
    if re.search(r"\b(?:what is this paper about|what is the paper about|what problem|research question|main contribution)\b", q):
        return "section_summary"
    if re.search(r"\b(?:main|key) (?:experimental )?(?:results?|findings?)\b", q):
        return "section_summary"
    if re.search(r"\b(?:what|which) (?:methodology|methods?|approach|limitations?)\b", q):
        return "section_summary"
    if re.search(r"\b(?:compare|compared|comparing|comparison|differences?|versus|vs\.?|relative to|contrast)\b", q):
        return "comparison"
    if re.search(r"\b(?:explain|why|how does|how did|methodology|approach|limitations?|findings?|experiments?)\b", q):
        return "explanation"
    if len(question) > 300:
        return "unsupported"
    return "factual"


def ranked_chunks(index: PDFIndex, question: str, count: int = 6):
    started = time.monotonic()
    query = index.vectorizer.transform([question]).toarray().astype("float32")
    scores, indices = index.index.search(np.ascontiguousarray(query), len(index.chunks))
    body = {page.page_number for page in _main_body_pages(index)}
    ranked = [(index.chunks[int(i)], float(s)) for i, s in zip(indices[0], scores[0])
              if i >= 0 and index.chunks[int(i)].page_number in body
              and index.chunks[int(i)].metadata.get("section_heading", "").lower() != "references"]
    return ranked[:count], (time.monotonic() - started) * 1000


def _main_body_pages(index: PDFIndex):
    """Exclude clearly marked bibliography/appendix pages from first-pass answers.

    This does not delete or rewrite indexed text. A heading must occur near the
    beginning of a page, where it can mark the start of back matter.
    """
    pages = sorted(index.pages, key=lambda page: page.page_number)
    for position, page in enumerate(pages):
        if position < 2:
            continue
        opening = "\n".join(page.page_content.splitlines()[:8])
        if re.search(r"(?im)^(?:references|bibliography|appendix contents)\s*$", opening):
            return pages[:position]
    return pages


def _source(index: PDFIndex, chunk, score: float, number: int, char_limit: int = 420,
            question: str = ""):
    # The chunker records start_index. Confirm it against the extracted page.
    page = next(p for p in index.pages if p.page_number == chunk.page_number)
    hint = chunk.metadata.get("start_index", 0)
    pos = page.page_content.find(chunk.chunk_text, max(0, hint - 4))
    if pos < 0:
        pos = page.page_content.find(chunk.chunk_text)
    if pos < 0:
        raise ValueError("Evidence chunk cannot be traced to the extracted PDF page")
    offset = _best_window_start(chunk.chunk_text, question, char_limit) if question else 0
    end = _clean_excerpt_end(chunk.chunk_text, offset, min(len(chunk.chunk_text), offset + char_limit))
    excerpt = chunk.chunk_text[offset:end]
    if page.page_content[pos + offset:pos + offset + len(excerpt)] != excerpt:
        raise ValueError("Evidence excerpt differs from extracted page text")
    return {"id": f"E{number}", "document_id": index.document_id,
            "filename": index.filename, "page_number": chunk.page_number,
            "section": chunk.metadata.get("section_heading", "Unlabelled section"),
            "chunk_index": chunk.chunk_index, "page_start": pos + offset,
            "page_end": pos + offset + len(excerpt), "text": excerpt, "retrieval_score": score}


def _best_window_start(text: str, question: str, limit: int):
    """Prefer the part of a retrieved chunk containing the question terms."""
    if len(text) <= limit:
        return 0
    terms = question_terms(question)
    starts = [0] + [match.start() for match in re.finditer(r"(?m)^\S", text)]
    for term in terms:
        for match in list(re.finditer(rf"\b{re.escape(term)}\b", text, re.I))[:4]:
            start = max(0, match.start() - 60)
            while start and not text[start - 1].isspace():
                start -= 1
            starts.append(start)
    def score(start):
        excerpt = text[start:start + limit]
        found = {word.lower() for word in re.findall(r"[A-Za-z0-9_]+", excerpt)}
        return (len(terms & found), sum(excerpt.lower().count(term) for term in terms), -start)
    return max(starts, key=score)


def _clean_excerpt_end(text: str, start: int, end: int):
    if end == len(text):
        return end
    stops = [start + match.end() for match in re.finditer(r"[.!?](?=\s|$)", text[start:end])]
    if stops and stops[-1] - start >= 180:
        return stops[-1]
    space = max(text.rfind(" ", start + 80, end), text.rfind("\n", start + 80, end))
    return space if space > start else end


_HEADINGS = {
    "method": re.compile(r"(?im)^(?:\d+(?:\.\d+)*\s*\n)?(?:methods?|methodology|approach)\s*$"),
    "results": re.compile(r"(?im)^(?:\d+(?:\.\d+)*\s*\n)?(?:main results|results|findings)\s*$"),
    "experiments": re.compile(r"(?im)^(?:\d+(?:\.\d+)*\s*\n)?(?:experiments?|evaluation)\s*$"),
    "conclusion": re.compile(r"(?im)^(?:\d+(?:\.\d+)*\s*\n)?(?:conclusions?|limitations?|discussion)\s*$"),
}


def _first_prose_offset(text: str):
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if len(stripped) >= 55 and len(stripped.split()) >= 8 and not re.match(r"(?i)^(figure|table|references)\b", stripped):
            return cursor + len(line) - len(line.lstrip())
        cursor += len(line)
    return 0


def _heading_location(pages, category):
    patterns = (_HEADINGS["results"], _HEADINGS["experiments"]) if category == "results" else (_HEADINGS[category],)
    for pattern in patterns:
        for page in pages:
            match = pattern.search(page.page_content)
            if match and len(page.page_content) - match.end() >= 130:
                return page, match.end(), category.title()
    return None


_SUMMARY_TERMS = {
    "method": ("approach", "method", "algorithm", "framework", "pipeline", "dependency",
               "audit", "diagnosis", "system", "model", "propose", "first", "then"),
    "results": ("result", "achieve", "accuracy", "improve", "outperform", "baseline",
                "higher", "best", "compare", "performance", "percentage", "evaluation"),
}


def _best_prose_location(pages, category):
    """Find source prose, not a section name embedded in a PDF table.

    The selected excerpt is still an exact substring with page offsets. This
    only chooses which existing text to show; it never rewrites PDF evidence.
    """
    best = None
    for page in pages:
        cursor = 0
        for line in page.page_content.splitlines(keepends=True):
            stripped = line.strip()
            start = cursor + len(line) - len(line.lstrip())
            cursor += len(line)
            if len(stripped) < 55 or len(stripped.split()) < 8:
                continue
            excerpt = page.page_content[start:start + 430]
            words = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", excerpt)
            if len(words) < 40:
                continue
            prose_lines = sum(len(part.split()) >= 7 for part in excerpt.splitlines())
            numeric_lines = sum(bool(re.fullmatch(r"\s*\d+(?:\.\d+)?%?\s*", part))
                                for part in excerpt.splitlines())
            if prose_lines < 2 or numeric_lines > 2:
                continue
            if sum(bool(re.search(r"\d", word)) for word in words) / len(words) > 0.17:
                continue
            lowered = excerpt.lower()
            hits = sum(len(re.findall(rf"\b{term}\w*\b", lowered))
                       for term in _SUMMARY_TERMS[category])
            if category == "results" and hits < 2:
                continue
            outcome_hits = len(re.findall(
                r"\b(?:achiev\w*|improv\w*|outperform\w*|increas\w*|reduc\w*|"
                r"gain\w*|higher|better|reveal\w*|find\w*|show\w*|report\w*)\b", lowered))
            if category == "results" and not outcome_hits:
                continue
            score = (hits * 4 + prose_lines + min(excerpt.count("."), 4)
                     - numeric_lines * 8 - (8 if "experimental setup" in lowered else 0))
            if category == "results":
                score += min(outcome_hits, 4) * 10
                score -= 16 if re.search(r"\b(?:benchmark and baselines|test set|experimental setup)\b", lowered) else 0
                score += 20 if re.search(r"\b(?:our (?:method|model|system|approach|training|results?)|main observations)\b", lowered) else 0
            if category == "method":
                score -= 12 * len(re.findall(r"\b(?:accuracy|performance|outperform\w*|improv\w*|results?)\b", lowered))
                score -= 8 if "%" in excerpt else 0
                score += 16 * len(re.findall(
                    r"\b(?:we (?:propose|introduce|develop|use)|our (?:approach|method|system)|"
                    r"at its core|the proposed|first reconstructs|consists of)\b", lowered))
                score -= 18 if re.search(r"\b(?:recent studies|prior work|related work)\b", lowered) else 0
            candidate = (score, -page.page_number, -start, page, start)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return (best[3], best[4], category.title()) if best else None


def _short_heading_location(pages, category):
    """Allow concise prose in short documents, but never a numeric table."""
    found = _heading_location(pages, category)
    if found is None:
        return None
    page, start, heading = found
    excerpt = page.page_content[start:start + 430]
    numeric_lines = sum(bool(re.fullmatch(r"\s*\d+(?:\.\d+)?%?\s*", line))
                        for line in excerpt.splitlines())
    if numeric_lines > 2 or len(re.findall(r"[A-Za-z]+", excerpt)) < 12:
        return None
    return page, start, heading


def _result_observation_location(pages):
    primary = re.compile(r"(?im)^.{0,25}\bour (?:training|method|model|system|approach)\b"
                         r"[^\n]{0,100}\b(?:achiev\w*|improv\w*|outperform\w*|"
                         r"increas\w*|reduc\w*|gain\w*)\b")
    for page in pages:
        match = primary.search(page.page_content)
        if match:
            opening = re.search(r"\b(?:First|Second|Third|Finally),?\s+our\b", match.group(), re.I)
            start = match.start() + (opening.start() if opening else match.group().lower().find("our "))
            return page, start, "Results"
    marker = re.compile(r"(?im)^(?:main (?:observations|findings|results)|key (?:findings|results)|"
                        r"overall [^\n]{0,50}(?:accuracy|performance|results))\.\s*")
    for page in pages:
        for match in marker.finditer(page.page_content):
            excerpt = page.page_content[match.start():match.start() + 430].lower()
            if re.search(r"\b(?:achiev\w*|improv\w*|outperform\w*|increas\w*|reduc\w*|"
                         r"gain\w*|higher|better|reveal\w*|show\w*)\b", excerpt):
                return page, match.start(), "Results"
    return None


def _page_excerpt(index: PDFIndex, page, start: int, section: str, number: int, limit: int = 430):
    text = page.page_content
    while start < len(text) and text[start].isspace():
        start += 1
    end = _clean_excerpt_end(text, start, min(len(text), start + limit))
    excerpt = text[start:end]
    if len(excerpt) < 80:
        raise ValueError("Selected paper section has too little extractable text")
    related = [c for c in index.chunks if c.page_number == page.page_number]
    chunk = next((c for c in related if c.chunk_text[:70] in text[max(0, start - 900):end + 900]), related[0])
    return {"id": f"E{number}", "document_id": index.document_id, "filename": index.filename,
            "page_number": page.page_number, "section": section, "chunk_index": chunk.chunk_index,
            "page_start": start, "page_end": end, "text": excerpt, "retrieval_score": 0.0}


def _summary_categories(question: str, intent: str):
    if intent == "whole_paper_summary":
        return ("problem", "method", "results", "conclusion")
    q = question.lower()
    if re.search(r"\b(?:method|methodology|approach|how did|how does)\b", q):
        return ("method",)
    if re.search(r"\b(?:results?|findings?|experiments?|experimental|accuracy|performance)\b", q):
        return ("results",)
    if re.search(r"\b(?:limitation|conclusion|discussion)\b", q):
        return ("conclusion",)
    return ("problem",)


def _summary_sources(index: PDFIndex, question: str, intent: str):
    pages = _main_body_pages(index)
    if not pages:
        return []
    selected = []
    results_heading = _heading_location(pages, "results")
    conclusion_heading = _heading_location(pages, "conclusion")
    for category in _summary_categories(question, intent):
        if category == "problem":
            page, start, heading = pages[0], _first_prose_offset(pages[0].page_content), "Opening/abstract"
        elif category in ("method", "results"):
            if category == "method":
                candidates = [p for p in pages[1:] if results_heading is None or
                              p.page_number < results_heading[0].page_number]
            else:
                candidates = [p for p in pages if
                              (results_heading is None or p.page_number >= results_heading[0].page_number)
                              and (conclusion_heading is None or p.page_number < conclusion_heading[0].page_number)]
            location = _result_observation_location(candidates) if category == "results" else None
            if location is None:
                location = _best_prose_location(candidates, category)
            if location is None:
                location = _short_heading_location(candidates, category)
            if location is None:
                continue
            page, start, heading = location
        else:
            location = _heading_location(pages, category)
            if location is None:
                continue
            page, start, heading = location
        selected.append(_page_excerpt(index, page, start, heading, len(selected) + 1))
    return selected


def _spread(chunks, maximum):
    if len(chunks) <= maximum:
        return chunks
    if maximum == 1:
        return chunks[:1]
    positions = [round(i * (len(chunks) - 1) / (maximum - 1)) for i in range(maximum)]
    return [chunks[i] for i in dict.fromkeys(positions)]


def plan(index: PDFIndex, question: str, *, max_sources: int = 12):
    intent = classify_intent(question)
    ranked, latency = ranked_chunks(index, question, count=8)
    if intent == "unsupported":
        return {"intent": intent, "sources": [], "retrieval_latency_ms": latency,
                "coverage": {"pages_sampled": 0, "pages_total": len(index.pages)}}
    if intent in ("whole_paper_summary", "section_summary"):
        sources = _summary_sources(index, question, intent)[:max_sources]
    else:
        scored = ranked[:min(max_sources, 4)]
        sources = [_source(index, chunk, score, i + 1, question=question)
                   for i, (chunk, score) in enumerate(scored)]
    body_pages = _main_body_pages(index)
    return {"intent": intent, "sources": sources, "retrieval_latency_ms": latency,
            "coverage": {"pages_sampled": len({s["page_number"] for s in sources}),
                         "pages_total": len(index.pages),
                         "main_body_pages": len(body_pages),
                         "back_matter_pages_excluded": len(index.pages) - len(body_pages),
                         "sections_sampled": sorted({s["section"] for s in sources}),
                         "chunks_sampled": len(sources), "chunks_total": len(index.chunks)},
            "top_score": ranked[0][1] if ranked else 0.0}
