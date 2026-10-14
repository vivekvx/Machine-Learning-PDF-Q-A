"""Dependency-free, localhost-only evidence review UI for the frozen 64-row sample."""
import argparse
import html
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit

from review_core import ABSTENTION, DECISIONS, JUDGMENTS, SAMPLE, append_review, latest_reviews, load_sample, review_events, sha256


def esc(value):
    return html.escape(str(value), quote=True)


def choices(name, values, selected=None):
    return "".join(f'<label class="choice"><input type="radio" name="{esc(name)}" value="{esc(value)}" required {"checked" if value == selected else ""}> {esc(value)}</label>' for value in values)


def page(row, index, total, reviewer, own=None, error="", navigation=None, next_unreviewed=None, csrf_token=""):
    flags = json.loads(row["automatic_flags_json"])
    annotations = json.dumps(json.loads(row["original_annotations_json"]), indent=2, ensure_ascii=False)
    evidence = json.dumps(json.loads(row["original_evidence_passages_json"]), indent=2, ensure_ascii=False)
    previous = f'<p class="notice">Your previous decision: {esc(own["judgment"])}; abstention {esc(own["abstention_appropriate"])}. Saving again appends a correction event.</p>' if own else ""
    navigation = navigation or []
    selector = "".join(f'<option value="{i}" {"selected" if i == index else ""}>{i+1}. {"✓ " if done else ""}{esc(title[:65])}</option>'
                       for i, title, done in navigation)
    progress = sum(done for _, _, done in navigation)
    jump = f'<a href="/?reviewer={quote(reviewer)}&i={next_unreviewed}">Next unreviewed</a>' if next_unreviewed is not None else ""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evidence review {index+1}/{total}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:auto;padding:1.2rem;color:#17212b;background:#f8fafc}}main,aside{{background:white;padding:1.2rem;border:1px solid #d8e0e8;border-radius:8px;margin-bottom:1rem}}h1{{line-height:1.2}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem;border-radius:6px}}.passage{{font-size:1.05rem}}.choice{{display:block;padding:.35rem}}fieldset{{margin:1rem 0;border:1px solid #cbd5e1}}legend{{font-weight:700}}button{{padding:.6rem 1rem;font-weight:700}}.notice{{background:#fff5d6;padding:.7rem}}.error{{background:#ffe1e1;padding:.7rem}}a{{margin-right:1rem}}</style></head><body>
<header><h1>Train evidence review <small>{index+1}/{total}</small></h1><p>Independent human judgment. Automated flags are prompts for inspection, not verdicts. No model predictions are shown. An excerpt alone cannot establish document-wide unanswerability.</p>
<p>Your saved decisions: {progress}/{total}. <a href="/?reviewer={quote(reviewer)}&i={max(index-1,0)}">Previous</a><a href="/?reviewer={quote(reviewer)}&i={min(index+1,total-1)}">Next</a>{jump}</p>
<form method="get" action="/"><input type="hidden" name="reviewer" value="{esc(reviewer)}"><label>Jump to question <select name="i">{selector}</select></label> <button type="submit">Go</button></form></header>
{f'<p class="error">{esc(error)}</p>' if error else ''}{previous}
<main><p><strong>Document:</strong> {esc(row['title'])} · <strong>Source ID:</strong> {esc(row['document_id'])} · <strong>Question ID:</strong> {esc(row['question_id'])}</p>
<p><a href="https://arxiv.org/abs/{quote(row['document_id'])}" target="_blank" rel="noopener noreferrer">Open arXiv source identifier</a> (verify the exact QASPER text version before judging document-wide answerability or rights).</p>
<p><strong>Original QASPER label:</strong> {esc(row['answerability'])} · <strong>Section:</strong> {esc(row['section_name'])} · <strong>Passage ID:</strong> {esc(row['selected_passage_id'])}</p>
<h2>Question</h2><p>{esc(row['question'])}</p><h2>Reference answer</h2><p>{esc(row['reference_answer'])}</p>
<h2>Supplied source passage</h2><pre class="passage">{esc(row['selected_passage_text'])}</pre>
<details><summary>Document abstract</summary><p>{esc(row['abstract'])}</p></details>
<details><summary>Original annotation and evidence</summary><h3>Annotations</h3><pre>{esc(annotations)}</pre><h3>Original evidence records</h3><pre>{esc(evidence)}</pre></details>
<details><summary>Automated warning flags (open after judging the text)</summary><pre>{esc(json.dumps(flags,indent=2))}</pre></details></main>
<aside><h2>Save your review</h2><form method="post" action="/review"><input type="hidden" name="csrf_token" value="{esc(csrf_token)}"><input type="hidden" name="question_id" value="{esc(row['question_id'])}"><input type="hidden" name="i" value="{index}">
<label>Reviewer ID <input name="reviewer_id" value="{esc(reviewer)}" required minlength="3" maxlength="64" pattern="[A-Za-z0-9._-]+"></label>
<fieldset><legend>Is the reference answer supported by this supplied passage?</legend>{choices('judgment',JUDGMENTS,own['judgment'] if own else None)}</fieldset>
<fieldset><legend>Is abstention appropriate given this supplied context?</legend>{choices('abstention_appropriate',ABSTENTION,own['abstention_appropriate'] if own else None)}</fieldset>
<fieldset><legend>Is the original document-level answerability label consistent with the source you reviewed?</legend>{choices('document_answerability_consistent',ABSTENTION,own['document_answerability_consistent'] if own else None)}</fieldset>
<fieldset><legend>Did you inspect the full source document for this decision?</legend>{choices('full_document_reviewed',ABSTENTION,own['full_document_reviewed'] if own else None)}</fieldset>
<fieldset><legend>Is this medical, biomedical, clinical, pharmaceutical, or healthcare research?</legend>{choices('medical_domain',ABSTENTION,own['medical_domain'] if own else None)}</fieldset>
<label>Reason and source notes<br><textarea name="notes" rows="5" cols="85" maxlength="4000">{esc(own['notes'] if own else '')}</textarea></label><p><button type="submit">Save independent decision</button></p></form></aside></body></html>'''


class Handler(BaseHTTPRequestHandler):
    sample_path = SAMPLE
    decisions_path = DECISIONS
    csrf_token = secrets.token_urlsafe(32)

    def send_html(self, body, status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path != "/":
            self.send_error(404)
            return
        query = parse_qs(url.query)
        try:
            rows = load_sample(self.sample_path)
            index = max(0, min(int(query.get("i", ["0"])[0]), len(rows)-1))
            reviewer = query.get("reviewer", [""])[0]
            latest = latest_reviews(review_events(self.decisions_path), {r["question_id"] for r in rows}, sha256(self.sample_path)) if reviewer else {}
            previous = latest.get((rows[index]["question_id"], reviewer))
            navigation = [(i, row["title"], (row["question_id"], reviewer) in latest) for i, row in enumerate(rows)]
            remaining = [i for i, _, done in navigation if not done and i > index] + [i for i, _, done in navigation if not done and i <= index]
            self.send_html(page(rows[index], index, len(rows), reviewer, previous,
                                navigation=navigation, next_unreviewed=remaining[0] if remaining else None,
                                csrf_token=self.csrf_token))
        except (ValueError, KeyError) as exc:
            self.send_error(400, str(exc))

    def do_POST(self):
        if urlsplit(self.path).path != "/review":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16000:
                raise ValueError("Invalid form size")
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            if form.get("csrf_token", [""])[0] != self.csrf_token:
                raise ValueError("Invalid local form token")
            decision = {key: form.get(key, [""])[0] for key in ("question_id", "reviewer_id", "judgment", "abstention_appropriate", "document_answerability_consistent", "full_document_reviewed", "medical_domain", "notes")}
            rows = load_sample(self.sample_path)
            index = int(form.get("i", ["0"])[0])
            if not 0 <= index < len(rows) or rows[index]["question_id"] != decision["question_id"]:
                raise ValueError("Question and review position disagree")
            append_review(decision, sample_path=self.sample_path, decisions_path=self.decisions_path)
            self.send_response(303)
            self.send_header("Location", f'/?reviewer={quote(decision["reviewer_id"])}&i={min(index+1,len(rows)-1)}')
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            self.send_html(f'<p>Review not saved: {esc(exc)}</p><p><a href="/">Return to review</a></p>', 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local review UI: http://127.0.0.1:{args.port}/ (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
