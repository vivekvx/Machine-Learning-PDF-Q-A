# Local evidence and source-rights review

This workflow reads only the fixed 64-row QASPER **train** review CSV and the frozen 352-row train audit. It never reads held-out test answers, runs a model, or edits the prior experiment artifacts. The localhost interface uses Python's standard library; it binds `127.0.0.1` only and saves append-only review events separately from the sample CSV.

## Evidence review

From the repository root:

```sh
python3 research/slm/human_review/server.py --port 8765
```

Open `http://127.0.0.1:8765/`. Each reviewer enters their own stable ID, reviews the passage and original annotation, opens the automated flags **after** forming an initial judgment, and saves one decision per question. The interface shows only that reviewer's prior decision and progress. Revisions append a new event; previous events remain in `human_review/v1/reviews.jsonl`. The original sample remains unchanged. The server requires an unpredictable per-process form token to prevent unsolicited browser submissions.

Support judgments are **Supported** (the supplied passage entails the reference), **Partially Supported** (some but not every indispensable fact), **Unsupported** (the supplied passage does not support the reference), **Ambiguous** (multiple reasonable readings), or **Needs Review** (source/annotation issue requiring investigation). The abstention question concerns the **supplied context**, not document-level QASPER answerability. For QASPER-unanswerable questions, `UNANSWERABLE` is a target marker rather than a fact established by a passage: mark the excerpt **Unsupported** when it supplies no answer, mark abstention **Yes**, and mark document-level consistency **Yes** only after checking the full source. An excerpt alone does not prove the full document lacks an answer; full-document inspection must be recorded separately. Reviewers also record original-label consistency and medical-domain concerns. Choose `Uncertain` when the information needed for a judgment is unavailable. No field has an automatic human verdict.

After independent review, run:

```sh
python3 research/slm/human_review/summary.py
```

The summary counts latest decisions per reviewer/question, retains raw event counts, and reports pairwise raw agreement and Cohen's kappa where reviewers overlap. Kappa is `null` when mathematically undefined. Review disagreements require adjudication outside the original event log; do not silently choose a majority or overwrite either review. Reviewer IDs are labels; the software cannot prove that two IDs belong to independent people. The 64-row sample is triage; it cannot certify all 352 training examples or unmaterialized evidence conditions.

## Publication and rights review

The pinned official ACL Anthology BibTeX export and copyright-policy page are in `human_review/v1/sources/`; their URLs and SHA-256 hashes are in `v1/resolver_config.json`. The export is bibliographic metadata, not full paper text. Rebuild the provenance ledger into a new empty directory with:

```sh
python3 research/slm/human_review/resolve_acl.py --output-dir /tmp/slm-acl-check
```

Compare `/tmp/slm-acl-check/provenance_ledger.jsonl` with `v1/provenance_ledger.jsonl` by SHA-256. Only **unique normalized exact title** matches are accepted automatically as candidate publication identities. Near titles are suggestions, not resolved IDs. The official [ACL copyright policy](https://aclanthology.org/faq/copyright/) applies CC BY 4.0 to ACL materials from 2016 and CC BY-NC-SA 3.0 before 2016; it explicitly excludes third-party materials. The ledger records ACL publication-license class separately from the still-unresolved rights of QASPER's arXiv/S2ORC text version and any third-party passages or figures.

A qualified reviewer can append a separate per-document rights event with `record_rights.py`. It requires an official license evidence URL, attribution, a reasoned note, explicit source-version equivalence, third-party clearance, and training/redistribution scope. Example syntax (replace every placeholder with verified facts):

```sh
python3 research/slm/human_review/record_rights.py \
  --document-id DOCUMENT_ID --reviewer-id REVIEWER_ID \
  --decision unresolved \
  --source-version-matches Uncertain --third-party-material-cleared Uncertain \
  --training-and-redistribution-allowed Uncertain \
  --license-evidence-url 'https://aclanthology.org/faq/copyright/' \
  --attribution 'Author and title from the source' \
  --notes 'Explain what remains unverified'
```

Rights events live in `human_review/v1/rights_decisions.jsonl`; they never alter the official metadata ledger. The software cannot independently establish reviewer qualifications, source-version equivalence, third-party rights, or legal sufficiency. A permissive ACL publication label alone does not clear a document for training.

## Candidate snapshot

Run `cohort.py --output-dir NEW_EMPTY_DIRECTORY` after reviews. The builder requires two independent, agreeing evidence reviews, a cleared rights event, the frozen train IDs, and no medical concern. For document-unanswerable examples it also requires full-document review and abstention approval. It writes a candidate JSONL plus a reason record for every excluded or unresolved train question. It refuses overwrite. A candidate remains **not variant-training-ready** until all partial/irrelevant/distractor contexts and cloud controls in `EXPERIMENT_PROTOCOL_V2.md` are independently reviewed. The initial v1 snapshot has zero candidates and retains all 352 reason records.

## Checks

```sh
python3 -m unittest discover -s research/slm/human_review/tests -v
python3 -m compileall -q research/slm/human_review
```
