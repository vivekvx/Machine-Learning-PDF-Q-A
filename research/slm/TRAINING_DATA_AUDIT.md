# Train-only evidence and expansion audit

**Status:** Automated triage completed on 2026-10-11. **Independent human evidence reviews completed: 0.** No training, baseline reruns, FastAPI edits, or held-out test-answer access occurred. The previous portable train file, fixed 80-question validation file, and all prior baseline/feasibility outputs retain their original hashes.

## Frozen sources and annotation preservation

The selected QASPER training cohort is **352 questions from 352 distinct included train documents**: 256 answerable and 96 consensus document-unanswerable. The train JSONL SHA-256 remains `34ee127615ebbdf800066257b6d06022be37ecc6a0b25edb3cac0422d35cda9a`; the validation JSONL SHA-256 remains `54ef6bbffda3072918d51e0ce37cdf24c87be9e844b64b23b02742794eeedba0` (80 questions from 80 separate validation documents). These files and QASPER's official document-level splits were not regenerated or changed.

The new `training_data_v2/v1/selected_audit.jsonl` joins each selected row to its **full original train-only annotations**, all original evidence fields, section name, complete selected paragraph, annotation/evidence IDs, source revision, and source paragraph SHA-256. It does not replace the original curated records. An integrity validator confirmed the join for all 352 rows. Of the selected questions, 343 have one original annotation and nine have two; each answerable row has a source-matched paragraph and its QASPER evidence ID, while each unanswerable row has no gold evidence and uses a labeled document excerpt.

## What automated checks found

The flags are **review triggers**, not semantic entailment judgments. Category totals overlap.

| Automated category | Selected rows flagged | Meaning |
| --- | ---: | --- |
| Missing/broken selected passage, evidence ID, or source link | 0 | Exact structural checks found no missing selected evidence in the 256 answerable rows. |
| Possible contradiction | 1 | Two original annotators gave opposite yes/no answers to the same question. |
| Ambiguous answer representation | 47 | Includes 45 one-content-token answers and four low-overlap/absent answer variants; some flags overlap. |
| Potentially insufficient context | 153 | 57 answerable rows have low question-term overlap with the selected paragraph; all 96 unanswerable excerpts remain unverified as proof of absence. |

The possible-contradiction row is question `5679fabeadf680e35a4f7b092d39e8638dca6b4d`: the selected target is “No,” while its two preserved yes/no annotations are `True` and `False`. It is **not resolved** by choosing one reference. Four selected answerable rows have a non-selected answer variant that is not a literal match in the selected paragraph; four have low-overlap answer variants. The literal occurrence of the selected reference in a passage is a construction rule, **not proof that the passage answers the question**. In particular, short answers can coincide with unrelated text. The 96 unanswerable excerpts are explicitly **not gold evidence**; a single excerpt cannot establish that an entire paper lacks an answer. Four of them have high question-term overlap and receive extra review priority.

The deterministic [64-row evidence review sample](/Users/vivek/Machine-Learning-PDF-Q-A/research/slm/EVIDENCE_REVIEW_SAMPLE.csv) includes **40 answerable and 24 unanswerable** selected questions. It samples 20 higher-risk and 20 lower-risk answerable rows; all four high-overlap unanswerable rows and 20 lower-overlap unanswerable rows are included. Selection uses SHA-256 order of seed `20261011` and question ID within those strata. Each CSV row contains the full original selected paragraph, title, abstract, question, reference variants, raw annotations/evidence, flags, and blank fields for reviewer decisions. **No human verdicts were invented or prefilled.** See the review instructions below before any cloud training.

## Additional train-only candidates

Reapplying the *unchanged* strict preparation rules to all 2,311 included train questions found **752 eligible questions across 474 documents**. The 352 selected documents leave **122 further eligible documents** (61 answerable, 61 document-unanswerable under first eligible source order). These contain 27 additional eligible questions beyond the one candidate per document. The first candidate per extra document is recorded in `training_data_v2/v1/candidate_expansion.jsonl`; none was added to the frozen training IDs.

All 122 first candidates fit the 512-token sequence rule (maximum full length **442 tokens**). Among other questions in those candidate documents, two failed the token budget. Across **all** 2,311 included train questions, 15 failed the token budget; other rejection counts were 633 reference answers absent from matched gold text, 473 multi-paragraph gold evidence, 362 missing/unmatched evidence, and 76 source mismatches. These are counts of rejected questions, not additional usable records. Candidate flags include 61 unanswerable-excerpt uncertainty signals, 17 low question/passage overlaps, 15 short-answer weak matches, and eight high-overlap unanswerable excerpts. None of these has been independently adjudicated.

Across the 474 selected-plus-candidate train documents, normalized full-document hashes found **zero exact duplicate groups** and normalized-title similarity at threshold 0.92 found **zero near-title pairs**. These limited detectors do not rule out paraphrased or semantically duplicate papers. The earlier curated cross-split duplicate audit remains the existing split-control record; this task did not inspect held-out test answers. The candidate documents have the same **unverified individual-paper training-rights status** as the selected cohort.

## Review instructions and release gate

1. Two independent reviewers should first read the question, complete selected passage, title/abstract, raw annotations, and source ID **without seeing model outputs or the automated flag as a proposed verdict**. The flags may then guide adjudication.
2. Record whether the reference is fully supported, partially supported, unsupported, contradicted, or uncertain by the **supplied passage**; separately judge the original document-answerability annotation. Check every preserved answer variant, yes/no conflict, citation/source identity, and possible medical subject matter. A mere lexical match is insufficient.
3. For unanswerable rows, decide whether the shown excerpt actually supports an answer. Do not claim it proves document-wide unanswerability; compare the original annotation and full train document only when adjudication requires it. Record any ambiguity.
4. Assign reviewer IDs and notes independently, then adjudicate disagreements with a third reviewer. Keep original annotations and both initial judgments. Do not edit the frozen source rows to hide conflicts.
5. Review individual-paper training and redistribution rights separately from QASPER's dataset-level CC BY 4.0 metadata. If rights cannot be established, retain the ledger status as **unverified** and withhold that paper from an authorized training release.
6. The 64-row sample is a **triage sample**, not a substitute for reviewing every selected question and every constructed distractor/negative context. Any exclusion must be applied symmetrically to all model arms in a new version; do not silently replace questions from the 122 candidates.

**Data GO criteria:** all proposed training questions and constructed evidence conditions receive independent support/answerability review; disagreements are adjudicated; source paper training rights are cleared; every passage fits the tokenizer budget without silent truncation; duplicate/medical checks are documented; and a new immutable manifest records the approved cohort. **Current recommendation: NO-GO** for cloud training. The 352 IDs, 80 validation IDs, and design are frozen as candidates, while evidence-condition materialization and legal clearance remain blocked.

## Reproducibility

The versioned [configuration](/Users/vivek/Machine-Learning-PDF-Q-A/research/slm/training_data_v2/v1/config.json), [manifest](/Users/vivek/Machine-Learning-PDF-Q-A/research/slm/training_data_v2/v1/manifest.json), 352-row audit, 122-row candidate list, 888-row train-document rights ledger, fixed train/validation ID lists, and three-arm variant plan are under `training_data_v2/v1/`. From the repository root, `research/slm/.venv/bin/python research/slm/training_data_v2/validate.py` checks the saved artifacts; `research/slm/.venv/bin/python -m unittest discover -s research/slm/training_data_v2/tests -v` runs focused tests. `build.py` refuses to overwrite v1 outputs. No script in this workflow opens held-out test question or answer files.

## Checks executed

On 2026-10-11, the train-only validator passed for all 352 selected records, 64 blank-human-review sample rows, 122 candidate documents, 888 ledger documents, ID isolation, source passage hashes, and pinned input/output/tool hashes. A clean temporary rebuild matched **all eight artifact SHA-256 hashes and the entire saved manifest**. Ten focused audit/abstention tests and six existing retrieval-baseline artifact tests passed. These checks verify integrity and deterministic construction; they are not independent semantic evidence review, per-paper rights clearance, cloud compatibility, or a new model evaluation.
