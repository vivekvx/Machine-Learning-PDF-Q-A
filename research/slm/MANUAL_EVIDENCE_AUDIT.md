# Manual validation evidence check

Reviewed the four answerable examples in `validation_gold_eval.jsonl` against the
selected source passages on 2026-10-11. These are spot checks, not a full annotation review.
No test answers were inspected for this review.

| Document ID | Question ID prefix | Judgment | Note |
| --- | --- | --- | --- |
| 1912.01214 | f5e6f434 | Partial | The selected passage supports the translate-through-pivot answer variant; it does not explicitly name the `pivoting_m` variant. |
| 1810.08699 | 18c5d366 | Supported | The passage names Stanford NER, spaCy 2.0, and the recurrent CRF model. |
| 1609.00425 | b6ae8e10 | Supported | The passage lists politics, business, science, AskReddit, and frontpage posts. |
| 1811.00383 | 37861be6 | Supported | The passage lists five source languages translated into Hindi. |

The first example shows why a nonempty QASPER evidence field does not by itself
guarantee support for every annotated answer variant. The baseline evaluates against
any preserved variant and reports this limitation.
