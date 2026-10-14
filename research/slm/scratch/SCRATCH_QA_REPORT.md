# SLM-A synthetic evidence-conditioned QA experiment

## Outcome

**NO-GO for a local QA interface or real-corpus QA training.** The scientifically
usable v3 held-out evaluation has 160 examples (80 answerable, 80 genuinely
unanswerable). The 1,082,496-parameter model generated `UNANSWERABLE` for all
160. Its exact-marker abstention was 80/80, while answerable exact match and
evidence support were both 0/80. The preregistered pass gate failed. These
predictions are the model's own greedy outputs; no rule answer was substituted.

| Condition | Validation exact | Test exact | Test answerable exact | Test abstention | Test token F1 | Test evidence support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch Transformer, selected step 700 | 80/160 (50%) | 80/160 (50%) | 0/80 | 80/80 | 0.50 | 0/80 |
| Rule extractor, exact entity boundary | 160/160 | 160/160 | 80/80 | 80/80 | 1.00 | 80/80 |

The rule result is expected for rigid generated templates. It establishes that
the passage/question pairs are mechanically extractable; it does not imply the
task is representative of open-domain QA. Model abstention precision was 0.50
because it also abstained on all answerable questions. Median model generation
latency was 4.65 ms per test example; peak MLX allocation during evaluation
was 22,717,528 bytes. See the [immutable evaluation results](qa/v3/run-001/evaluation.json)
and [individual model predictions](qa/v3/run-001/test_model_predictions.jsonl).

## Corpus and validity gates

The corpus uses only original synthetic technical templates. Train comprises
1,200 examples from three families; validation has 160 from a fourth family;
test has 160 from a fifth family. Each split is 50% answerable. Every target
passage has a distractor with the requested field, and unanswerable targets
lack that field. v3 has 150 positives and 150 negatives for each field in
training, and 20 of each per field in validation and test. Opaque entity IDs
and fact IDs are split-disjoint. Numeric value ranges are split-specific;
categorical vocabulary is shared so unseen composition, rather than unseen
English words, is tested. The new byte BPE vocabulary has 1,024 entries and
was fit to training prompts/answers only. All sequences fit the 192-token cap
(train max 96, validation max 160, test max 129). The 4-layer, width-128,
4-head, context-256 model was freshly randomized; no earlier checkpoint was
loaded. Data/config IDs and hashes are in the [v3 manifest](qa/v3/manifest.json)
and [frozen sample plan](qa/v3/plan.json).

Two preliminary design versions are retained to make failures inspectable.
v1 linked answerability directly to the requested field; its first rule
extractor also matched a target ID inside a distractor ID. The [rule correction](qa/v1/run-001/rule_correction_v2.json)
preserves the original evaluation and fixes the parser on validation examples.
v2 balanced requested fields but still exposed the label through numeric
entity-ID parity. Its 500-step guarded training run was completed, but its
test set was **not** opened and its QA performance is not claimed. v3 fixes
both shortcuts. The v3 test file was opened only after step 700 was selected
by validation loss. The pre-existing QASPER test artifacts were untouched.

## Training and resource measurements

The [preregistered v3 config](qa/v3/config.json) capped AdamW at 800 updates,
batch size 1, LR 0.0005, weight decay 0.01, 180 seconds wall time, and
evaluated/checkpointed every 100 steps. The fixed train set had 64 examples;
the fixed validation set had 80. Loss is answer-token-plus-EOS cross entropy,
not full-prompt loss. The worker performed 800 steps in 25.1 seconds, processing
62,203 input tokens at 3,121 input tokens/second (mean update 24.9 ms).

| Fixed-set measure | Initial | Best step 700 | Final step 800 |
| --- | ---: | ---: | ---: |
| Training answer loss | 7.2933 | 1.4327 | 1.3743 |
| Validation answer loss | 6.9493 | 1.7908 | 1.8706 |
| Validation answer perplexity | 1042.47 | 5.99 | 6.49 |

The selected checkpoint's 28 model tensors restored exactly; restored
validation loss differed by 0.0 and the optimizer step was 700. The decline
in answer-token loss proves optimization progress on this distribution, while
the all-abstain generation result shows it did **not** learn usable extraction.
The rise from step 700 to 800 is a single short fixed-set observation, not a
general overfitting claim. Full curves and per-step gradient/parameter norms
are in [curve.jsonl](qa/v3/run-001/curve.jsonl) and [steps.jsonl](qa/v3/run-001/steps.jsonl).

The preflight passed with 1.87 GB system-available memory, 45% reported free
memory pressure and 22.0 GB free disk. During training, minimum available
memory was 1.59 GB, minimum pressure-free percentage 39%, maximum sampled
worker RSS 84.4 MB, and peak MLX allocation 45,095,908 bytes. Swap **used**
grew 0 bytes; cumulative swap-out activity grew 10,960,896 bytes. Minimum
disk headroom was 21.9 GB. The parent watchdog enforced the existing floors
and stopped no trial. These are measurements of this bounded run, not a
guarantee for a larger or prolonged experiment. See [run.json](qa/v3/run-001/run.json).

## Interpretation and next step

The model learned the easy marker strongly while failing extraction under a
new family and unseen facts. A plausible cause is that one repeated marker is
easier to optimize than many varied answer spans; this is not established as
the sole cause. The large train-to-validation tokenization-length difference
and family shift may also matter. The current experiment cannot distinguish
representation limits, optimization limits, and synthetic-task design effects.
Do not infer academic PDF QA ability from these losses or from the rule parser.

Next, audit **validation-only** conditional answer probabilities and training
example generations, then preregister one revised synthetic task or loss
design before any additional training. Keep the v3 test result sealed as a
final observed result; a revised task needs a new untouched test partition.
No inference interface was released because the preregistered answerable and
evidence-support thresholds were not met.

Reproduction commands and exact metric definitions are in [qa/README.md](qa/README.md).
Focused unit tests passed for v1, v2 and v3 (seven tests per version; one
v3-specific test skipped on the earlier versions).
