# Final scratch SLM and evidence QA implementation report

## Result and scope

**The requested working neural QA acceptance gate was not met.** A new
1,017,346-parameter MLX Transformer reader learned eight deliberately tiny
training examples, but its selected checkpoint predicted `UNANSWERABLE` for
every one of 320 unseen validation examples. The independent, previously
trained scratch decoder-only language model remains an actual from-scratch
language-model artifact. The new QA reader is an extractive bidirectional
encoder, **not** a generative SLM. No neural QA demo or PDF RAG integration was
released. This is a verified implementation and failed generalization result,
not a claim of successful offline document QA.

The original RAG application, QASPER work, previous scratch runs, reports,
checkpoints and checksum ledgers were left unchanged. All new work is under
`working_qa/` and in this report. No pretrained weights, tokenizer, hosted
LLM, external technical corpus, or held-out QA answers were used to train or
select a model.

## Why the architecture changed

The earlier generative QA decoder correctly masked prompt tokens for loss,
but its selected checkpoint assigned much higher first-token probability to
`UNANSWERABLE` than to the gold answer for answerable validation examples.
It did not hit a generation length cap or merely fail output parsing. Its
prior start/end extractive successor used bidirectional attention, yet the
separate answerability head stayed near 0.5 and span heads often picked a
fixed non-value fragment. Those observations motivated a candidate-value
reader with an explicit NULL choice. This avoids unconstrained start/end
combinations and ensures a selected non-null answer is copied from a source
span. It does **not** use the template rule extractor to choose an answer.

The v1 reader reuses the project's scratch full-attention QKV, feed-forward,
and RMSNorm blocks: four layers, width 128, four heads, 256-token context,
1,024-token fresh byte BPE vocabulary, batch size 1. It encodes question and
passage together. A learned score compares a pooled question vector with
each bracketed source-value sentence and value; a learned NULL logit joins
one softmax. The loss is cross-entropy over the gold candidate or NULL.
There are **1,017,346 actual trainable parameters**, counted from MLX tensors.
The original autoregressive decoder and its checkpoints were not reused.

Because v1 still collapsed to NULL, one v2 comparison was preregistered. It
uses the same source dataset, tokenizer, dimensions and fresh seed, but scores
each question/candidate sentence pair with full attention, a learned lexical
alignment feature, and balanced binary candidate loss with fixed logit-zero
abstention. It has **1,017,345 parameters**. Its short tiny trial was stopped
by the unchanged live safety guard, so no v2 validation conclusion is
available. Both readers are neural extractors; neither generates free text.

## Corpus and supervision audit

Original synthetic, non-medical technical templates created 2,400 training,
320 validation, and 320 **sealed test** examples. Each split is half
answerable and half unanswerable, including each of four tiers: single fact,
numeric value, multiple entities, and paraphrase/conflict. The splits have
different document-family labels and disjoint full entity names and fact IDs.
All answers on the inspected train/validation examples have exact source
character spans; unanswerable targets lack the queried fact. The tokenizer
was fitted on training questions/passages only and has byte fallback with no
unknown-token ID. Maximum tokenized input lengths are 122 train, 148
validation, and 150 sealed-test *metadata* tokens, below the 192 training
limit. The sealed file's hash and aggregate manifest statistics were created
at build time; its example answers were not opened for selection or tuning.

The [quality audit](working_qa/v1/analysis/quality_audit.json) found zero
shared exact passage/question pairs, fact IDs, or full entity names across
train and validation. Candidate count alone has no label correlation: each
of counts 1, 2 and 4 is half positive. However **311/320 validation examples
have a tier/label/sentence-style signature already present in training**.
The family opener changes, but the three underlying sentence forms are
shared. The separate explicit template rule extractor scored 160/160
answerable and 160/160 unanswerable on validation. This confirms mechanical
extractability; it also limits any claim of broad linguistic generalization.
It is never used as a neural fallback.

## Frozen computation and measurements

The [v1 configuration](working_qa/v1/config.json) capped tiny memorization
at 400 steps and the main run at 3,000, with 420 seconds total wall budget,
AdamW LR 0.001, weight decay 0.01, gradient clipping 1.0, batch size 1,
fixed evaluation windows, early stopping and checkpoint selection by balanced
fixed-validation accuracy then NLL then earlier step. The plan froze IDs,
order, hashes and expected parameter count before training. The new v2 plan
fit inside the same **3,400 combined-step ceiling**, accounting for v1's
1,830 actual updates: v2 tiny cap 120, main cap 1,450. Actual total updates
were **1,910** (80 + 1,750 + 80); no v2 main run started.

| Run | Updates | Outcome | Fixed training | Fixed validation | MLX peak | Tokens/s | Wall |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| v1 tiny, fresh weights | 80 | 8/8 exact; checkpoint exact reload | NLL 1.4911 → 0.0143 | not used | 40,358,696 B | 5,986 | 1.53 s |
| v1 main, fresh weights | 1,750 | early stopped; selected step 750 | step 750: 40/80 exact, NLL 1.1792 | step 750: 80/160 exact, NLL 1.1794 | 44,691,184 B | 10,262 | 19.32 s |
| v2 tiny, fresh weights | 80 | **guard stopped** before pass/checkpoint | 6/8 at last completed evaluation (step 70); 7/8 at step 60 | not run | 36,608,481 B sampled | 4,475 | 2.90 s |

At v1's best fixed-validation checkpoint, the 80 correct predictions were
**all abstentions**; the 80 answerable questions were all wrong. Exact
restoration matched 33 model tensors, the saved optimizer step 750, and the
fixed validation NLL (difference 0.0). Initial fixed validation NLL was
1.3660 and final step-1750 NLL was 1.4865. Fixed training accuracy was also
only 50% at the selected checkpoint. This is observed underfitting/NULL
collapse on the fixed sets; the late loss rise does not establish overfitting.
The cause is not uniquely identified. Plausible contributors are the weak
candidate ranking inductive bias, varying answer values versus one easy NULL
target, the family shift, and the short exposure to unique examples.

The full, model-only [validation prediction artifact](working_qa/v1/analysis/validation_best/predictions.jsonl)
and [metrics](working_qa/v1/analysis/validation_best/metrics.json) used the
step-750 checkpoint without any rule fallback:

| Measure | Actual result | Frozen gate |
| --- | ---: | ---: |
| Answerable exact match | **0/160 (0%)** | ≥70% |
| Answerable token F1 | **0.000** | reported |
| Correct abstention | **160/160 (100%)** | ≥70% |
| Overall exact match | 160/320 (50%) | not sufficient alone |
| Non-abstaining source-span support | no predictions; undefined | 100% required |
| Harder multi-entity/paraphrase answerable EM | 0% / 0% | ≥10% each |
| Mean / median answer latency | 1.74 / 1.31 ms | reported |

The selected model made zero non-abstaining predictions, so span-support
rate is **undefined**, not 100%. The validation gate failed. The new sealed
test remains unopened and has **no reported test metrics**. The evaluator
technically refuses test mode without a passed validation lock.

During v1 main training, minimum sampled system-available memory was 1.16
GB, minimum `memory_pressure` free percentage 28%, peak worker RSS 82.9 MB,
swap-used growth 98.6 MB, cumulative swap-out growth 8.7 MB, and minimum
disk headroom 17.08 GB. The guard did not trip. During v2 tiny, swap used
grew **646.1 MB**, exceeding the unchanged 600 MB stop ceiling; the parent
terminated the worker. Minimum available memory was 1.09 GB, pressure-free
26%, peak worker RSS 70.0 MB, and minimum disk 17.00 GB. These are system
measurements and do not establish that this 1M-parameter worker alone caused
the swap growth. The observed pressure makes another training launch unsafe
under this experiment's safeguards. No macOS-wide GPU memory limit or guard
threshold was changed.

## Verification and next step

Ten new focused tests pass: split/label checks, train-only tokenizer and
span alignment, bidirectional visibility, finite gradients and parameter
update, checkpoint reproducibility, and guard triggers. The 38 existing
scratch core tests, nine prior extractive tests, and seven prior generative QA
tests (one earlier-version skip) also pass. The new v1 manifest and checkpoint
hashes verify. A first combined legacy QA test command failed only because
its `qa/` module path was omitted; rerunning that suite with its documented
`PYTHONPATH` passed. No production application test was needed because no
production file changed.

**NO-GO for presenting a working neural PDF QA model or running a longer
training job now.** The next safe experiment is to wait for swap activity to
stabilize, inspect whether the v2 pairwise reader can reach 8/8 tiny-set
memorization under a newly frozen remaining compute budget, and then require
the unchanged 320-example validation gate before opening this sealed test.
If that gate eventually passes, independently test an actual PDF passage
with retrieval provenance and real page metadata before connecting it to
the RAG application. Synthetic character offsets are not PDF page citations.

Reproduction and exact artifact locations are in
[working_qa/README.md](working_qa/README.md). The
[artifact hash ledger](working_qa/ARTIFACT_HASHES.json) fixes the new files.
