# Extractive QA feasibility experiment

## Result

**NO-GO for a local neural QA demo or real-document deployment.** A fresh,
951,811-parameter bidirectional MLX Transformer was trained on synthetic
passage/question pairs with learned start, end and answerability heads. It did
not pass the preregistered validation gate. The selected checkpoint at step
600 scored **0/120 answerable exact spans and 0/120 correct abstentions** on
the full validation set; it predicted a span for all 240 examples. Its
held-out test score was 3/120 answerable exact spans and 28/120 correct
abstentions. The deterministic extractor scored 240/240 on both partitions.
No rule answer was substituted for a neural prediction.

| Frozen condition | Validation exact | Test exact | Test answerable EM | Test token F1 | Test gold-span support | Test unanswerable accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh extractive Transformer, step 600 | 0/240 | 31/240 (12.9%) | 3/120 (2.5%) | 0.131 | 3/120 (2.5%) | 28/120 (23.3%) |
| Rule extractor on the same passages | 240/240 | 240/240 | 120/120 | 1.000 | 120/120 | 120/120 |

The neural answerability confusion matrix on test was TP 98, FN 22, FP 92,
TN 28 (positive means the question is answerable). On validation it was TP
120, FN 0, FP 120, TN 0. Its validation answerability probability was near
0.503 on sampled examples; it often selected a fixed early token fragment
such as `tin M` instead of a field value. This points to weak generalization
under the new template family and absolute token positions. It is an observed
failure mode, not proof of a single root cause. The rule score is expected on
rigid generated templates and says nothing about open-domain QA.

The prior generative and current extractive runs use **different synthetic
corpora and held-out families**, so their percentages are not a controlled
head-to-head improvement claim. The extractive architecture avoided the
all-abstain behavior but failed both span selection and abstention. See
[individual predictions](extractive_qa/v1/run-001/test_neural_predictions.jsonl)
and the [evaluation summary](extractive_qa/v1/run-001/evaluation.json).

## Read-only diagnosis of the frozen generative checkpoint

Only its 1,200 training and 160 validation examples were read. The earlier
sealed test set was not reopened or rescored. The v3 generative target mask
starts at the first answer token and includes EOS; passage/question targets
are zero-masked. Every input fit its declared 192-token limit. The frozen
validation decoding used one token for `UNANSWERABLE`, followed by EOS, on all
160 examples; none reached the 16-token generation cap. Thus the previous
zero answerable score was a first-token choice, not output truncation or a
parser/marker mismatch.

| Original generative checkpoint | Gold first-token probability | `UNANSWERABLE` first-token probability | Gold first-token argmax | Answer-token CE |
| --- | ---: | ---: | ---: | ---: |
| Training answerable (600) | 0.046 | 0.262 | 1% | 2.394 |
| Training unanswerable (600) | 0.543 | 0.543 | 100% | 0.335 |
| Validation answerable (80) | 0.043 | 0.399 | 0% | 3.089 |
| Validation unanswerable (80) | 0.405 | 0.405 | 100% | 0.462 |

Balanced category counts mean answerable examples contributed far more
answer-token loss than unanswerable examples at the selected checkpoint.
This diagnosis does not isolate why the optimizer favored the repeated marker.
Detailed minimum/maximum probabilities, mask lengths and source hashes are
in [generative_diagnosis.json](extractive_qa/v1/generative_diagnosis.json).

## Extractive design and data controls

The [versioned config](extractive_qa/v1/config.json) fixes four encoder
layers, width 128, four attention heads, context 256, 1,024-token fresh byte
BPE vocabulary and batch size 1. The 951,811 parameters include independent
start, end and answerability heads. The QKV, feed-forward and normalization
building blocks are reused from SLM-A; the causal mask is deliberately removed.
Unit tests verify that changing a future token affects an earlier attention
position and that changing the question affects passage span logits. A causal
attention control test confirms the contrast. This model is an extractive
encoder, not a newly trained generative language model.

The original synthetic corpus has 1,800 training examples across three
template families, 240 validation examples in a fourth, and 240 sealed test
examples in a fifth. Each split is 50% answerable and balanced by requested
field. Opaque entity IDs, fact IDs and families are disjoint across splits;
each passage has a target and distractor with two field/value facts. An
unanswerable target lacks the asked field while the distractor contains it.
Values include method names, quantities, ports and owners. Exact answer
character offsets roundtrip to tokenizer spans; there is no annotation guess.
The tokenizer was fit on training inputs only. Maximum input lengths were 103,
139 and 131 tokens for train, validation and test, below the 192-token cap.
The maximum gold answer span on inspected training/validation examples was
four tokens, within the preregistered eight-token decoding span cap. See the
[manifest](extractive_qa/v1/manifest.json) and [frozen sample plan](extractive_qa/v1/plan.json).

The neural span support metric requires the predicted start and end to equal
the gold token span. Exact match compares normalized answer text;
unanswerable accuracy requires the exact `UNANSWERABLE` output. The rule
baseline sees passage and question only, resolves the exact target entity,
and extracts the requested bracketed value; its outputs and metrics are
stored separately. The validation demo gate required at least 200 examples,
60% answerable exact match and span support, and 80% unanswerable accuracy.
It failed before the sealed test file was opened. The selected checkpoint
was not changed after that evaluation.

## Training, safety and reproducibility

The preregistered cap was 1,600 AdamW optimizer steps at LR 0.0005, weight
decay 0.01 and a 180-second wall limit. Fixed, balanced 96-example training
and validation windows were evaluated every 200 steps. Selection used lowest
joint validation loss, with the earliest step winning ties. Patience stopped
the run at 1,200 steps after step 600 remained best. The initial, best and
final fixed-set joint losses were:

| Step | Training | Validation |
| ---: | ---: | ---: |
| 0 | 2.9135 | 3.1736 |
| 600, selected | 1.5997 | 3.0387 |
| 1,200, stopped | 1.5108 | 3.3773 |

Training processed 116,595 input tokens in 25.46 seconds wall time, at 5,612
input tokens/second across optimizer steps (mean step 17.31 ms). The preflight
passed with 1.76 GB system-available memory, 43% free memory pressure and
19.84 GB free disk. During the run, available memory stayed at or above
1.33 GB, free memory pressure at or above 32%, and disk headroom at or above
19.77 GB. Maximum sampled worker RSS was 80.6 MB; peak MLX allocation was
41,765,421 bytes. Swap used grew 0 bytes while cumulative swap-out activity
grew 6,946,816 bytes. Existing guard thresholds were unchanged and did not
trigger. Selected checkpoint restoration matched 33 model tensors exactly;
the restored fixed validation loss differed by 0.0. Full per-step gradient,
parameter, timing and resource records are in [run.json](extractive_qa/v1/run-001/run.json),
[steps.jsonl](extractive_qa/v1/run-001/steps.jsonl) and
[curve.jsonl](extractive_qa/v1/run-001/curve.jsonl).

The findings support only bounded training feasibility on this M2, not useful
QA ability. The new validation family tokenizes longer than training and the
model may exploit absolute positions; the current evidence does not separate
data shift, architecture, objective and sample-budget effects. Any next
revision should be preregistered with a **new** sealed test partition and
should diagnose candidate-value masking or position behavior using training
and validation examples first. Do not tune on these test results or release
this checkpoint as an answerer.

Rebuild instructions are in [extractive_qa/README.md](extractive_qa/README.md).
Nine focused tests passed, including offsets, split balance, attention
visibility, gradients, rule parsing, confusion-matrix orientation and
checkpoint reload. Existing RAG and earlier SLM-A artifacts remain unchanged.
