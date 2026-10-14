# SLM-A synthetic evidence-conditioned QA

This is an isolated, from-scratch MLX experiment. It does not use QASPER,
CPython prose, pretrained weights/tokenizers, the PDF RAG application, or an
external model API. The v3 experiment is the interpretable result. v1 and v2
are retained as rejected design iterations; do not use their scores as evidence
of generalization.

## Reproduce data and checks in a fresh checkout

Use the pinned `research/slm/scratch/.venv` environment described in the
parent scratch README. No command below downloads model assets. The existing
version directories are immutable, so build commands require an empty version
directory in a separate checkout.

```sh
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python research/slm/scratch/qa/prepare.py --build
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python research/slm/scratch/qa/experiment.py --plan
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/qa -p 'test_qa.py' -v
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python research/slm/scratch/qa/experiment.py --preflight
```

The following commands run a **new bounded training and evaluation job**;
choose a new output directory. The parent process enforces the unchanged SLM-A
memory, swap, pressure, disk and wall-time guards. Evaluation reads the sealed
test partition only after checkpoint selection and refuses to overwrite an
earlier evaluation.

```sh
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python research/slm/scratch/qa/experiment.py --execute research/slm/scratch/qa/v3/run-NEW
SLMA_QA_VERSION=v3 research/slm/scratch/.venv/bin/python research/slm/scratch/qa/evaluate.py --run-dir research/slm/scratch/qa/v3/run-NEW
```

The frozen observed result is in `v3/run-001/evaluation.json`. The model failed
the held-out answerable-question threshold, so no model inference CLI is
provided as a working QA interface. `evaluate.py` contains the direct greedy
generation function used for the benchmark; its outputs are never replaced by
rule predictions.

## Data and metric definitions

- Passage contains a target entity sentence and a distractor entity sentence.
  When the target fact is absent, the distractor still contains the requested
  field. The required target is the exact `UNANSWERABLE` marker.
- Families, entity IDs and fact IDs are disjoint across train, validation and
  sealed test. v3 balances each requested field across both labels and uses
  opaque entity names. Numeric fact ranges are split-specific.
- The tokenizer is byte BPE trained only from training prompts and answers.
  UTF-8 byte fallback avoids unknown tokens. Loss covers answer tokens and
  answer EOS, with passage/question tokens masked.
- Exact match on answerable examples ignores case and token punctuation;
  abstention requires the exact marker after outer whitespace trim. Token F1
  uses lowercase alphanumeric tokens. Evidence support requires the exact
  normalized answer to occur in the target sentence. The baseline uses a
  rule-based field extractor on passage and question only.
- The v3 gate required at least 120 held-out examples, answerable exact match
  and evidence support at least 0.60, and exact abstention at least 0.80.

The synthetic benchmark measures this template task only. A perfect parser on
these templates is not evidence of general document QA capability.
