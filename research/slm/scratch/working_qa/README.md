# Scratch evidence QA reader (new, independent experiment)

This directory preserves a new synthetic-only QA experiment. The original
decoder-only scratch language model remains in `../model.py` and its prior
checkpoints are unchanged. These QA readers are **bidirectional extractive
encoders**, not generative language models. They use only randomly initialized
MLX weights and a byte BPE tokenizer fitted to the new training inputs.

`v1/` contains a frozen 2,400/320/320 train/validation/sealed-test dataset,
configuration, tokenizer, plan, guarded tiny and main training runs, selected
checkpoint, and validation predictions. `v2/` contains a preregistered
candidate-specific comparison; its tiny run was stopped by the live swap-growth
guard before it passed the 8/8 memorization gate. No v2 main run exists. The
new test file has never been opened for model selection or evaluation.

## Inspect verified artifacts

From the repository root:

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/prepare.py --verify
research/slm/scratch/.venv/bin/python -m unittest research/slm/scratch/working_qa/test_working_qa.py -v
cat research/slm/scratch/working_qa/v1/analysis/validation_best/metrics.json
cat research/slm/scratch/working_qa/v1/analysis/quality_audit.json
cat research/slm/scratch/working_qa/v2/runs/tiny/run.json
```

The scripts refuse to overwrite existing versioned files. In a **separate
clean copy** of this repository containing the source but not generated
`working_qa/v1` artifacts, the build/training sequence is:

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/prepare.py --build
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/train.py --plan
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/train.py --execute tiny
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/train.py --execute main
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/evaluate.py --split validation
```

The guarded `main` worker checks that tiny memorization passed. `test` mode in
the evaluator refuses access until full validation passes the frozen gate and
writes a selection lock. It did not pass here; therefore **do not run test
mode**. The `--execute` entrypoints call the existing scratch preflight and a
live parent watchdog. Direct workers refuse execution. Run commands only on
the pinned Apple Silicon MLX environment and only after other memory activity
has stabilized.

The deterministic `rule_baseline.py` is reported separately. It is never
called by either neural training or inference. Brackets delimit candidate
source values; deterministic code may enumerate them and copy a *neural*
choice, but cannot choose a value from the question in the neural condition.
