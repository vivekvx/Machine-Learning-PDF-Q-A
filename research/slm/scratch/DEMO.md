# Offline SLM-A pointer demo

This command runs the **trained from-scratch neural pointer**. It assumes the question is answerable from **exactly four bracketed values** in a short passage. The separately trained answerability head did not pass validation and is not part of this demo. An unsupported question can receive a false answer; do not treat this as unrestricted document QA or connect it to the PDF RAG application.

From the repository root on the Apple Silicon machine with the existing `research/slm/scratch/.venv`:

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/demo.py \
  --passage 'For silver router, the method is [ hash join ]. For silver router, the metric is [ 48 ms ]. For amber cache, the method is [ range scan ]. For amber cache, the metric is [ 96 ms ].' \
  --question 'What is the metric for amber cache?'
```

Verified output for this new illustrative passage: `96 ms`, character offsets `172:177`, supporting sentence `For amber cache, the metric is [ 96 ms ].`, uncalibrated pointer confidence `0.8256`, and model-only latency `56.8 ms` on the measured run. The JSON also states `answerability_checked: false` and `rule_fallback_used: false`. First-call and system load affect latency; the 240-question validation mean was 6.00 ms per model call.

The command first verifies frozen input hashes and the selected checkpoint, then runs the unchanged memory preflight and live guard. It raises an error for missing markers, a candidate count other than four, excess token length, or unsafe resources. There is no rule-based answer fallback.

## Verify the existing experiment

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/prepare.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/answerability/prepare.py --verify
research/slm/scratch/.venv/bin/python -m unittest research/slm/scratch/pointer_qa/test_pointer.py research/slm/scratch/pointer_qa/answerability/test_answerability.py -v
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/hash_artifacts.py --verify
research/slm/scratch/.venv/bin/python research/slm/scratch/working_qa/hash_artifacts.py --verify
```

The two new tests contain no sealed-test answer reads. The existing output directories deliberately reject overwrite. **Verification is read-only; the following rebuild recipe is for a fresh versioned workspace, not this frozen run.**

## Rebuild protocol from fresh scratch experiment inputs

All stages are explicit and opt-in. Start from a clean versioned copy with the same source code/configs and the existing scratch MLX environment. Do not remove or overwrite the current experiment to replay it.

```bash
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/prepare.py --build
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/train.py --plan
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/train.py --execute tiny
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/v2/train.py --plan
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/v2/train.py --execute tiny
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/v2/train.py --execute main
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/evaluate.py --validation
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/answerability/prepare.py --build
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/answerability/train.py --plan
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/answerability/train.py --execute
research/slm/scratch/.venv/bin/python research/slm/scratch/pointer_qa/answerability/evaluate_predicted.py
```

The v1 tiny attempt is intentionally retained as a failed result; v2 is the one preregistered correction. Each training entrypoint invokes a preflight and parent watchdog. Child workers reject direct invocation without the watchdog environment marker. The frozen plans bind data, tokenizer, configs, source hashes, sample order, and split IDs; checkpoint metadata binds weights and optimizer state. The last step is **validation only**. The new sealed test remains unopened because answerability failed its gate.

See [`POINTER_QA_REPORT.md`](POINTER_QA_REPORT.md) for measured accuracy, resource use, failure modes, and limits.
