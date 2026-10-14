# SLM-A extractive QA v1

This experiment trains a fresh bidirectional Transformer encoder from random
weights on original synthetic technical passages. It shares low-level MLX
building blocks with SLM-A but is **not** a decoder-only generative SLM. It
does not use pretrained weights, pretrained tokenizers, APIs, copyrighted
corpora, QASPER, or the production PDF RAG application.

The frozen run is `v1/run-001`. In a fresh checkout with an empty `v1/`
artifact directory, rebuild and run with the pinned `scratch/.venv`:

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/prepare.py --build
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/train.py --plan
research/slm/scratch/.venv/bin/python -m unittest discover -s research/slm/scratch/extractive_qa -p 'test_extractive.py' -v
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/train.py --preflight
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/train.py --execute research/slm/scratch/extractive_qa/v1/run-NEW
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/evaluate.py --run-dir research/slm/scratch/extractive_qa/v1/run-NEW
```

The first four commands are data preparation/checks. `--execute` starts a new
bounded training job only if the existing SLM-A resource preflight passes. Its
parent process enforces live available-memory, memory-pressure, swap-growth,
RSS, disk, and wall-time abort conditions. Evaluation refuses to overwrite an
existing result. It scores validation and writes the validation gate **before**
opening the sealed test partition once.

The separate frozen generative checkpoint diagnosis is reproducible without
test data:

```sh
research/slm/scratch/.venv/bin/python research/slm/scratch/extractive_qa/diagnose_generative.py
```

That command writes once to `v1/generative_diagnosis.json`; use a fresh
checkout for reproduction. All train, validation and test artifacts contain
only generated text. The tokenizer is byte BPE fit on extractive training
inputs only. Exact target character spans are converted to token spans and
checked by roundtrip decoding. The encoder uses full self-attention so span
positions can see the entire passage and later question. Its start/end logits
are limited to passage positions; answerability is a separate learned head.
The neural output is never replaced by the deterministic extractor.

The preregistered validation gate did not pass, so no model inference demo is
released. The selected model and its raw predictions remain available as
research artifacts.
