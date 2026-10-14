# SLM-A demo and viva guide

## What can be shown truthfully

The repository has an independently trained, randomly initialized
decoder-only language model under `research/slm/scratch/`; its architecture,
next-token loss, tokenizer and earlier synthetic/real-text results are in
`ARCHITECTURE.md`, `OVERFIT_REPORT.md` and `REAL_TEXT_PILOT_REPORT.md`.
It is a genuine small language model, but its prior evidence QA attempt did
not work. The new `working_qa/` system is a separate **extractive neural
Transformer**, not a generative language model. It learned 8/8 tiny training
examples, then failed unseen validation by abstaining on all 320 questions.
It must not be presented as a working PDF answerer.

To demonstrate the verified result on this Mac from the repository root:

```sh
research/slm/scratch/.venv/bin/python -m unittest research/slm/scratch/working_qa/test_working_qa.py -v
cat research/slm/scratch/working_qa/v1/analysis/validation_best/metrics.json
head -n 3 research/slm/scratch/working_qa/v1/analysis/validation_best/predictions.jsonl
cat research/slm/scratch/working_qa/v1/analysis/rule_validation.json
cat research/slm/scratch/working_qa/v2/runs/tiny/run.json
```

The prediction file contains actual network outputs, references, source-span
metadata and latency. The rule baseline file is explicitly separate and is
never substituted into neural outputs. The v2 run JSON shows the automatic
swap-growth stop. No safe neural question-answering CLI was released because
the fixed validation gate failed. The sealed new test set is still closed.

## Short oral explanation

1. **Language model:** Our original four-layer, width-128 causal Transformer
   learns next-token prediction from random weights. It passed a tiny
   memorization check, but this alone does not give useful document QA.
2. **Reader:** For QA we tried a bidirectional four-layer encoder. It sees a
   short question and passage, scores bracketed candidate value spans and
   an explicit NULL option. Selecting a value is neural; copying that chosen
   source substring and reporting its character offsets is deterministic.
3. **Training:** Data are original synthetic non-medical technical facts.
   Families and full entity names are split-disjoint; the byte BPE tokenizer
   sees training text only. Loss supervises the correct candidate or NULL.
4. **Actual result:** Tiny training memorization reached 8/8, but the best
   broad checkpoint abstained on all 320 validation cases. Answerable exact
   match is 0/160. The 70% answerable gate fails, so we did not inspect the
   new test labels or release a neural QA interface.
5. **Safety and limitations:** The MLX allocation was tens of MB for this
   short run, but unified system memory and swap are shared. A second
   comparison stopped when swap used grew 646 MB. Synthetic sentence patterns
   overlap heavily across train and validation; the template rule extractor
   scored 320/320, which does not imply open-domain language understanding.
   No PDF page numbers are known in these synthetic passages.

If asked whether the professor's complete deliverable is ready, answer
**no**: the from-scratch language model exists, but an unseen-example neural
QA demonstration and safe PDF integration remain unverified. Do not describe
the rule extractor, a pretrained Qwen model, or the failed reader as a
successful from-scratch generative QA model.
