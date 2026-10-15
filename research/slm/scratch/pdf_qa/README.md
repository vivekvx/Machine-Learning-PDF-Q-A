# Offline research PDF assistant

This is a separate local FastAPI service. It does not change `server.py` or the existing RAG application. Its primary browser interface now offers **Local Qwen RAG** and **Hybrid Scratch + Qwen RAG**, plus the prior scratch-only pointer demo. PDF extraction and chunking come from `src/`; retrieval is local TF-IDF+FAISS in both generative modes. The existing main application has a different retrieval stack and remains separate. The hybrid uses the frozen scratch pointer only for factual questions with four legitimate candidates; the failed evidence checker is never used. Qwen3-0.6B is externally pretrained by the Qwen team. The pointer was trained from scratch by this project; it is an extractive selector, not a generative SLM.

From a fresh clone on Apple Silicon macOS, install Python 3.14 and `uv`, then run this **opt-in** setup from the repository root. It creates three isolated environments and downloads only the pinned official 4-bit Qwen snapshot. It never trains or downloads a scratch checkpoint. The project's 3.6 MB frozen pointer weight and tokenizer are included with this research service and verified against the committed selection ledger.

```sh
python3 research/slm/scratch/pdf_qa/setup_local.py --install --download-qwen
python3 research/slm/scratch/pdf_qa/setup_local.py
```

The second command is a read-only installation audit. It reports pointer and Qwen readiness separately and exits nonzero on missing or changed artifacts. The installer does not bypass the live memory guard: an installation may be complete yet local Qwen inference may still stop on an 8 GB Mac. If you want to prepare only the environments without downloading Qwen yet, omit `--download-qwen`; the audit will correctly report Qwen as incomplete.

To start an already prepared installation:

```sh
research/slm/scratch/pdf_qa/.venv/bin/uvicorn research.slm.scratch.pdf_qa.app:app --host 127.0.0.1 --port 8766
```

The Qwen environment is the isolated `research/slm/.venv` with `mlx-lm==0.32.0` and `mlx==0.32.3`. The pinned official model is `Qwen/Qwen3-0.6B-MLX-4bit` revision `173234aa840d113125e9f2271100ddbaf16c9620` in the ignored local folder `research/slm/.cache/qwen3-official-mlx-4bit`. The worker checks its weight, tokenizer, and license hashes before loading and never downloads during inference. The pointer's own `research/slm/scratch/.venv` and frozen checkpoint remain required only for eligible hybrid or scratch-only requests.

The server checks available memory, macOS memory pressure, swap growth and activity, process RSS, and disk space. It runs the Qwen and pointer workers sequentially with one GPU-heavy request at a time. An unsafe state returns HTTP 503. It does not treat free space in already allocated swap files as a fixed macOS limit. The service accepts PDFs up to 12 MB, 120 text pages, and one million extracted characters. It keeps uploaded indexes in memory only for the current server session; no OCR is provided.

The browser uses `/api/upload-memory`: it sends a bounded raw PDF request to localhost, and PyMuPDF extracts it from memory. The older multipart `/api/upload` remains for compatibility and may be spooled by the web framework; use the browser route when avoiding a temporary PDF copy matters. “Remove PDF from this session” deletes its in-memory index; restarting the server also clears all indexes. This is application-level deletion, not a claim about macOS-managed swap or the original file on your computer.

The current experimental configuration is [hybrid_config.v14.json](hybrid_config.v14.json). It selects up to four main-paper sections for an overview, rejects table-like text when looking for method or result prose, and excludes clearly marked references and appendices. It now prioritizes sentences that actually state an observed result over benchmark setup. Each selected excerpt retains its page and exact text offset. Broad summaries bypass the four-span pointer and use the same local Qwen generator as Local Qwen mode. A summary based on a few excerpts is explicitly labeled a *selective overview*. If Qwen puts an existing citation at the start of a summary bullet, the display moves that same ID to the end without changing the claim; the raw generated answer remains in call diagnostics. The browser can show selected passages without running a model when the memory guard stops inference. Incomplete, fragmentary, or unsupported generation is labeled as a draft rather than a finished answer. The earlier versioned configurations and previous experiment outputs remain unchanged.

On an 8 GB Mac, the MLX worker limits its cache of *freed* allocations to 64 MiB. A local retry of one user-supplied technical paper after that change completed in 4.1 seconds with 870 MB peak MLX allocation, 393 MB swap-used growth, and 1.23 GB minimum available memory; the prior request had stopped at 627 MB swap growth. These runs had different system conditions, so they do not prove the cache change alone caused the improvement. Version 14 also requests 128-token model prefill batches and requires 2 GB of system-available memory before starting Qwen. The 2 GB start floor is based on the narrow headroom observed in prior successful runs; it is not a guarantee. Runtime floors and the 600 MB live swap-growth ceiling remain unchanged. Changing the PDF does not itself fix system memory pressure. The app never retries automatically.

With v12, the same local technical-paper request returned a three-bullet overview in 5.5 seconds. The raw Qwen response used the right source IDs at the start of bullets; display-only normalization moved those IDs to the ends. Peak MLX allocation was 939 MB, maximum observed swap-used growth 492 MB, minimum available system memory 770 MB, and minimum pressure-free 17%. The automatic citation-support audit still requested manual review, so the browser labeled this a draft. This is one successful request close to the 700 MB runtime memory floor and 600 MB swap-growth ceiling, not a guarantee for other papers or concurrent workloads. No user PDF or extracted passages were written as part of these measurements.

In a later 44-page paper summary, the old runtime guard stopped Qwen after 778 MB of live swap growth. A guarded attempt with bounded prefill did not reach model loading because system-available memory fell below its preflight floor. The effect of bounded prefill on this paper is therefore unverified. The source planner did select four bounded exact excerpts from that paper, but selected passages alone are not an answer. No copy of that user PDF was added to this repository.

Open `http://127.0.0.1:8766/`, upload a readable non-medical technical PDF, choose a mode, and ask a factual question or request a paper summary. The browser shows the answer, the pointer usage or bypass reason, page-linked exact excerpts, coverage, and resource diagnostics. Unknown citation IDs and output truncation are explicitly labeled unverified. A valid ID links to real extracted text but does not establish semantic support; the local lexical warning is only a prompt for manual inspection.

The **Check Mac readiness for Qwen** button reports the current available memory and start floor without loading the model. This is a resource snapshot, not a promise that a subsequent request will succeed; full hash checks and live resource guards still run on every request. The cached Qwen model is already 4-bit quantized (128-value groups; approximately 317 MB of weights). On one busy session after closing browser tabs, memory pressure was 43–47% free and swap-out growth was zero over a 12-second sample, but available memory after preflight checks was only 1.76–1.84 GB against the 2 GB start floor. Previously measured successful inference began near 2 GB and approached the 700 MB runtime floor, so the start rule was not lowered on the basis of that sample. A separate earlier 44-page run exceeded the 600 MB live swap-growth guard. The scratch pointer cannot summarize a paper; the browser now explains this immediately when a summary is requested in pointer-only mode.

For the reproducible local comparison and limits, read [HYBRID_OFFLINE_RAG_REPORT.md](../../HYBRID_OFFLINE_RAG_REPORT.md) and [HYBRID_RESEARCH_EVALUATION.md](../../HYBRID_RESEARCH_EVALUATION.md).

## Historical scratch-only pointer mode

Choose “Scratch pointer only” in the same browser, or use the unchanged `/api/ask` route, for the original four-candidate extractive experiment. The pointer worker verifies the checkpoint and tokenizer hashes, then runs the original macOS preflight and guard. An answer is copied exactly from extracted PDF text, with a page number and character offsets. `INSUFFICIENT_CANDIDATES` means fewer than four representable spans, while `INSUFFICIENT_EVIDENCE` means a deterministic veto rejected the neural selection. Neither status establishes semantic unanswerability.

On ordinary research papers, the candidate extractor often cannot produce four valid short spans. In that case **the scratch model is not run**. Broad summaries, methodology explanations, and synthesis questions are outside its trained four-choice task; Hybrid routes them to externally pretrained Qwen. The browser now labels those outputs as Qwen-only when the scratch selector was skipped. An end-to-end installation can run the documented experimental modes, but it does **not** make the scratch pointer a general PDF question-answering model. Establishing that capability would require a new, licensed real-document training/evaluation protocol and an independently validated abstention mechanism.

## Checks and benchmark

```sh
research/slm/scratch/pdf_qa/.venv/bin/python -m unittest research.slm.scratch.pdf_qa.test_pdf_qa -v
research/slm/scratch/pdf_qa/.venv/bin/python -m unittest research.slm.scratch.pdf_qa.test_hybrid -v
research/slm/scratch/pdf_qa/.venv/bin/python -m unittest research.slm.scratch.pdf_qa.test_setup_local -v
venv/bin/python -m unittest test_rag_improvements -v
```

`benchmark/v1/` contains original synthetic PDFs and document-separated frozen development and sealed records. Its manifest includes the acceptance gate and SHA-256 hashes. Development predictions and the one interrupted sealed attempt are under `benchmark/results_v1/`. The sealed evaluator refuses another run. Do not regenerate v1 in place or use those results to tune the frozen pointer. For a future experiment, create a new benchmark version and pre-register its protocol before evaluation.

The service is a narrowly scoped research prototype. Candidate extraction currently recognizes short explicit method, numeric, and identifier patterns. Scanned PDFs, tables, multi-hop questions, broad summaries, and complex scientific prose are outside the verified scope. Exact substring provenance and surface vetoes do not prove that a selected value correctly answers a question.
