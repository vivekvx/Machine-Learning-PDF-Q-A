# Run the local research PDF assistant

This prototype runs at **[http://127.0.0.1:8766/](http://127.0.0.1:8766/)** on the same Mac. Its preferred mode is **Local Qwen RAG**. **Hybrid Scratch + Qwen RAG** is an experimental comparison; the scratch pointer chooses among four source spans when a factual question meets its input contract. The separate **Scratch pointer only** mode remains available. The older production RAG application is unchanged.

## Start or restart

From `/Users/vivek/Machine-Learning-PDF-Q-A`:

```sh
research/slm/scratch/pdf_qa/.venv/bin/uvicorn research.slm.scratch.pdf_qa.app:app --host 127.0.0.1 --port 8766
```

Open the URL above in a browser. `/api/status` displays whether the local official Qwen snapshot is installed. The service uses the pinned model in `research/slm/.cache/qwen3-official-mlx-4bit/` and refuses a changed weight, tokenizer or license file. It will not download a model during a request. Keep the terminal open while testing. To stop, press Ctrl-C in that terminal.

If setting up on another Mac, first install dependencies from [`requirements.lock`](scratch/pdf_qa/requirements.lock) into the separate scratch service environment as shown in the [service README](scratch/pdf_qa/README.md), and use the existing compatible MLX environment for Qwen. The official Hugging Face snapshot is `Qwen/Qwen3-0.6B-MLX-4bit`, exact revision `173234aa840d113125e9f2271100ddbaf16c9620`. With rights, disk and memory checked, its one-time setup is:

```sh
research/slm/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B-MLX-4bit",
    revision="173234aa840d113125e9f2271100ddbaf16c9620",
    local_dir="research/slm/.cache/qwen3-official-mlx-4bit",
)
PY
```

That is a **setup-only network operation**. After setup, inference reads local files and its worker blocks network connections. Qwen is a pretrained external model by the Qwen team, licensed under the pinned Apache-2.0 notice stored [here](scratch/pdf_qa/QWEN_APACHE_2.0_LICENSE.txt). The pointer alone was trained from random initialization in this project. Do not describe the complete assistant as a from-scratch generative SLM.

## Try a PDF

1. Choose a readable, non-medical technical PDF that you are allowed to process. Upload it and wait for page and indexed-chunk counts. No OCR is available for scanned pages.
2. Leave **Local Qwen RAG** selected for a practical first test. Ask a short fact such as a stated method or measured result. Expand the cited excerpt and check that its actual page text supports the answer.
3. Ask “Give me a summary of this paper.” The app samples multiple sections within a fixed context budget and shows coverage. For large or irregular documents it may stop with an explicit resource or evidence error; it does not promise whole-paper comprehension.
4. Select **Hybrid Scratch + Qwen RAG** and repeat an eligible factual question. The result explains whether the frozen pointer was used or bypassed. Its guidance is not verified truth.
5. Ask a question with no stated answer. Inspect the evidence and uncertainty status yourself. A successful deterministic veto on one kind of missing metric does not prove broad abstention reliability.

The same uploaded document can be questioned repeatedly without re-uploading until the server restarts. A second upload gets a distinct document ID and index. Uploaded PDF text is held in the service process for that session; it is not added to training or sent to a remote inference API. The UI caps uploads at 12 MB, 120 extracted text pages, and one million extracted characters. Errors display a readable message rather than a stack trace.

The browser's current upload path reads PDF bytes in memory and does not make an application-managed PDF copy. Use **Remove PDF from this session** when done; it clears that document's index from the running service. **Preview source passages** shows selected excerpts without running Qwen and labels them as passages rather than an answer. When swap or memory growth trips the safety guard, the UI explains the stop and offers the preview; it does not retry automatically. The earlier v8 evaluation remains unchanged, and the v9 long-document answer behavior is awaiting a safe guarded inference run.

The answer panel shows page-numbered exact text, evidence IDs, citation warnings, response time, and resource diagnostics. **An ID that resolves to a real passage does not guarantee that the passage supports the claim.** The four-page synthetic check worked for explicit values, but a CPython real-text check found a wrong citation. Please verify important claims against the original PDF. See the [implementation report](HYBRID_OFFLINE_RAG_REPORT.md) and [paired evaluation](HYBRID_RESEARCH_EVALUATION.md).

## Quick local checks

```sh
curl -s http://127.0.0.1:8766/api/status
research/slm/scratch/pdf_qa/.venv/bin/python -m unittest research.slm.scratch.pdf_qa.test_hybrid research.slm.scratch.pdf_qa.test_pdf_qa -v
venv/bin/python -m unittest test_rag_improvements -v
```

If `/api/status` fails, the server process stopped. Restart with the command above. A 503 response means the resource preflight or watchdog declined to run; close other heavy apps and inspect local memory pressure before retrying. Do not relax the guards to obtain a demo answer.
