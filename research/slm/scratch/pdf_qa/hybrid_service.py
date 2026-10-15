"""Serialized, watchdog-protected bridge from the PDF service to local MLX."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from .hybrid import INSUFFICIENT, plan, validate_citations
from .pipeline import PDFIndex, validate_citation
from .support_audit import audit as audit_support

HERE = Path(__file__).resolve().parent
SLM = HERE.parents[1]
ROOT = HERE.parents[3]
CONFIG = json.loads((HERE / "hybrid_config.v14.json").read_text())
QWEN_PYTHON = SLM / ".venv/bin/python"
WORKER = HERE / "qwen_worker.py"
GPU_LOCK = threading.Lock()


def pressure_free():
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5)
        found = re.search(r"System-wide memory free percentage:\s*(\d+)%", out.stdout)
        return int(found.group(1)) if out.returncode == 0 and found else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def snapshot(pid=None):
    process_rss = None
    if pid is not None:
        try:
            process_rss = psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            pass
    swap = psutil.swap_memory()
    return {"available_bytes": psutil.virtual_memory().available,
            "pressure_free_percent": pressure_free(),
            "swap_used_bytes": swap.used, "swap_out_bytes": swap.sout,
            "swap_allocated_bytes": swap.total, "swap_free_allocated_bytes": swap.free,
            "disk_free_bytes": shutil.disk_usage(HERE).free,
            "process_rss_bytes": process_rss}


def reasons(current, initial, *, loading=False):
    problems = []
    available_floor = CONFIG["min_preflight_available_bytes"] if loading else CONFIG["stop_available_bytes"]
    if current["available_bytes"] < available_floor:
        problems.append(f"available memory {current['available_bytes'] / 1e9:.2f} GB below "
                        f"{available_floor / 1e9:.2f} GB {'start' if loading else 'runtime'} safety floor")
    minimum_pressure = (CONFIG["min_preflight_pressure_free_percent"] if loading
                        else CONFIG["stop_pressure_free_percent"])
    if current["pressure_free_percent"] is None or current["pressure_free_percent"] < minimum_pressure:
        problems.append("system memory pressure unsafe or unavailable")
    if current["disk_free_bytes"] < CONFIG["min_disk_free_bytes"]:
        problems.append("disk headroom below safety floor")
    swap_growth = current["swap_used_bytes"] - initial["swap_used_bytes"]
    if swap_growth > CONFIG["stop_swap_growth_bytes"]:
        problems.append(f"swap use grew {swap_growth / 1e6:.0f} MB beyond "
                        f"{CONFIG['stop_swap_growth_bytes'] / 1e6:.0f} MB safety ceiling")
    swap_out_growth = current["swap_out_bytes"] - initial["swap_out_bytes"]
    if swap_out_growth > CONFIG["stop_swap_growth_bytes"]:
        problems.append(f"swap-out activity grew {swap_out_growth / 1e6:.0f} MB beyond "
                        f"{CONFIG['stop_swap_growth_bytes'] / 1e6:.0f} MB safety ceiling")
    if current["process_rss_bytes"] is not None and current["process_rss_bytes"] > CONFIG["stop_process_rss_bytes"]:
        problems.append("worker process memory beyond safety ceiling")
    return problems


def preflight():
    errors = []
    model_dir = SLM / ".cache" / CONFIG["local_dir"]
    weight = model_dir / "model.safetensors"
    tokenizer = model_dir / "tokenizer_config.json"
    license_file = model_dir / "LICENSE"
    for path, expected in ((weight, CONFIG["model_sha256"]),
                           (tokenizer, CONFIG["tokenizer_config_sha256"]),
                           (license_file, CONFIG["license_sha256"])):
        if not path.is_file():
            errors.append("Pinned local Qwen artifact missing: " + path.name)
        else:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                errors.append("Pinned local Qwen artifact checksum mismatch: " + path.name)
    if not QWEN_PYTHON.is_file():
        errors.append("Isolated local Qwen runtime missing")
    else:
        # The model runtime is a separate venv; check pins with its interpreter.
        check = subprocess.run([str(QWEN_PYTHON), "-c",
                                "import importlib.metadata as m,json; print(json.dumps({x:m.version(x) for x in ('mlx-lm','mlx')}))"],
                               capture_output=True, text=True, timeout=10)
        if check.returncode:
            errors.append("Could not verify the isolated MLX package versions")
        else:
            installed = json.loads(check.stdout)
            if installed != {"mlx-lm": CONFIG["mlx_lm"], "mlx": CONFIG["mlx"]}:
                errors.append("Isolated MLX package versions differ from pins")
    first = snapshot()
    time.sleep(0.3)
    second = snapshot()
    errors.extend(reasons(first, first, loading=True))
    errors.extend(reasons(second, first, loading=True))
    return {"ready": not errors, "errors": list(dict.fromkeys(errors)), "resources": second,
            "allocated_swap_is_fixed_limit": False}


def invoke_worker(payload, before):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "NO_PROXY": "*"}
    proc = subprocess.Popen([str(QWEN_PYTHON), str(WORKER)], cwd=str(ROOT), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.stdin.write(json.dumps(payload, ensure_ascii=False))
    proc.stdin.close()
    started = time.monotonic()
    samples = []
    stopped = None
    try:
        while proc.poll() is None:
            now = snapshot(proc.pid)
            samples.append(now)
            problems = reasons(now, before)
            if problems:
                stopped = "; ".join(problems)
                proc.terminate()
                break
            if time.monotonic() - started > CONFIG["worker_timeout_seconds"]:
                stopped = "model request exceeded its wall-time limit"
                proc.terminate()
                break
            time.sleep(0.25)
        try:
            stdout, _stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _stderr = proc.communicate(timeout=5)
        if stopped:
            raise RuntimeError("Resource guard stopped local inference: " + stopped)
        try:
            result = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Local Qwen worker returned no structured result") from exc
        if proc.returncode or result.get("status") != "complete":
            raise RuntimeError(result.get("error", "Local Qwen inference failed"))
        result["watchdog"] = {
            "samples": len(samples), "minimum_available_bytes": min([x["available_bytes"] for x in samples], default=None),
            "minimum_pressure_free_percent": min([x["pressure_free_percent"] for x in samples if x["pressure_free_percent"] is not None], default=None),
            "peak_process_rss_bytes": max([x["process_rss_bytes"] or 0 for x in samples], default=None),
            "swap_used_growth_bytes": max([x["swap_used_bytes"] - before["swap_used_bytes"] for x in samples], default=0),
            "swap_out_growth_bytes": max([x["swap_out_bytes"] - before["swap_out_bytes"] for x in samples], default=0),
            "minimum_disk_free_bytes": min([x["disk_free_bytes"] for x in samples], default=None),
            "worker_wall_ms": (time.monotonic() - started) * 1000}
        return result
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def answer(index: PDFIndex, question: str, mode: str, pointer_request):
    if mode not in ("qwen", "hybrid"):
        raise ValueError("Choose Local Qwen RAG or Hybrid Scratch + Qwen RAG")
    if not question.strip() or len(question) > 300:
        raise ValueError("Enter a question of at most 300 characters")
    if not GPU_LOCK.acquire(blocking=False):
        raise RuntimeError("A local model request is already running; try again when it finishes")
    started = time.monotonic()
    try:
        prepared = plan(index, question, max_sources=12)
        if not prepared["sources"]:
            return {"status": "INSUFFICIENT_EVIDENCE", "answer": INSUFFICIENT,
                    "intent": prepared["intent"], "sources": [], "pointer": {"used": False, "reason": "unsupported_request"}}
        # A zero lexical hit is a conservative stop for short fact lookups;
        # broad summaries select representative sections by design.
        if prepared["intent"] == "factual" and prepared["top_score"] <= 0:
            return {"status": "INSUFFICIENT_EVIDENCE", "answer": INSUFFICIENT,
                    "intent": prepared["intent"], "sources": prepared["sources"],
                    "pointer": {"used": False, "reason": "no_retrieval_overlap"}}
        if prepared["intent"] == "factual":
            # A requested measured field must occur in the retrieved evidence.
            # This is a conservative abstention check, never an answer source.
            asked_fields = [term for term in ("accuracy", "precision", "recall", "latency", "throughput", "memory")
                            if re.search(rf"\b{term}\b", question, re.I)]
            if asked_fields and not any(re.search(rf"\b{field}\b", source["text"], re.I)
                                        for field in asked_fields for source in prepared["sources"]):
                return {"status": "INSUFFICIENT_EVIDENCE", "mode": mode,
                        "answer": INSUFFICIENT, "intent": prepared["intent"], "sources": [],
                        "retrieved_sources": prepared["sources"], "coverage": prepared["coverage"],
                        "pointer": {"used": False, "reason": "requested_measured_field_absent_from_evidence"}}
        checked = preflight()
        if not checked["ready"]:
            raise RuntimeError("Memory/model preflight refused: " + "; ".join(checked["errors"]))
        pointer = {"used": False, "reason": "qwen_only_mode" if mode == "qwen" else "broad_query_bypasses_pointer"}
        guidance = None
        if mode == "hybrid" and prepared["intent"] == "factual":
            candidate_data = index.retrieve_candidates(question)
            if len(candidate_data["candidates"]) >= 4:
                selected = pointer_request(question, candidate_data["candidates"])
                if selected["status"] == "SELECTED":
                    candidate = selected["four_candidates"][selected["candidate_index"]]
                    if not validate_citation(index, candidate):
                        raise RuntimeError("Frozen pointer returned a span without exact PDF provenance")
                    source = next((s for s in prepared["sources"] if s["chunk_index"] == candidate["chunk_index"]
                                   and candidate["sentence"] in s["text"]), None)
                    if source:
                        guidance = {"value": selected["prediction"], "source_id": source["id"]}
                        pointer = {"used": True, "reason": "four_valid_source_candidates", "guidance": guidance,
                                   "neural_candidate_index": selected["candidate_index"],
                                   "pointer_latency_ms": selected["pointer_latency_ms"],
                                   "pointer_peak_mlx_bytes": selected["peak_mlx_bytes"]}
                    else:
                        pointer = {"used": True, "reason": "selected_span_not_in_common_qwen_context_no_guidance",
                                   "neural_candidate_index": selected["candidate_index"]}
                else:
                    pointer = {"used": False, "reason": selected.get("reason", "pointer_input_rejected")}
            else:
                pointer = {"used": False, "reason": "fewer_than_four_legitimate_candidates"}
        if reasons(snapshot(), checked["resources"]):
            raise RuntimeError("Resource guard refused after pointer inference")
        result = invoke_worker({"mode": mode, "question": question, "intent": prepared["intent"],
                                "sources": prepared["sources"], "pointer_guidance": guidance}, checked["resources"])
        cited = result["citations"]
        support = audit_support(result["answer"], prepared["sources"])
        if not result.get("calls"):
            raise RuntimeError("Local generator returned no call metadata")
        final_call = result["calls"][-1]
        if not cited["valid"]:
            status = "INVALID_CITATION_UNVERIFIED"
        elif final_call["finish_reason"] == "length":
            status = "TRUNCATED_UNVERIFIED"
        elif result.get("summary_format_issues"):
            status = "FORMAT_UNVERIFIED"
        elif prepared["intent"] in ("whole_paper_summary", "section_summary") and re.search(
                r"(?im)^\s*(?:[-*]\s*)?claim\s*\[E\d+\]", result["answer"]):
            status = "FORMAT_UNVERIFIED"
        elif result["answer"] == INSUFFICIENT:
            status = "INSUFFICIENT_EVIDENCE"
        else:
            status = "ANSWERED" if cited["has_citation"] else "UNCITED_UNVERIFIED"
        return {"status": status, "mode": mode, "intent": prepared["intent"],
                "answer": result["answer"], "citation_status": cited,
                "automated_support_warning": support,
                "sources": [s for s in prepared["sources"] if s["id"] in cited["cited_ids"]],
                "retrieved_sources": prepared["sources"], "coverage": prepared["coverage"],
                "pointer": pointer, "retrieval_latency_ms": prepared["retrieval_latency_ms"],
                "generation": {k: result[k] for k in ("model_load_ms", "peak_mlx_bytes", "process_rss_bytes", "calls", "watchdog", "offline_network_blocked", "model_revision", "model_sha256", "prompt_version", "final_raw_source_ids")},
                "summary_format_issues": result.get("summary_format_issues", []),
                "output_normalization": result.get("output_normalization", "none"),
                "prefill_step_size": result.get("prefill_step_size"),
                "total_latency_ms": (time.monotonic() - started) * 1000}
    finally:
        GPU_LOCK.release()
