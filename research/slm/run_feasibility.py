"""Preflight and watchdog for the bounded training worker; never edits system limits."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "training_feasibility_run.json"
LOG = ROOT / "training_worker.log"
STEPS = ROOT / "training_steps.jsonl"
MODEL = ROOT / ".cache/qwen3-0.6b-4bit"
MODEL_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
EXPECTED_MODEL_SHA = "392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2"
MIN_AVAILABLE = 1_200_000_000
MIN_SWAP_FREE = 750_000_000
STOP_AVAILABLE = 700_000_000
STOP_SWAP_FREE = 400_000_000
STOP_SWAP_GROWTH = 600_000_000
MAX_RSS = 2_500_000_000
TIMEOUT_SECONDS = 240


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pressure_free():
    result = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", result.stdout)
    return int(match.group(1)) if match else None


def snapshot(pid=None):
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = None
    if pid:
        try:
            rss = psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return {"time": time.time(), "available_bytes": memory.available,
            "swap_used_bytes": swap.used, "swap_free_bytes": swap.free,
            "process_rss_bytes": rss, "memory_pressure_free_percent": pressure_free()}


def preflight():
    if OUT.exists() or LOG.exists() or STEPS.exists() or (ROOT / "feasibility_adapter").exists():
        raise RuntimeError("Existing run artifacts: move them before a new run")
    model_sha = sha(MODEL / "model.safetensors")
    if model_sha != EXPECTED_MODEL_SHA:
        raise RuntimeError("Local pinned 4-bit model hash mismatch")
    manifest = json.loads((ROOT / "feasibility_data/manifest.json").read_text())
    for name in ("train", "valid"):
        if sha(ROOT / f"feasibility_data/{name}.jsonl") != manifest["file_sha256"][name]:
            raise RuntimeError(f"Sample checksum mismatch: {name}")
    now = snapshot()
    disk = shutil.disk_usage(ROOT)
    reasons = []
    if now["available_bytes"] < MIN_AVAILABLE:
        reasons.append("available memory below 1.2 GB")
    if now["swap_free_bytes"] < MIN_SWAP_FREE:
        reasons.append("free swap below 0.75 GB")
    if now["memory_pressure_free_percent"] is None or now["memory_pressure_free_percent"] < 20:
        reasons.append("system memory-pressure free percentage below 20 or unavailable")
    if disk.free < 1_000_000_000:
        reasons.append("disk free below 1 GB")
    return {"snapshot": now, "disk_free_bytes": disk.free, "model_sha256": model_sha,
            "model_revision": MODEL_REVISION,
            "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                         "physical_memory_bytes": memory_total()},
            "software": {name: importlib.metadata.version(name) for name in
                         ("mlx-lm", "mlx", "transformers", "psutil")},
            "sample_manifest_sha256": sha(ROOT / "feasibility_data/manifest.json"),
            "reasons": reasons}


def memory_total():
    return psutil.virtual_memory().total


def run():
    before = preflight()
    result = {"preflight": before, "status": "preflight_refused" if before["reasons"] else "running",
              "limits": {"min_available_bytes": MIN_AVAILABLE, "min_swap_free_bytes": MIN_SWAP_FREE,
                         "stop_available_bytes": STOP_AVAILABLE, "stop_swap_free_bytes": STOP_SWAP_FREE,
                         "stop_swap_growth_bytes": STOP_SWAP_GROWTH, "max_rss_bytes": MAX_RSS,
                         "timeout_seconds": TIMEOUT_SECONDS}, "snapshots": []}
    if before["reasons"]:
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        return result
    started = time.monotonic()
    with LOG.open("w", encoding="utf-8") as log:
        worker_env = os.environ.copy()
        worker_env["SLM_FEASIBILITY_WORKER"] = "1"
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "training_child.py")],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=worker_env)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = []
            if snap["available_bytes"] < STOP_AVAILABLE:
                reasons.append("available memory below 700 MB")
            if snap["swap_free_bytes"] < STOP_SWAP_FREE:
                reasons.append("free swap below 400 MB")
            if snap["swap_used_bytes"] - before["snapshot"]["swap_used_bytes"] > STOP_SWAP_GROWTH:
                reasons.append("swap growth above 600 MB")
            if snap["process_rss_bytes"] and snap["process_rss_bytes"] > MAX_RSS:
                reasons.append("worker RSS above 2.5 GB")
            if snap["memory_pressure_free_percent"] is not None and snap["memory_pressure_free_percent"] < 10:
                reasons.append("system memory-pressure free percentage below 10")
            if time.monotonic() - started > TIMEOUT_SECONDS:
                reasons.append("240-second timeout")
            if reasons:
                result["stop_reasons"] = reasons
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.5)
        result["exit_code"] = process.wait()
    result["elapsed_seconds"] = time.monotonic() - started
    result["after"] = snapshot()
    result["completed_steps"] = len(STEPS.read_text().splitlines()) if STEPS.exists() else 0
    result["status"] = "completed" if result["exit_code"] == 0 and result["completed_steps"] == 5 else (
        "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually launch the bounded five-step worker")
    cli = parser.parse_args()
    if cli.execute:
        print(json.dumps({k: v for k, v in run().items() if k != "snapshots"}, indent=2))
    else:
        try:
            print(json.dumps({"mode": "preflight_only", **preflight()}, indent=2))
        except RuntimeError as exc:
            print(json.dumps({"mode": "preflight_only", "ready": False, "reason": str(exc)}, indent=2))
