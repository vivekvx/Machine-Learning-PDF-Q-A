"""Resource preflight and active watchdog for bounded SLM-A training only."""
import argparse
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

from checkpoint import digest
from data import DATA

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.v1.json"
REQUIRED = {"mlx": "0.32.3", "mlx-metal": "0.32.3", "psutil": "7.1.0"}


def pressure_free():
    try:
        result = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5)
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", result.stdout)
        return int(match.group(1)) if result.returncode == 0 and match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def snapshot(pid=None):
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = None
    if pid:
        try:
            rss = psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return {"timestamp": time.time(), "available_bytes": memory.available,
            "swap_allocated_total_bytes": swap.total,
            "swap_used_bytes": swap.used, "swap_free_bytes": swap.free,
            "swap_in_total_bytes": swap.sin, "swap_out_total_bytes": swap.sout,
            "memory_pressure_free_percent": pressure_free(), "process_rss_bytes": rss,
            "disk_free_bytes": shutil.disk_usage(ROOT).free}


def guard_reasons(snap, before, safety, *, preflight=False):
    reasons = []
    if preflight:
        if snap["available_bytes"] < safety["min_preflight_available_bytes"]:
            reasons.append("available memory below preflight floor")
        if snap["memory_pressure_free_percent"] is None or snap["memory_pressure_free_percent"] < safety["min_preflight_pressure_free_percent"]:
            reasons.append("memory pressure unavailable or below preflight floor")
        if snap["disk_free_bytes"] < safety["min_preflight_disk_free_bytes"]:
            reasons.append("disk headroom below preflight floor")
        # macOS can allocate additional swap files. swap.free is headroom in the
        # currently allocated files, not a fixed system swap capacity.
        if snap["swap_out_total_bytes"] - before["swap_out_total_bytes"] > safety["stop_swap_growth_bytes"]:
            reasons.append("swap-out activity above preflight ceiling")
        return reasons
    if snap["available_bytes"] < safety["stop_available_bytes"]:
        reasons.append("available memory below stop floor")
    if snap["swap_used_bytes"] - before["swap_used_bytes"] > safety["stop_swap_growth_bytes"]:
        reasons.append("swap growth above stop ceiling")
    if snap["swap_out_total_bytes"] - before["swap_out_total_bytes"] > safety["stop_swap_growth_bytes"]:
        reasons.append("swap-out activity above stop ceiling")
    if snap["process_rss_bytes"] is not None and snap["process_rss_bytes"] > safety["stop_process_rss_bytes"]:
        reasons.append("worker RSS above stop ceiling")
    if snap["memory_pressure_free_percent"] is None or snap["memory_pressure_free_percent"] < safety["stop_pressure_free_percent"]:
        reasons.append("memory pressure unavailable or below stop floor")
    if snap["disk_free_bytes"] < safety["min_preflight_disk_free_bytes"]:
        reasons.append("disk headroom below stop floor")
    return reasons


def preflight(mode, resume=None):
    config = json.loads(CONFIG.read_text())
    reasons = []
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        reasons.append("not running inside isolated scratch/.venv")
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        reasons.append("Apple Silicon macOS required")
    installed = {}
    for package, version in REQUIRED.items():
        try:
            installed[package] = importlib.metadata.version(package)
            if installed[package] != version:
                reasons.append(f"{package} version differs from pin")
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
            reasons.append(f"{package} is missing")
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    if data_manifest["config_sha256"] != digest(CONFIG):
        reasons.append("config hash differs from synthetic data manifest")
    for name, expected in data_manifest["file_sha256"].items():
        if digest(DATA / name) != expected:
            reasons.append(f"synthetic artifact hash mismatch: {name}")
    if data_manifest["actual_vocab"] != config["model"]["vocabulary_target"]:
        reasons.append("tokenizer vocabulary target not met")
    if mode == "overfit":
        if resume is None or not (Path(resume) / "metadata.json").exists():
            reasons.append("overfit requires saved smoke checkpoint")
        else:
            try:
                checkpoint = json.loads((Path(resume) / "metadata.json").read_text())
                prior = json.loads((Path(resume).parent / "run.json").read_text())
                if (checkpoint["step"] != config["training"]["smoke_steps"] or
                        prior["mode"] != "smoke" or prior["status"] != "completed"):
                    reasons.append("overfit requires a complete five-step smoke run")
            except (OSError, ValueError, KeyError):
                reasons.append("smoke checkpoint or run record cannot be verified")
    first = snapshot()
    time.sleep(1.0)
    now = snapshot()
    for observed in (first, now):
        reasons.extend(guard_reasons(observed, first, config["safety"], preflight=True))
    reasons = list(dict.fromkeys(reasons))
    return {"ready": not reasons, "reasons": reasons, "resources": now,
            "initial_resources": first, "sample_window_seconds": now["timestamp"] - first["timestamp"],
            "allocated_swap_free_is_fixed_limit": False,
            "disk_free_bytes": now["disk_free_bytes"], "hardware": {"platform": platform.platform(),
            "machine": platform.machine(), "total_memory_bytes": psutil.virtual_memory().total},
            "environment": {"python": sys.version.split()[0], **installed},
            "config_sha256": digest(CONFIG), "data_manifest_sha256": digest(DATA / "manifest.json"),
            "safety": config["safety"]}


def run(mode, run_dir, resume=None):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Run directory exists; frozen logs/checkpoints are not overwritten")
    before = preflight(mode, resume)
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "mode": mode, "preflight": before, "snapshots": []}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    config = json.loads(CONFIG.read_text())
    timeout = config["safety"][f"{mode}_timeout_seconds"]
    args = [sys.executable, "-u", str(ROOT / "train_worker.py"), "--mode", mode, "--run-dir", str(run_dir)]
    if resume:
        args.extend(["--resume", str(Path(resume).resolve())])
    environment = os.environ.copy()
    environment["SLMA_SAFETY_PARENT"] = "1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, before["resources"], config["safety"])
            if time.monotonic() - started > timeout:
                reasons.append("time cap exceeded")
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
    result["wall_seconds"] = time.monotonic() - started
    result["after"] = snapshot()
    step_file = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in step_file.read_text().splitlines()] if step_file.exists() else []
    result["completed_steps"] = len(steps)
    result["total_tokens"] = sum(x["tokens"] for x in steps)
    result["aggregate_step_tokens_per_second"] = (result["total_tokens"] / sum(x["duration_seconds"] for x in steps)) if steps else None
    result["peak_worker_rss_bytes"] = max((x["process_rss_bytes"] or 0 for x in result["snapshots"]), default=0)
    result["minimum_system_available_bytes"] = min((x["available_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_swap_free_bytes"] = min((x["swap_free_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_growth_bytes"] = max((x["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((x["memory_pressure_free_percent"] for x in result["snapshots"] if x["memory_pressure_free_percent"] is not None), default=None)
    result["minimum_disk_free_bytes"] = min((x["disk_free_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((x["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for x in result["snapshots"]), default=None)
    worker_result_file = run_dir / "worker_result.json"
    if worker_result_file.exists():
        result["worker_result"] = json.loads(worker_result_file.read_text())
    expected_steps = config["training"]["smoke_steps"] if mode == "smoke" else config["training"]["overfit_additional_steps_cap"]
    result["status"] = ("completed" if result["exit_code"] == 0 and result.get("worker_result") and
                        ((mode == "smoke" and len(steps) == expected_steps) or
                         (mode == "overfit" and 1 <= len(steps) <= expected_steps)) else
                        "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "overfit"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true", help="launch the guarded worker")
    args = parser.parse_args()
    if args.execute and args.run_dir is None:
        parser.error("--execute requires --run-dir")
    output = run(args.mode, args.run_dir, args.resume) if args.execute else {"mode": "preflight_only", **preflight(args.mode, args.resume)}
    print(json.dumps({key: value for key, value in output.items() if key != "snapshots"}, indent=2))
