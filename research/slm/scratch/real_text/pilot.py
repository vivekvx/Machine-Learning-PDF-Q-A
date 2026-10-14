"""Separate parent watchdog for the official-documentation pilot."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "v1"
sys.path.insert(0, str(ROOT.parent))
from run import guard_reasons, preflight, snapshot  # noqa: E402
from worker import load_verified  # noqa: E402


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Versioned run path already exists")
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    if config["max_optimizer_steps"] > 100 or config["wall_time_cap_seconds"] > 180:
        raise ValueError("Pilot safety cap exceeded")
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running",
              "preflight": before, "snapshots": [], "corpus_manifest_sha256": __import__("hashlib").sha256((V1 / "manifest.json").read_bytes()).hexdigest()}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    env = os.environ.copy()
    env["SLMA_SAFETY_PARENT"] = "real-text-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "worker.py"), str(run_dir)],
                                   cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, before["resources"], before["safety"])
            if time.monotonic() - started > config["wall_time_cap_seconds"]:
                reasons.append("parent wall-time cap exceeded")
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
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text().splitlines()] if (run_dir / "steps.jsonl").exists() else []
    result["completed_steps"] = len(steps)
    result["tokens_per_second"] = sum(x["tokens"] for x in steps) / sum(x["duration_seconds"] for x in steps) if steps else None
    result["mean_step_seconds"] = sum(x["duration_seconds"] for x in steps) / len(steps) if steps else None
    result["peak_worker_rss_bytes"] = max((x["process_rss_bytes"] or 0 for x in result["snapshots"]), default=None)
    result["minimum_available_bytes"] = min((x["available_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((x["memory_pressure_free_percent"] for x in result["snapshots"] if x["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_growth_bytes"] = max((x["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for x in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((x["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for x in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((x["disk_free_bytes"] for x in result["snapshots"]), default=None)
    if (run_dir / "worker_result.json").exists():
        result["worker_result"] = json.loads((run_dir / "worker_result.json").read_text())
    result["status"] = "completed" if result["exit_code"] == 0 and result.get("worker_result") and 1 <= len(steps) <= config["max_optimizer_steps"] else "stopped_by_guard" if result.get("stop_reasons") else "worker_failed"
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and not args.run_dir:
        parser.error("--execute requires --run-dir")
    print(json.dumps(execute(args.run_dir) if args.execute else preflight("smoke"), indent=2))
