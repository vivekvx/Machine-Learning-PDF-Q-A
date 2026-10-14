"""Opt-in v3 pilot parent process with the unchanged scratch memory watchdog."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRATCH = ROOT.parent
sys.path[:0] = [str(ROOT), str(SCRATCH)]
from checkpoint import digest  # noqa: E402
from run import guard_reasons, preflight, snapshot  # noqa: E402
from v3_data import V3, load_verified, load_plan  # noqa: E402


def execute(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("A prior run cannot be overwritten")
    config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
    load_plan(config, train_windows, val_windows)
    if config["optimizer_steps_cap"] > 100 or config["wall_time_cap_seconds"] > 180 or config["batch_size"] != 1:
        raise ValueError("v3 pilot exceeds hard caps")
    before = preflight("smoke")
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(before, indent=2) + "\n")
    result = {"status": "preflight_refused" if not before["ready"] else "running", "preflight": before,
              "snapshots": [], "v3_manifest_sha256": digest(V3 / "manifest.json"),
              "pilot_config_sha256": digest(V3 / "pilot_config.v1.json"),
              "pilot_plan_sha256": digest(V3 / "pilot_plan.v1.json")}
    if not before["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    environment = os.environ.copy()
    environment["SLMA_SAFETY_PARENT"] = "cpython-v3"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "v3_worker.py"), str(run_dir)],
                                   cwd=SCRATCH, env=environment, stdout=log, stderr=subprocess.STDOUT)
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
    step_file = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in step_file.read_text().splitlines()] if step_file.exists() else []
    result["completed_steps"] = len(steps)
    result["training_target_tokens_processed"] = sum(s["tokens"] for s in steps)
    result["tokens_per_second"] = (sum(s["tokens"] for s in steps) / sum(s["duration_seconds"] for s in steps)) if steps else None
    result["mean_step_seconds"] = (sum(s["duration_seconds"] for s in steps) / len(steps)) if steps else None
    result["peak_sampled_worker_rss_bytes"] = max((s["process_rss_bytes"] or 0 for s in result["snapshots"]), default=None)
    result["minimum_system_available_bytes"] = min((s["available_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_pressure_free_percent"] = min((s["memory_pressure_free_percent"] for s in result["snapshots"] if s["memory_pressure_free_percent"] is not None), default=None)
    result["maximum_swap_used_growth_bytes"] = max((s["swap_used_bytes"] - before["resources"]["swap_used_bytes"] for s in result["snapshots"]), default=None)
    result["maximum_swap_out_growth_bytes"] = max((s["swap_out_total_bytes"] - before["resources"]["swap_out_total_bytes"] for s in result["snapshots"]), default=None)
    result["minimum_disk_free_bytes"] = min((s["disk_free_bytes"] for s in result["snapshots"]), default=None)
    worker_result = run_dir / "worker_result.json"
    if worker_result.exists():
        result["worker_result"] = json.loads(worker_result.read_text())
    result["status"] = ("completed" if result["exit_code"] == 0 and result.get("worker_result") and len(steps) == config["optimizer_steps_cap"]
                        else "stopped_by_guard" if result.get("stop_reasons") else "worker_failed")
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.execute and args.run_dir is None:
        parser.error("--execute requires --run-dir")
    output = execute(args.run_dir) if args.execute else {"mode": "preflight_only", **preflight("smoke")}
    print(json.dumps({key: value for key, value in output.items() if key != "snapshots"}, indent=2))
