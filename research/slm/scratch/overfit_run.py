"""Separate 200-step maximum watchdog for the synthetic memorization probe."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from checkpoint import digest
from run import CONFIG, ROOT, guard_reasons, preflight, snapshot

EXPERIMENT = ROOT / "overfit/v1"
CONFIG_FILE = EXPERIMENT / "config.json"
MANIFEST_FILE = EXPERIMENT / "manifest.json"


def validate_experiment():
    config = json.loads(CONFIG_FILE.read_text())
    manifest = json.loads(MANIFEST_FILE.read_text())
    base = json.loads(CONFIG.read_text())
    reasons = []
    if not manifest["synthetic_only"] or manifest["academic_qa_records"] != 0:
        reasons.append("overfit data must be synthetic only")
    if manifest["model_config_sha256"] != digest(CONFIG):
        reasons.append("base model config hash mismatch")
    if manifest["tokenizer_sha256"] != digest(ROOT / "data/v1/tokenizer.json"):
        reasons.append("tokenizer hash mismatch")
    if manifest["data_sha256"] != digest(EXPERIMENT / "data.jsonl"):
        reasons.append("overfit data hash mismatch")
    if manifest["builder_sha256"] != digest(ROOT / "prepare_overfit.py"):
        reasons.append("overfit data builder changed")
    if manifest["sequence_count"] != 6:
        reasons.append("overfit fixture must contain six sequences")
    if (not 1 <= config["max_optimizer_steps"] <= 200 or
            not 0 < config["wall_time_cap_seconds"] <= 120):
        reasons.append("overfit step or wall-time cap exceeds hard limit")
    if config["seed"] != base["seed"] or config["batch_size"] != base["model"]["batch_size"]:
        reasons.append("overfit seed or batch size differs from base configuration")
    if config["optimizer"] != "AdamW" or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        reasons.append("invalid overfit optimizer or learning rate")
    return config, reasons


def overfit_preflight():
    base_check = preflight("smoke")
    experiment, reasons = validate_experiment()
    return {**base_check, "ready": base_check["ready"] and not reasons,
            "reasons": base_check["reasons"] + reasons,
            "experiment_config_sha256": digest(CONFIG_FILE),
            "fixture_manifest_sha256": digest(MANIFEST_FILE),
            "worker_source_sha256": digest(ROOT / "overfit_worker.py"),
            "step_cap": experiment["max_optimizer_steps"],
            "wall_time_cap_seconds": experiment["wall_time_cap_seconds"]}


def run_overfit(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("Run directory already exists; old results are immutable")
    readiness = overfit_preflight()
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(json.dumps(readiness, indent=2) + "\n")
    result = {"status": "preflight_refused" if not readiness["ready"] else "running",
              "preflight": readiness, "snapshots": []}
    if not readiness["ready"]:
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    experiment = json.loads(CONFIG_FILE.read_text())
    base = json.loads(CONFIG.read_text())
    environment = os.environ.copy()
    environment["SLMA_SAFETY_PARENT"] = "overfit-v1"
    started = time.monotonic()
    with (run_dir / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "overfit_worker.py"),
                                    "--run-dir", str(run_dir)], cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            snap = snapshot(process.pid)
            result["snapshots"].append(snap)
            reasons = guard_reasons(snap, readiness["resources"], base["safety"])
            if time.monotonic() - started > experiment["wall_time_cap_seconds"]:
                reasons.append("overfit wall-time cap exceeded")
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
    steps_file = run_dir / "steps.jsonl"
    steps = [json.loads(line) for line in steps_file.read_text().splitlines()] if steps_file.exists() else []
    result["completed_steps"] = len(steps)
    result["total_tokens"] = sum(row["tokens"] for row in steps)
    result["aggregate_step_tokens_per_second"] = (
        result["total_tokens"] / sum(row["duration_seconds"] for row in steps)) if steps else None
    samples = result["snapshots"]
    result["peak_sampled_worker_rss_bytes"] = max((s["process_rss_bytes"] or 0 for s in samples), default=0)
    result["minimum_available_bytes"] = min((s["available_bytes"] for s in samples), default=None)
    result["minimum_pressure_free_percent"] = min((s["memory_pressure_free_percent"] for s in samples if s["memory_pressure_free_percent"] is not None), default=None)
    result["minimum_disk_free_bytes"] = min((s["disk_free_bytes"] for s in samples), default=None)
    result["maximum_swap_used_growth_bytes"] = max((s["swap_used_bytes"] - readiness["resources"]["swap_used_bytes"] for s in samples), default=None)
    result["maximum_swap_out_growth_bytes"] = max((s["swap_out_total_bytes"] - readiness["resources"]["swap_out_total_bytes"] for s in samples), default=None)
    worker_file = run_dir / "worker_result.json"
    if worker_file.exists():
        result["worker_result"] = json.loads(worker_file.read_text())
    completed = (result["exit_code"] == 0 and "stop_reasons" not in result and
                 result.get("worker_result", {}).get("status") == "completed" and
                 1 <= len(steps) <= experiment["max_optimizer_steps"] and
                 result["worker_result"]["completed_steps"] == len(steps) and
                 result["worker_result"]["restore_verification"]["generated_token_ids_identical"])
    result["status"] = "completed" if completed else "stopped_by_guard" if "stop_reasons" in result else "worker_failed"
    (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.execute and args.run_dir is None:
        parser.error("--execute requires --run-dir")
    output = run_overfit(args.run_dir) if args.execute else overfit_preflight()
    print(json.dumps({key: value for key, value in output.items() if key != "snapshots"}, indent=2))
