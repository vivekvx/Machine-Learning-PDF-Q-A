"""Install or audit the isolated local PDF QA runtimes; never train a model.

An install downloads Qwen only with --download-qwen. The scratch pointer is a
small project-trained checkpoint distributed with the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
SLM = SCRATCH.parent
ROOT = SLM.parent.parent
CONFIG = json.loads((HERE / "hybrid_config.v14.json").read_text())
POINTER_SELECTION = SCRATCH / "pointer_qa/v2/validation/pointer_selection.json"
POINTER_WEIGHT = SCRATCH / "pointer_qa/v2/runs/main/checkpoint-step-2000/model.safetensors"
POINTER_TOKENIZER = SCRATCH / "pointer_qa/v1/tokenizer.json"
MODEL = SLM / ".cache" / CONFIG["local_dir"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_status(path: Path, expected: str) -> str:
    if not path.is_file():
        return "missing"
    return "verified" if sha256(path) == expected else "checksum_mismatch"


def environment_status(python: Path, requirements: dict[str, str]) -> str:
    if not python.is_file():
        return "missing"
    script = ("import importlib.metadata as m,json; "
              "print(json.dumps({x:m.version(x) for x in " + repr(tuple(requirements)) + "}))")
    try:
        result = subprocess.run([str(python), "-c", script], capture_output=True,
                                text=True, timeout=20, check=False)
        found = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "package_check_failed"
    return "verified" if found == requirements else "version_mismatch"


def audit() -> dict:
    if not POINTER_SELECTION.is_file():
        pointer = {"selection_ledger": "missing"}
    else:
        ledger = json.loads(POINTER_SELECTION.read_text())
        pointer = {
            "checkpoint": artifact_status(POINTER_WEIGHT, ledger["checkpoint_sha256"]),
            "tokenizer": artifact_status(POINTER_TOKENIZER, ledger["tokenizer_sha256"]),
        }
    qwen = {
        "weights": artifact_status(MODEL / "model.safetensors", CONFIG["model_sha256"]),
        "tokenizer_config": artifact_status(MODEL / "tokenizer_config.json", CONFIG["tokenizer_config_sha256"]),
        "license": artifact_status(MODEL / "LICENSE", CONFIG["license_sha256"]),
    }
    environments = {
        "service": environment_status(HERE / ".venv/bin/python", {
            "fastapi": "0.139.0", "faiss-cpu": "1.12.0", "pymupdf": "1.26.4"}),
        "scratch_pointer": environment_status(SCRATCH / ".venv/bin/python", {"mlx": "0.32.3"}),
        "qwen": environment_status(SLM / ".venv/bin/python", {
            "mlx": CONFIG["mlx"], "mlx-lm": CONFIG["mlx_lm"]}),
    }
    platform_ok = sys.platform == "darwin" and platform.machine() == "arm64"
    return {"platform": "verified" if platform_ok else "requires_macos_apple_silicon",
            "pointer": pointer, "qwen": qwen, "environments": environments,
            "pointer_ready": platform_ok and all(x == "verified" for x in pointer.values())
                             and environments["scratch_pointer"] == "verified"
                             and environments["service"] == "verified",
            "qwen_ready": platform_ok and all(x == "verified" for x in qwen.values())
                          and environments["qwen"] == "verified"
                          and environments["service"] == "verified"}


def install(download_qwen: bool) -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX requires Apple Silicon macOS")
    if shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise RuntimeError("At least 2 GB of free disk space is required before installation")
    uv = shutil.which("uv")
    python = shutil.which("python3.14")
    if not uv or not python:
        raise RuntimeError("Install uv and Python 3.14 first; no automatic system-wide install is attempted")
    for location, requirements in (
        (HERE / ".venv", HERE / "requirements.lock"),
        (SCRATCH / ".venv", SCRATCH / "requirements.txt"),
        (SLM / ".venv", SLM / "requirements-baseline.lock.txt"),
    ):
        if not (location / "bin/python").is_file():
            subprocess.run([uv, "venv", "--python", python, str(location)], check=True, cwd=ROOT)
        subprocess.run([uv, "pip", "install", "--python", str(location / "bin/python"),
                        "-r", str(requirements)], check=True, cwd=ROOT)
    if download_qwen:
        script = ("from huggingface_hub import snapshot_download; "
                  "snapshot_download(repo_id=" + repr(CONFIG["model_id"]) + ", revision="
                  + repr(CONFIG["revision"]) + ", local_dir=" + repr(str(MODEL)) + ")")
        subprocess.run([str(SLM / ".venv/bin/python"), "-c", script], check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="create three pinned, isolated environments")
    parser.add_argument("--download-qwen", action="store_true", help="explicitly download the pinned 4-bit model during install")
    args = parser.parse_args()
    if args.download_qwen and not args.install:
        parser.error("--download-qwen requires --install")
    if args.install:
        install(args.download_qwen)
    status = audit()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["pointer_ready"] and status["qwen_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
