"""Download only the pinned 4-bit Qwen3-0.6B MLX inference artifact."""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "mlx-community/Qwen3-0.6B-4bit"
REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
MODEL_SHA256 = "392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2"
FILES = ("added_tokens.json", "config.json", "model.safetensors.index.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "model.safetensors")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    directory = ROOT / ".cache" / "qwen3-0.6b-4bit"
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = directory / filename
        if path.exists() and (filename != "model.safetensors" or sha256(path) == MODEL_SHA256):
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}"
        temp = path.with_suffix(path.suffix + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
            if filename == "model.safetensors" and sha256(temp) != MODEL_SHA256:
                raise ValueError("pinned model checksum mismatch")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    print("Verified pinned model:", directory)


if __name__ == "__main__":
    download()
