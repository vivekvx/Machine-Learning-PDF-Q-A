"""Export the immutable overfit curve JSONL to a simple plotting CSV."""
import argparse
import csv
import json
import math
from pathlib import Path


def export(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "curve.jsonl").read_text().splitlines()]
    steps = [row["step"] for row in rows]
    if not rows or steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError("Curve steps must be unique, ordered, and start at zero")
    if any(not math.isfinite(row["full_set_loss"]) for row in rows):
        raise ValueError("Curve contains non-finite loss")
    target = run_dir / "curve.csv"
    with target.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "full_set_loss", "exact_reproductions", "continuations_evaluated"))
        for row in rows:
            writer.writerow((row["step"], row["full_set_loss"], row["exact_reproductions"],
                             row.get("continuations_evaluated", False)))
    return {"rows": len(rows), "path": str(target)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir)))
