"""Versioned correction for entity-boundary bug in the first rule baseline.

The correction was established on validation examples after the model run. It
never changes checkpoint selection or the original evaluation artifacts.
"""
import argparse
import json
import time
from pathlib import Path

from evaluate import assess, rule_prediction, score
from prepare import V1, digest, verify


def correct(run_dir):
    run_dir = Path(run_dir).resolve()
    verify()
    if not (run_dir / "evaluation.json").exists():
        raise ValueError("Original evaluation must exist")
    if (run_dir / "rule_correction_v2.json").exists():
        raise FileExistsError("Correction already exists")
    scores = {}
    outputs = {}
    for split in ("validation", "test"):
        source = V1 / ("test.sealed.jsonl" if split == "test" else "validation.jsonl")
        records = []
        for line in source.read_text().splitlines():
            row = json.loads(line)
            tick = time.monotonic()
            prediction = rule_prediction(row["passage"], row["question"])
            records.append(assess(row, prediction, time.monotonic() - tick, 0, False))
        output = run_dir / f"{split}_rule_v2_predictions.jsonl"
        with output.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        scores[split] = score(records)
        outputs[output.name] = digest(output)
    report = {"reason": "Original rule baseline matched target entity as a suffix of distractor entity; fixed on validation IDs only.",
              "model_predictions_changed": False, "original_evaluation_sha256": digest(run_dir / "evaluation.json"),
              "rule_source_sha256": digest(Path(__file__).with_name("evaluate.py")),
              "scores": scores, "prediction_file_sha256": outputs}
    (run_dir / "rule_correction_v2.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(correct(args.run_dir), indent=2))
