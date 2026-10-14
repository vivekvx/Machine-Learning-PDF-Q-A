"""Final read-only combined gate; prevents a gold-only pass opening test."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare import sha

HERE = Path(__file__).resolve().parent


def decision():
    destination = HERE / "combined_gate.json"
    if destination.exists():
        raise FileExistsError("Combined gate decision already frozen")
    run_path = HERE / "run" / "run.json"
    run = json.loads(run_path.read_text())
    if run["status"] != "completed":
        raise RuntimeError("Guarded checker training did not complete")
    worker = run["worker_result"]
    gold_path = HERE / "run" / "gate_gold_predictions.jsonl"
    selected_path = HERE / "run" / "gate_pointer_predictions.jsonl"
    if (sha(gold_path) != worker["gate_gold_predictions_sha256"] or
        sha(selected_path) != worker["gate_pointer_predictions_sha256"]):
        raise ValueError("Gate prediction hashes changed")
    selected = [json.loads(line) for line in selected_path.read_text().splitlines()]
    by_kind = defaultdict(list)
    for row in selected:
        key = "missing_fact" if row["scenario_kind"].startswith("missing-") else row["scenario_kind"]
        by_kind[key].append(row)
    groups = {kind: {"count": len(items),
                     "unsupported": sum(not x["label_supported"] for x in items),
                     "unsupported_false_answers": sum(not x["label_supported"] and x["accepted"] for x in items),
                     "supported_rejected": sum(x["label_supported"] and not x["accepted"] for x in items)}
              for kind, items in sorted(by_kind.items())}
    requirements = json.loads((HERE / "config.json").read_text())["final_validation_gate"]
    gold = worker["final_gate_gold"]
    predicted = worker["final_gate_pointer_selected"]["predicted_support_classification"]
    gold_pass = (gold["supported_acceptance"] >= requirements["supported_acceptance"] and
                 gold["unsupported_rejection"] >= requirements["unsupported_rejection"])
    predicted_pass = (predicted["supported_acceptance"] >= requirements["supported_acceptance"] and
                      predicted["unsupported_rejection"] >= requirements["unsupported_rejection"])
    outcome = {"schema_version": 1, "combined_pipeline_approved": gold_pass and predicted_pass,
               "gold_candidate_gate_passed": gold_pass, "pointer_selected_gate_passed": predicted_pass,
               "test_may_be_opened": gold_pass and predicted_pass,
               "frozen_gold_selection_file_is_not_combined_approval": True,
               "threshold": worker["threshold"],
               "gold_candidate": {"supported_acceptance": gold["supported_acceptance"],
                                  "unsupported_rejection": gold["unsupported_rejection"],
                                  "confusion": gold["confusion"]},
               "pointer_selected": {"supported_acceptance": predicted["supported_acceptance"],
                                    "unsupported_rejection": predicted["unsupported_rejection"],
                                    "confusion": predicted["confusion"]},
               "end_to_end": {k: v for k, v in worker["final_gate_pointer_selected"].items()
                              if k != "predicted_support_classification"},
               "pointer_selected_error_groups": groups,
               "run_sha256": sha(run_path), "gold_predictions_sha256": sha(gold_path),
               "pointer_predictions_sha256": sha(selected_path),
               "sealed_test_read": False}
    destination.write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


if __name__ == "__main__":
    print(json.dumps(decision(), indent=2))
