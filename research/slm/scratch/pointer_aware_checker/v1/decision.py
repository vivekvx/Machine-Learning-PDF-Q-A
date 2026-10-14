"""Final scientific decision; a feature-label shortcut keeps test sealed."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def decide():
    path = HERE / "decision.json"
    if path.exists():
        raise FileExistsError("Decision already frozen")
    worker = json.loads((HERE / "run" / "worker_result.json").read_text())
    comparison = json.loads((HERE / "run" / "comparison.json").read_text())
    audit = json.loads((HERE / "run" / "shortcut_audit.json").read_text())
    gate = worker["final_gate"]["pointer_selected"]
    required = json.loads((HERE / "config.json").read_text())["gate"]
    numeric = (gate["supported_acceptance"] >= required["supported_acceptance_minimum"] and
        gate["unsupported_rejection"] >= required["unsupported_rejection_minimum"] and
        worker["final_gate"]["combined_original_exact"] >= required["original_answerable_exact_minimum"] and
        worker["final_gate"]["obsolete_value_rejection"] >= required["obsolete_value_rejection_minimum"] and
        worker["final_gate"]["correct_answer_retention"] >= required["correct_answer_retention_vs_pointer_minimum"])
    shortcut = audit["simple_boolean_shortcut"]["final_gate"]
    shortcut_reproduces_labels = shortcut["all_candidates_correct"] == shortcut["all_candidates_count"]
    if comparison["new_checker"] != worker["final_gate"]:
        raise ValueError("Comparison and worker gate differ")
    result = {"schema_version": 1, "numeric_validation_gate_passed": numeric,
        "independent_learned_sufficiency_established": False if shortcut_reproduces_labels else None,
        "feature_label_shortcut_detected": shortcut_reproduces_labels,
        "combined_pipeline_approved": numeric and not shortcut_reproduces_labels,
        "test_may_be_opened": numeric and not shortcut_reproduces_labels,
        "reason": "Engineered joint-match and supersession features deterministically recover every generated label; model performance does not establish learned sufficiency outside these templates." if shortcut_reproduces_labels else "No shortcut found by this audit",
        "sealed_test_read": False,
        "pointer_checkpoint_sha256": worker["pointer_checkpoint_sha256"],
        "checker_checkpoint_sha256": worker["checkpoint_sha256"],
        "threshold": worker["threshold"],
        "comparison_file": "run/comparison.json", "audit_file": "run/shortcut_audit.json"}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(decide(), indent=2))
