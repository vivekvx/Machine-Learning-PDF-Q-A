"""Read-only threshold audit of the frozen v1 checker predictions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
V1 = SCRATCH / "pointer_qa" / "answerability" / "v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc(records):
    positive = [r["probability"] for r in records if r["label_supported"]]
    negative = [r["probability"] for r in records if not r["label_supported"]]
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def at_threshold(records, threshold):
    positive = [r for r in records if r["label_supported"]]
    negative = [r for r in records if not r["label_supported"]]
    tp = sum(r["probability"] >= threshold for r in positive)
    tn = sum(r["probability"] < threshold for r in negative)
    return {"threshold": threshold, "supported_acceptance": tp / len(positive),
            "unsupported_rejection": tn / len(negative),
            "balanced_accuracy": .5 * (tp / len(positive) + tn / len(negative)),
            "confusion": {"TP": tp, "FN": len(positive) - tp,
                          "TN": tn, "FP": len(negative) - tn}}


def sweep(records):
    unique = sorted({r["probability"] for r in records})
    thresholds = [0.0] + [(a + b) / 2 for a, b in zip(unique[:-1], unique[1:])] + [1.0]
    all_results = [at_threshold(records, t) for t in thresholds]
    feasible = [r for r in all_results if r["supported_acceptance"] >= .70 and
                r["unsupported_rejection"] >= .70]
    best = max(all_results, key=lambda r: (min(r["supported_acceptance"], r["unsupported_rejection"]),
                                           r["balanced_accuracy"], -abs(r["threshold"] - .5)))
    return {"count": len(records), "auroc": auc(records), "frozen_gate_feasible": bool(feasible),
            "feasible_threshold_count": len(feasible), "best_minimum_class_accuracy": best,
            "registered_threshold": at_threshold(records, .70),
            "selected_grid": [at_threshold(records, t) for t in (.30, .40, .50, .60, .70, .80)],
            "category_auroc": {category: auc([r for r in records if r["category"] == category])
                               for category in sorted({r["category"] for r in records})}}


def audit():
    output = HERE / "v1_threshold_audit.json"
    if output.exists():
        raise FileExistsError("Frozen v1 threshold audit already exists")
    predictions = V1 / "run" / "validation_predictions.jsonl"
    plan_path = V1 / "plan.json"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    plan = json.loads(plan_path.read_text())
    calibration = set(plan["calibration_document_ids"])
    assessment = set(plan["assessment_document_ids"])
    if calibration & assessment or {r["document_id"] for r in rows} != calibration | assessment:
        raise ValueError("Validation document partition changed")
    report = {"schema_version": 1, "source_predictions_sha256": sha(predictions),
              "source_plan_sha256": sha(plan_path), "trained_weights_unchanged": True,
              "calibration": sweep([r for r in rows if r["document_id"] in calibration]),
              "assessment": sweep([r for r in rows if r["document_id"] in assessment]),
              "all_validation": sweep(rows)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return {name: {"auroc": report[name]["auroc"],
                   "frozen_gate_feasible": report[name]["frozen_gate_feasible"],
                   "best_minimum_class_accuracy": report[name]["best_minimum_class_accuracy"]}
            for name in ("calibration", "assessment", "all_validation")}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
