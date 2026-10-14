"""Read-only audit of already-frozen v2 checker validation predictions."""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parents[1]
SOURCE = SCRATCH / "evidence_checker" / "v2"


def auc(rows):
    positive = [r["probability"] for r in rows if r["label_supported"]]
    negative = [r["probability"] for r in rows if not r["label_supported"]]
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive)*len(negative))


def metrics(rows, threshold):
    pos = [r for r in rows if r["label_supported"]]
    neg = [r for r in rows if not r["label_supported"]]
    tp = sum(r["probability"] >= threshold for r in pos)
    tn = sum(r["probability"] < threshold for r in neg)
    return {"threshold": threshold,
        "supported_acceptance": tp/len(pos), "unsupported_rejection": tn/len(neg),
        "balanced_accuracy": (tp/len(pos)+tn/len(neg))/2,
        "confusion": {"TP": tp, "FN": len(pos)-tp, "TN": tn, "FP": len(neg)-tn}}


def build():
    from hashlib import sha256
    predictions_path = SOURCE / "run" / "gate_pointer_predictions.jsonl"
    labels_path = SOURCE / "validation.jsonl"
    frozen = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    by_key = {(r["scenario_id"], r["candidate_index"]): r["category"] for r in labels}
    groups = defaultdict(list)
    for r in frozen:
        if r["scenario_kind"] == "original":
            category = by_key.get((r["scenario_id"], r["candidate_index"]),
                                  "other_pointer_error")
        elif r["scenario_kind"] == "correction":
            category = "current_value" if r["label_supported"] else "obsolete_value"
        else:
            category = "missing_fact"
        groups[category].append(r["probability"])
    distributions = {name: {"count": len(values), "min": min(values),
        "q25": sorted(values)[len(values)//4], "median": statistics.median(values),
        "q75": sorted(values)[3*len(values)//4], "max": max(values)}
        for name, values in sorted(groups.items())}
    sweep = [metrics(frozen, i/100) for i in range(1,100)]
    best = max(sweep, key=lambda x: (min(x["supported_acceptance"], x["unsupported_rejection"]),
                                     x["balanced_accuracy"], -abs(x["threshold"]-.5)))
    original = [r for r in frozen if r["scenario_kind"] == "original"]
    obsolete = [r for r in frozen if r["scenario_kind"] == "correction" and not r["label_supported"]]
    for item in sweep:
        t = item["threshold"]
        item["original_exact_count"] = sum(r["label_supported"] and r["probability"] >= t for r in original)
        item["obsolete_rejected_count"] = sum(r["probability"] < t for r in obsolete)
    result = {"input_prediction_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
        "input_label_sha256": sha256(labels_path.read_bytes()).hexdigest(),
        "read_only": True, "previously_evaluated_gate_documents": True,
        "selected_candidate_count": len(frozen), "auroc": auc(frozen),
        "distributions": distributions, "frozen_threshold_0_68": metrics(frozen,.68),
        "example_threshold_0_80": next(x for x in sweep if x["threshold"] == .8),
        "best_maximin": best, "sweep": sweep}
    out = HERE / "readonly_diagnosis.json"
    if out.exists():
        raise FileExistsError("Read-only diagnosis already frozen")
    out.write_text(json.dumps(result, indent=2) + "\n")
    return {k: value for k, value in result.items() if k != "sweep"}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
