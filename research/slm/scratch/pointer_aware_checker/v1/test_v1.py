import json
import sys
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SCRATCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRATCH))
from pointer_aware_checker.v1.prepare import verify  # noqa: E402
from pointer_aware_checker.v1.model import ContextChecker, all_features, loss  # noqa: E402
from pointer_aware_checker.v1.train import training_order, class_metrics, choose_threshold  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402

HERE = Path(__file__).resolve().parent


class CandidateAwareTests(unittest.TestCase):
    def test_split_integrity_and_source_hashes(self):
        verify()
        partitions = {}
        for name in ("train", "calibration", "selection", "final_gate"):
            rows = [json.loads(x) for x in (HERE / f"{name}.jsonl").read_text().splitlines()]
            partitions[name] = {r["document_id"] for r in rows}
            self.assertEqual(len(rows), len({r["scenario_id"] for r in rows}))
            for row in rows:
                candidates = enumerate_candidates(row["passage"])
                self.assertEqual(len(row["supported_indices"]), 0 if row["condition"] == "missing" else 1)
                self.assertTrue(all(0 <= i < len(candidates) for i in row["supported_indices"]))
        for first in partitions:
            for second in partitions:
                if first < second:
                    self.assertFalse(partitions[first] & partitions[second])

    def test_context_changes_support_for_same_candidate(self):
        rows = [json.loads(x) for x in (HERE / "train.jsonl").read_text().splitlines()]
        original = next(r for r in rows if r["condition"] == "original")
        correction = next(r for r in rows if r["source_question_id"] == original["source_question_id"] and r["condition"] == "correction")
        old_value = enumerate_candidates(original["passage"])[original["supported_indices"][0]]["value"]
        oi = next(i for i, x in enumerate(enumerate_candidates(correction["passage"])) if x["value"] == old_value)
        old_original_features = all_features(original["passage"], original["question"], [0., 1., 2., 3.])
        old_corrected_features = all_features(correction["passage"], correction["question"], [0., 1., 2., 3., 4.])
        self.assertEqual(old_original_features[original["supported_indices"][0]][-1], 0.)
        self.assertEqual(old_corrected_features[oi][-1], 1.)
        self.assertEqual(old_original_features[original["supported_indices"][0]][-3], 1.)

    def test_unrelated_correction_is_not_supersession(self):
        rows = [json.loads(x) for x in (HERE / "train.jsonl").read_text().splitlines()]
        item = next(r for r in rows if r["condition"] == "unrelated_correction")
        features = all_features(item["passage"], item["question"], [float(i) for i in range(5)])
        self.assertEqual(features[item["supported_indices"][0]][-1], 0.)

    def test_balanced_order_and_threshold(self):
        rows = [{"id": str(i), "label_supported": i % 2 == 0, "pointer_selected": i % 3 == 0}
                for i in range(40)]
        order = training_order(rows, 20, 7)
        self.assertEqual(len(order), len(set(order)))
        label = {r["id"]: r["label_supported"] for r in rows}
        self.assertEqual(sum(label[i] for i in order), 10)
        records = [{"label_supported": True, "probability": .9},
                   {"label_supported": False, "probability": .1}]
        self.assertEqual(class_metrics(records, .5)["confusion"], {"TP": 1, "FN": 0, "TN": 1, "FP": 0})
        self.assertTrue(.1 < choose_threshold(records, .01) <= .9)

    def test_checker_has_finite_gradients(self):
        model = ContextChecker()
        objective = nn.value_and_grad(model, loss)
        value, gradient = objective(model, mx.zeros((13,), dtype=mx.float32), True)
        mx.eval(value, gradient)
        self.assertTrue(bool(mx.isfinite(value).item()))

    def test_actual_pointer_training_order_and_restoration(self):
        run = HERE / "run"
        records = [json.loads(x) for x in (run / "train_features.jsonl").read_text().splitlines()]
        saved_order = json.loads((run / "training_order.json").read_text())
        self.assertEqual(saved_order, training_order(records, 800, 20261016))
        by_id = {r["id"]: r for r in records}
        self.assertEqual(sum(by_id[i]["label_supported"] for i in saved_order), 400)
        worker = json.loads((run / "worker_result.json").read_text())
        self.assertTrue(worker["checkpoint_restoration_exact"])
        self.assertEqual(worker["checkpoint_optimizer_step"], worker["selected_step"])

    def test_scientific_gate_keeps_test_sealed(self):
        decision = json.loads((HERE / "decision.json").read_text())
        shortcut = json.loads((HERE / "run" / "shortcut_audit.json").read_text())
        self.assertTrue(decision["numeric_validation_gate_passed"])
        self.assertFalse(decision["combined_pipeline_approved"])
        self.assertFalse(decision["test_may_be_opened"])
        self.assertEqual(shortcut["simple_boolean_shortcut"]["final_gate"]["all_candidates_correct"], 1280)


if __name__ == "__main__":
    unittest.main()
