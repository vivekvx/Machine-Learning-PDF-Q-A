"""Focused checks before freezing the second support-checker plan."""
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

SCRATCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRATCH))
from evidence_checker.v2.prepare import build_split, config, verify  # noqa: E402
from evidence_checker.v2.model import FEATURE_WIDTH, SupportCalibrator, candidate_features, loss, scenario_features  # noqa: E402
from evidence_checker.v2.train import _balanced_order, class_metrics, rows, select_threshold  # noqa: E402
from pointer_qa.answerability.train import selected_pointer  # noqa: E402
from run import guard_reasons  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402


class CheckerV2Tests(unittest.TestCase):
    def test_document_isolation_and_unique_candidate_records(self):
        stats = verify()
        self.assertEqual(stats["train"]["records"], 7200)
        self.assertEqual(stats["validation"]["records"], 1440)
        train, validation = rows("train"), rows("validation")
        self.assertFalse({r["document_id"] for r in train} & {r["document_id"] for r in validation})
        self.assertEqual(len({r["id"] for r in train + validation}), len(train) + len(validation))
        self.assertFalse({(r["passage"], r["question"], r["candidate_value"]) for r in train} &
                         {(r["passage"], r["question"], r["candidate_value"]) for r in validation})
        with self.assertRaises(ValueError):
            rows("test")

    def test_hard_negative_labels_and_offsets(self):
        for split in ("train", "validation"):
            all_rows = rows(split)
            counts = Counter(r["category"] for r in all_rows)
            self.assertEqual(len(set(counts.values())), 1)
            for row in all_rows:
                source = enumerate_candidates(row["passage"])[row["candidate_index"]]
                self.assertEqual(source["value"], row["candidate_value"])
                self.assertEqual(row["passage"][row["candidate_start"]:row["candidate_end"]], row["candidate_value"])
                self.assertEqual(row["label_supported"], row["candidate_index"] in row["supported_indices"])
            for sid, items in group_scenarios(all_rows).items():
                self.assertEqual(len({r["passage"] for r in items}), 1)
                self.assertEqual(len({r["question"] for r in items}), 1)
                self.assertEqual(len({tuple(r["supported_indices"]) for r in items}), 1)

    def test_balanced_fixed_order(self):
        train = rows("train")
        order = _balanced_order(train, 1200, config()["seed"])
        self.assertEqual(order, _balanced_order(train, 1200, config()["seed"]))
        by_id = {r["id"]: r for r in train}
        counts = Counter(by_id[id]["category"] for id in order)
        self.assertEqual(counts["supported_original"], 300)
        self.assertEqual(counts["supported_current_value"], 300)
        self.assertTrue(all(counts[c] == 150 for c in (
            "wrong_entity_same_field", "wrong_field_same_entity",
            "obsolete_conflicting_result", "required_fact_absent")))

    def test_candidate_scores_are_question_specific_and_no_position_feature(self):
        pointer, tok = selected_pointer()
        rows_same_doc = [r for r in rows("validation") if r["document_id"] == rows("validation")[0]["document_id"]]
        first = next(r for r in rows_same_doc if r["scenario_kind"] == "original")
        other = next(r for r in rows_same_doc if r["scenario_kind"] == "original" and r["question"] != first["question"])
        a, _, _, _ = scenario_features(pointer, tok, first["passage"], first["question"], 128)
        b, _, _, _ = scenario_features(pointer, tok, other["passage"], other["question"], 128)
        self.assertEqual(len(a[0]), FEATURE_WIDTH)
        self.assertTrue(all(math.isfinite(value) for group in a for value in group))
        self.assertGreater(max(abs(x-y) for x,y in zip(a[0],b[0])), 1e-5)
        self.assertEqual(candidate_features([1,3,2],1)[2],1)

    def test_head_has_gradients_and_updates(self):
        mx.random.seed(20261015)
        model = SupportCalibrator()
        objective = nn.value_and_grad(model, loss)
        features = mx.array([.1,.2,.3,.4,.5,.6,.7,.8])
        value, gradient = objective(model, features, True)
        mx.eval(value, gradient)
        flat = dict(tree_flatten(gradient))
        self.assertTrue(math.isfinite(float(value.item())))
        self.assertGreater(float(mx.sum(mx.abs(flat["output.weight"])).item()), 0)
        before = model.output.weight + mx.zeros_like(model.output.weight)
        mx.eval(before)
        optimizer = optim.AdamW(learning_rate=.01)
        optimizer.update(model, gradient)
        mx.eval(model.parameters())
        self.assertTrue(bool(mx.any(before != model.output.weight).item()))

    def test_threshold_accounting_and_resource_guard(self):
        records = [{"label_supported": i < 2, "probability": p, "category": "synthetic"}
                   for i,p in enumerate((.9,.8,.2,.1))]
        threshold = select_threshold(records, .01)
        result = class_metrics(records, threshold)
        self.assertEqual(result["confusion"], {"TP":2,"FN":0,"TN":2,"FP":0})
        base = {"available_bytes":2_000_000_000,"swap_used_bytes":0,"swap_out_total_bytes":0,
                "process_rss_bytes":100_000_000,"memory_pressure_free_percent":40,
                "disk_free_bytes":3_000_000_000}
        safety = {"stop_available_bytes":700_000_000,"stop_swap_free_bytes":400_000_000,
                  "stop_swap_growth_bytes":600_000_000,"stop_process_rss_bytes":2_500_000_000,
                  "stop_pressure_free_percent":10,"min_preflight_disk_free_bytes":1_000_000_000}
        self.assertTrue(guard_reasons({**base,"available_bytes":0},base,safety))


def group_scenarios(rows):
    groups={}
    for row in rows:
        groups.setdefault(row["scenario_id"],[]).append(row)
    return groups


if __name__ == "__main__":
    unittest.main()
