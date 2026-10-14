"""Focused split, label, offset, head-gradient and guard checks."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

SCRATCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRATCH))
from pointer_qa.answerability.prepare import V1, corrected_passage, scenario_records, source_rows, verify  # noqa: E402
from pointer_qa.answerability.model import FEATURE_WIDTH, SufficiencyHead, binary_loss, frozen_features  # noqa: E402
from pointer_qa.answerability.train import _threshold, metrics, rows, selected_pointer  # noqa: E402
from run import guard_reasons, preflight  # noqa: E402
from working_qa.prepare import enumerate_candidates  # noqa: E402


class SufficiencyTests(unittest.TestCase):
    def test_manifest_and_split_isolation(self):
        stats = verify()
        self.assertEqual(stats["train"]["records"], 2400)
        self.assertEqual(stats["validation"]["records"], 480)
        self.assertFalse({r["document_id"] for r in rows("train")} &
                         {r["document_id"] for r in rows("validation")})
        with self.assertRaises(ValueError):
            rows("test")
        with self.assertRaises(ValueError):
            source_rows("test")

    def test_balanced_categories_and_exact_source_offsets(self):
        for split in ("train", "validation"):
            sample = rows(split)
            counts = Counter((r["category"], r["label_supported"]) for r in sample)
            for category in ("wrong_entity", "wrong_field", "missing_fact", "contradiction"):
                self.assertEqual(counts[(category, True)], counts[(category, False)])
            for row in sample:
                self.assertEqual(row["candidate_value"],
                                 row["passage"][row["candidate_start"]:row["candidate_end"]])
                self.assertEqual(row["label_supported"],
                                 row["proposed_candidate_index"] in row["supported_candidate_indices"])

    def test_correction_is_not_a_fixed_position_shortcut(self):
        originals = [r for r in source_rows("train") if r["gold_candidate_index"] == 3]
        records = [x for row in originals for x in scenario_records(row, 20261014)]
        positions = Counter(r["proposed_candidate_index"] for r in records if r["label_supported"])
        self.assertGreater(len(positions), 2)

    def test_full_context_features_and_question_change(self):
        pointer, tok = selected_pointer()
        first, second = source_rows("validation")[:2]
        self.assertEqual(first["document_id"], second["document_id"])
        a, _ = frozen_features(pointer, tok, first["passage"], first["question"], 0, 240)
        b, _ = frozen_features(pointer, tok, first["passage"], second["question"], 0, 240)
        self.assertEqual(len(a), FEATURE_WIDTH)
        self.assertGreater(max(abs(x-y) for x, y in zip(a, b)), 1e-5)
        corrected = next(r for r in rows("validation") if r["category"] == "contradiction")
        feature, counts = frozen_features(pointer, tok, corrected["passage"], corrected["question"],
                                          corrected["proposed_candidate_index"], 240)
        self.assertEqual(len(feature), FEATURE_WIDTH)
        self.assertLessEqual(counts["full_input_tokens"], 240)

    def test_head_gradients_and_updates(self):
        mx.random.seed(20261014)
        head = SufficiencyHead(64)
        objective = nn.value_and_grad(head, binary_loss)
        feature = mx.random.normal((FEATURE_WIDTH,))
        before = dict(tree_flatten(head.parameters()))
        value, gradient = objective(head, feature, True)
        mx.eval(value, gradient)
        self.assertGreater(float(mx.linalg.norm(gradient["first"]["weight"]).item()), 0)
        import mlx.optimizers as optim
        optimizer = optim.AdamW(learning_rate=.001)
        optimizer.update(head, gradient)
        mx.eval(head.parameters())
        after = dict(tree_flatten(head.parameters()))
        self.assertTrue(any(bool(mx.any(before[k] != after[k]).item()) for k in before))

    def test_threshold_metrics_and_guard(self):
        records = [{"id": str(i), "category": "missing_fact", "label_supported": i < 2,
                    "probability": p} for i, p in enumerate((.8, .7, .2, .1))]
        t = _threshold(records, [.3, .5, .9])
        result = metrics(records, t)
        self.assertEqual(result["supported_accuracy"], 1)
        self.assertEqual(result["unsupported_rejection"], 1)
        pre = preflight("smoke")
        observed = dict(pre["resources"])
        observed["available_bytes"] = 0
        self.assertTrue(guard_reasons(observed, pre["resources"], pre["safety"]))


if __name__ == "__main__":
    unittest.main()
