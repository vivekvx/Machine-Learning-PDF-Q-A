import json
from collections import Counter
import re
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from experiment import load_plan, masked_loss, rows, tokenizer
from evaluate import rule_prediction
from prepare import V1, VERSION, answer_sequence, prompt, verify


class SyntheticQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify()
        cls.train = rows("train")
        cls.validation = rows("validation")
        cls.tok = tokenizer()

    def test_families_facts_and_entities_disjoint(self):
        manifest = json.loads((V1 / "manifest.json").read_text())
        self.assertFalse(set(manifest["split_families"]["train"]) & set(manifest["split_families"]["validation"]))
        self.assertFalse(set(manifest["split_families"]["test"]) & set(manifest["split_families"]["train"] + manifest["split_families"]["validation"]))
        self.assertTrue({r["target_entity"] for r in self.train}.isdisjoint(r["target_entity"] for r in self.validation))
        self.assertTrue({r["fact_id"] for r in self.train}.isdisjoint(r["fact_id"] for r in self.validation))
        self.assertEqual(len({(r["passage"], r["question"]) for r in self.train + self.validation}), len(self.train + self.validation))

    def test_answerability_and_distractor(self):
        for subset in (self.train, self.validation):
            self.assertEqual(sum(r["answerable"] for r in subset), len(subset) // 2)
            if VERSION == "v2":
                counts = Counter((r["asked_field"], r["answerable"]) for r in subset)
                self.assertEqual(len(set(counts.values())), 1)
            for row in subset:
                self.assertIn(row["asked_field"], row["distractor_sentence"])
                if row["answerable"]:
                    self.assertIn(row["expected_answer"], row["target_sentence"])
                    self.assertEqual(row["evidence_sentence"], row["target_sentence"])
                else:
                    self.assertEqual(row["expected_answer"], "UNANSWERABLE")
                    self.assertNotIn(row["asked_field"], row["target_sentence"])
                    self.assertIsNone(row["evidence_sentence"])

    def test_rule_uses_exact_target_entity(self):
        # The distractor deliberately contains the target ID as a suffix.
        self.assertTrue(all(rule_prediction(r["passage"], r["question"]) == r["expected_answer"]
                            for r in self.validation))

    def test_v3_has_no_visible_entity_parity_shortcut(self):
        if VERSION != "v3":
            self.skipTest("v3-only correction")
        for subset in (self.train, self.validation):
            self.assertTrue(all(re.fullmatch(r"(?:TM|TD|TO|VB|XF)-[A-F0-9]{8}", r["target_entity"])
                                for r in subset))
            for field in ("timeout", "port", "mode", "owner"):
                self.assertEqual(sum(r["answerable"] for r in subset if r["asked_field"] == field),
                                 sum(not r["answerable"] for r in subset if r["asked_field"] == field))

    def test_tokenizer_and_mask(self):
        for row in self.train[:12] + self.validation[:12]:
            ids, mask, prefix_len, answer_len = answer_sequence(self.tok, row)
            self.assertEqual(self.tok.decode(ids[:-1]), prompt(row["passage"], row["question"]) + row["expected_answer"])
            self.assertEqual(mask, [0] * (prefix_len - 1) + [1] * (answer_len + 1))
            self.assertLessEqual(len(ids), 192)

    def test_plan_is_fixed_and_excludes_test(self):
        plan = load_plan()
        self.assertEqual(len(plan["sample_order"]), 800)
        self.assertEqual(len(set(plan["sample_order"])), 800)
        self.assertEqual(len(plan["fixed_train_ids"]), 64)
        self.assertEqual(len(plan["fixed_validation_ids"]), 80)
        self.assertTrue(all(i.startswith("train-") for i in plan["sample_order"] + plan["fixed_train_ids"]))
        self.assertTrue(all(i.startswith("validation-") for i in plan["fixed_validation_ids"]))

    def test_loss_mask_ignores_prompt_targets(self):
        class Toy(nn.Module):
            def __call__(self, x):
                return mx.zeros((1, x.shape[1], 8))
        model = Toy()
        x = mx.array([[1, 2, 3]], dtype=mx.int32)
        y1 = mx.array([[1, 2, 3]], dtype=mx.int32)
        y2 = mx.array([[7, 6, 3]], dtype=mx.int32)
        mask = mx.array([[0, 0, 1]], dtype=mx.float32)
        self.assertAlmostEqual(float(masked_loss(model, x, y1, mask).item()), float(masked_loss(model, x, y2, mask).item()))


if __name__ == "__main__":
    unittest.main()
