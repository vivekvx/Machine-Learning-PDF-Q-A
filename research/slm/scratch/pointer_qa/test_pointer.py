"""Answer-only pointer data, segmentation, gradients, and guard invariants."""
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import guard_reasons  # noqa: E402
from pointer_qa.model import PointerReader, pointer_loss, pointer_predict  # noqa: E402
from pointer_qa.prepare import V1, pools, verify  # noqa: E402
from pointer_qa.train import config, encoded_row, rows, tokenizer  # noqa: E402
from pointer_qa.v2.model import AlignedPointer, loss as aligned_loss  # noqa: E402


class PointerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = rows("train")
        cls.validation = rows("validation")
        cls.tok = tokenizer()

    def test_manifest_splits_and_answer_positions(self):
        verify()
        c = config()
        self.assertEqual((len(self.train), len(self.validation)), (1200, 240))
        self.assertTrue(all(r["answerable"] and r["expected_answer"] != "UNANSWERABLE"
                            for r in self.train + self.validation))
        self.assertFalse({r["document_id"] for r in self.train} & {r["document_id"] for r in self.validation})
        self.assertFalse({r["family"] for r in self.train} & {r["family"] for r in self.validation})
        names = pools(c["seed"])
        self.assertFalse(set(names["train"]) & set(names["validation"]))
        self.assertFalse(set(names["train"]) & set(names["test"]))
        for subset in (self.train, self.validation):
            counts = Counter(r["gold_candidate_index"] for r in subset)
            self.assertEqual(len(set(counts.values())), 1)
            self.assertEqual(len({(r["passage"], r["question"]) for r in subset}), len(subset))

    def test_each_passage_has_four_distinct_question_targets(self):
        for subset in (self.train, self.validation):
            grouped = {}
            for row in subset:
                grouped.setdefault(row["document_id"], []).append(row)
            for group in grouped.values():
                self.assertEqual(len(group), 4)
                self.assertEqual({r["gold_candidate_index"] for r in group}, {0, 1, 2, 3})
                self.assertEqual(len({r["passage"] for r in group}), 1)
                self.assertEqual(len({r["question"] for r in group}), 4)

    def test_train_only_tokenizer_offsets_and_maximum_span(self):
        information = json.loads((V1 / "tokenizer.json").read_text())
        self.assertEqual(set(information["training_document_ids"]),
                         {r["document_id"] for r in self.train})
        for row in self.train[:80] + self.validation[:80]:
            pairs, candidates = encoded_row(self.tok, row)
            self.assertEqual(len(pairs), 4)
            self.assertEqual(row["passage"][row["answer_start"]:row["answer_end"]], row["expected_answer"])
            for pair, candidate in zip(pairs, candidates):
                self.assertEqual(self.tok.decode(pair["ids"][pair["value_start"]:pair["value_end"]]),
                                 candidate["value"])
                self.assertLessEqual(pair["value_end"] - pair["value_start"],
                                     config()["training"]["maximum_answer_tokens"])
        self.assertEqual(self.tok.decode(self.tok.encode("new Ω bytes")), "new Ω bytes")


class PointerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()
        cls.examples = rows("train")[:4]

    def test_question_changes_neural_scores_for_same_passage(self):
        mx.random.seed(107)
        model = PointerReader(config()["model"], self.tok.vocab_size)
        first = encoded_row(self.tok, self.examples[0])[0]
        second = encoded_row(self.tok, self.examples[1])[0]
        a, b = model(first), model(second)
        mx.eval(a, b)
        self.assertGreater(float(mx.max(mx.abs(a - b)).item()), 1e-5)

    def test_pointer_mask_and_span_copy(self):
        model = PointerReader(config()["model"], self.tok.vocab_size)
        row = self.examples[0]
        pairs, candidates = encoded_row(self.tok, row)
        with self.assertRaises(ValueError):
            pointer_loss(model, pairs, -1)
        with self.assertRaises(ValueError):
            pointer_loss(model, pairs, 4)
        result = pointer_predict(model, pairs, candidates)
        self.assertEqual(row["passage"][result["evidence"]["start"]:result["evidence"]["end"]],
                         result["prediction"])

    def test_finite_gradients_and_update(self):
        mx.random.seed(108)
        model = PointerReader(config()["model"], self.tok.vocab_size)
        row = self.examples[0]
        pairs, _ = encoded_row(self.tok, row)
        objective = nn.value_and_grad(model, pointer_loss)
        value, gradients = objective(model, pairs, row["gold_candidate_index"])
        mx.eval(value, gradients)
        norm = sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(gradients))
        self.assertTrue(math.isfinite(float(value.item())) and math.isfinite(norm) and norm > 0)
        prior = model.out.weight + mx.zeros_like(model.out.weight)
        mx.eval(prior)
        optimizer = optim.AdamW(learning_rate=config()["training"]["learning_rate"])
        optimizer.update(model, gradients)
        mx.eval(model.parameters())
        self.assertGreater(float(mx.max(mx.abs(prior - model.out.weight)).item()), 0)

    def test_unchanged_resource_guard_trips(self):
        safety = json.loads((HERE.parent / "config.v1.json").read_text())["safety"]
        base = {"available_bytes": 2_000_000_000, "swap_used_bytes": 0,
                "swap_out_total_bytes": 0, "process_rss_bytes": 100_000_000,
                "memory_pressure_free_percent": 40, "disk_free_bytes": 3_000_000_000}
        self.assertEqual(guard_reasons(base, base, safety), [])
        pressure = {**base, "memory_pressure_free_percent": 9}
        self.assertTrue(guard_reasons(pressure, base, safety))
        swap = {**base, "swap_used_bytes": 601_000_000}
        self.assertIn("swap growth above stop ceiling", guard_reasons(swap, base, safety))

    def test_corrected_alignment_sends_gradients_to_encoder_and_importance(self):
        settings = json.loads((HERE / "v2" / "config.json").read_text())
        mx.random.seed(settings["seed"])
        model = AlignedPointer(settings["model"], self.tok.vocab_size)
        row = self.examples[0]
        pairs, _ = encoded_row(self.tok, row)
        objective = nn.value_and_grad(model, aligned_loss)
        value, gradients = objective(model, pairs, row["gold_candidate_index"])
        mx.eval(value, gradients)
        flat = dict(tree_flatten(gradients))
        self.assertTrue(math.isfinite(float(value.item())))
        self.assertGreater(float(mx.sum(mx.abs(flat["importance.weight"])).item()), 0)
        self.assertGreater(float(mx.sum(mx.abs(flat["token.weight"])).item()), 0)
        self.assertTrue(any(name.startswith("blocks.") and float(mx.sum(mx.abs(grad)).item()) > 0
                            for name, grad in flat.items()))


if __name__ == "__main__":
    unittest.main()
