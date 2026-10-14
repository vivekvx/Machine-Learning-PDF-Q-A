import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from extractive_qa.model import ExtractiveTransformer, FullAttention, loss
from extractive_qa.prepare import V1, input_text, tokenized, verify
from extractive_qa.train import config, load_plan, rows, tokenizer
from extractive_qa.evaluate import rule, summarize
from checkpoint import load_checkpoint


class ExtractiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify()
        cls.train = rows("train")
        cls.val = rows("validation")
        cls.tok = tokenizer()

    def test_split_and_label_balance(self):
        manifest = json.loads((V1 / "manifest.json").read_text())
        names = manifest["split_families"]
        self.assertFalse(set(names["train"]) & set(names["validation"] + names["test"]))
        self.assertFalse(set(names["validation"]) & set(names["test"]))
        self.assertTrue({r["entity"] for r in self.train}.isdisjoint(r["entity"] for r in self.val))
        self.assertTrue({r["fact_id"] for r in self.train}.isdisjoint(r["fact_id"] for r in self.val))
        for subset in (self.train, self.val):
            counts = Counter((r["asked_field"], r["answerable"]) for r in subset)
            self.assertEqual(len(set(counts.values())), 1)
            self.assertEqual(len({(r["passage"], r["question"]) for r in subset}), len(subset))

    def test_exact_char_and_token_spans(self):
        for row in self.train[:80] + self.val[:80]:
            encoded = tokenized(self.tok, row)
            self.assertEqual(self.tok.decode(encoded["ids"]), input_text(row["passage"], row["question"]))
            self.assertLessEqual(len(encoded["ids"]), 192)
            if row["answerable"]:
                self.assertEqual(row["passage"][row["answer_char_start"]:row["answer_char_end"]], row["expected_answer"])
                self.assertEqual(self.tok.decode(encoded["ids"][encoded["answer_start_token"]:encoded["answer_end_token"] + 1]),
                                 row["expected_answer"])
            else:
                self.assertIsNone(encoded["answer_start_token"])
                self.assertNotIn(row["asked_field"] + " = [", row["target_line"])
                self.assertIn(row["asked_field"] + " = [", row["distractor_line"])

    def test_attention_sees_future_positions(self):
        mx.random.seed(101)
        full = FullAttention(16, 4)
        from model import CausalAttention
        causal = CausalAttention(16, 4)
        causal.update(full.parameters())
        first = mx.random.normal((1, 4, 16))
        changed = mx.concatenate([first[:, :3, :], first[:, 3:, :] + 10], axis=1)
        full_delta = float(mx.max(mx.abs(full(first)[:, 0, :] - full(changed)[:, 0, :])).item())
        causal_delta = float(mx.max(mx.abs(causal(first)[:, 0, :] - causal(changed)[:, 0, :])).item())
        self.assertGreater(full_delta, 1e-8)
        self.assertLess(causal_delta, 1e-8)

    def test_question_changes_passage_logits(self):
        row = next(r for r in self.val if r["answerable"])
        encoded = tokenized(self.tok, row)
        model = ExtractiveTransformer(config()["model"], self.tok.vocab_size)
        ids = list(encoded["ids"])
        modified = list(ids)
        modified[-2] = (modified[-2] + 37) % self.tok.vocab_size
        first = model(mx.array([ids], dtype=mx.int32), encoded["passage_start"], encoded["passage_end"])[0]
        second = model(mx.array([modified], dtype=mx.int32), encoded["passage_start"], encoded["passage_end"])[0]
        delta = float(mx.max(mx.abs(first[:, encoded["passage_start"]:encoded["passage_end"]] -
                                    second[:, encoded["passage_start"]:encoded["passage_end"]])).item())
        self.assertGreater(delta, 1e-8)

    def test_nonzero_finite_gradient(self):
        row = next(r for r in self.train if r["answerable"])
        encoded = tokenized(self.tok, row)
        model = ExtractiveTransformer(config()["model"], self.tok.vocab_size)
        objective = nn.value_and_grad(model, loss)
        value, grads = objective(model, mx.array([encoded["ids"]], dtype=mx.int32),
                                 encoded["passage_start"], encoded["passage_end"], True,
                                 encoded["answer_start_token"], encoded["answer_end_token"], 1.0, 1.0)
        mx.eval(value, grads)
        self.assertTrue(bool(mx.isfinite(value).item()))
        norm = sum(float(mx.sum(v * v).item()) for _, v in tree_flatten(grads))
        self.assertGreater(norm, 0)

    def test_frozen_plan_excludes_test(self):
        p = load_plan()
        self.assertEqual(len(p["sample_order"]), 1600)
        self.assertEqual(len(set(p["sample_order"])), 1600)
        self.assertTrue(all(x.startswith("train-") for x in p["sample_order"] + p["fixed_train_ids"]))
        self.assertTrue(all(x.startswith("validation-") for x in p["fixed_validation_ids"]))
        self.assertEqual(p["parameter_count_expected"], 951811)

    def test_rule_baseline_on_validation_only(self):
        self.assertTrue(all(rule(r["passage"], r["question"])["prediction"] == r["expected_answer"]
                            for r in self.val))

    def test_confusion_matrix_orientation(self):
        records = [{"answerable": True, "prediction": "value", "exact_match": True, "token_f1": 1.0, "span_support": True},
                   {"answerable": True, "prediction": "UNANSWERABLE", "exact_match": False, "token_f1": 0.0, "span_support": False},
                   {"answerable": False, "prediction": "wrong", "exact_match": False, "token_f1": 0.0, "span_support": False},
                   {"answerable": False, "prediction": "UNANSWERABLE", "exact_match": True, "token_f1": 1.0, "span_support": False}]
        matrix = summarize(records)["answerability_confusion_matrix"]
        self.assertEqual(list(matrix.values()), [1, 1, 1, 1])

    def test_selected_checkpoint_reloads(self):
        run = json.loads((V1 / "run-001/run.json").read_text())
        self.assertEqual(run["status"], "completed")
        selected = run["worker_result"]["best_step"]
        model = ExtractiveTransformer(config()["model"], self.tok.vocab_size)
        optimizer = optim.AdamW(learning_rate=config()["training"]["learning_rate"],
                                weight_decay=config()["training"]["weight_decay"])
        metadata = load_checkpoint(V1 / f"run-001/checkpoint-step-{selected:04d}", model, optimizer,
                                   expected_provenance=run["worker_result"]["provenance"])
        self.assertEqual(metadata["step"], selected)


if __name__ == "__main__":
    unittest.main()
