"""Focused QA supervision, visibility, reproducibility, and guard tests."""
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
from checkpoint import load_checkpoint  # noqa: E402
from run import guard_reasons  # noqa: E402
from extractive_qa.model import FullAttention  # noqa: E402
from working_qa.model import CandidateReader, loss, predict  # noqa: E402
from working_qa.prepare import V1, entity_pools, enumerate_candidates, tokenize, verify  # noqa: E402
from working_qa.train import config, load_plan, provenance, rows, tokenizer  # noqa: E402
from working_qa.evaluate import evaluate as evaluate_split  # noqa: E402
from working_qa.rule_baseline import evaluate_validation as rule_validation  # noqa: E402
from working_qa.v2.model import PairReader, loss as pair_loss  # noqa: E402
from working_qa.v2.prepare import pair_inputs  # noqa: E402


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = rows("train")
        cls.validation = rows("validation")
        cls.tok = tokenizer()

    def test_frozen_hashes_and_disjoint_source_units(self):
        verify()
        plan = load_plan()
        self.assertTrue(plan["test_sealed"])
        self.assertEqual(len(self.training), 2400)
        self.assertEqual(len(self.validation), 320)
        self.assertFalse({r["id"] for r in self.training} & {r["id"] for r in self.validation})
        self.assertFalse({r["family"] for r in self.training} & {r["family"] for r in self.validation})
        pools = entity_pools(config()["seed"])
        self.assertFalse(set(pools["train"]) & set(pools["validation"]))
        self.assertFalse(set(pools["train"]) & set(pools["test"]))
        self.assertFalse(set(pools["validation"]) & set(pools["test"]))
        self.assertFalse(any(id.startswith("test-") for id in plan["main_order"] + plan["fixed_validation_ids"]))

    def test_balanced_labels_and_no_exact_duplicate(self):
        for subset in (self.training, self.validation):
            self.assertEqual(sum(r["answerable"] for r in subset), len(subset) // 2)
            self.assertEqual(len({(r["passage"], r["question"]) for r in subset}), len(subset))
            for tier in config()["tiers"]:
                labels = Counter(r["answerable"] for r in subset if r["tier"] == tier)
                self.assertEqual(labels[True], labels[False])

    def test_answer_labels_source_spans_and_train_only_tokenizer(self):
        tokenizer_info = json.loads((V1 / "tokenizer.json").read_text())
        self.assertEqual(tokenizer_info["target_size"], self.tok.vocab_size)
        self.assertEqual(set(tokenizer_info["training_ids"]), {r["id"] for r in self.training})
        for row in self.training[:128] + self.validation[:128]:
            candidates = enumerate_candidates(row["passage"])
            self.assertEqual([(c["start"], c["end"]) for c in candidates],
                             [(c["start"], c["end"]) for c in row["candidates"]])
            enc = tokenize(self.tok, row)
            self.assertLessEqual(len(enc["ids"]), config()["training"]["maximum_sequence_length"])
            pairs, paired = pair_inputs(self.tok, row["passage"], row["question"])
            self.assertEqual(len(pairs), len(paired))
            if row["answerable"]:
                index = row["gold_candidate_index"]
                self.assertEqual(row["passage"][candidates[index]["start"]:candidates[index]["end"]],
                                 row["expected_answer"])
            else:
                self.assertIsNone(row["gold_candidate_index"])
                self.assertEqual(row["expected_answer"], "UNANSWERABLE")
        # Byte fallback has no unknown token even for unseen UTF-8 text.
        self.assertEqual(self.tok.decode(self.tok.encode("unseen Ω symbol")), "unseen Ω symbol")

    def test_candidate_parser_is_question_independent(self):
        row = self.validation[0]
        one = enumerate_candidates(row["passage"])
        self.assertEqual(one, enumerate_candidates(row["passage"]))
        self.assertTrue(all("field" not in item and "entity" not in item for item in one))

    def test_rule_is_separate_and_narrow(self):
        result = rule_validation()
        self.assertEqual(result["answerable_exact"], 160)
        self.assertEqual(result["unanswerable_correct"], 160)

    def test_sealed_test_requires_passed_validation_lock(self):
        self.assertFalse((V1 / "analysis" / "validation_best" / "frozen_selection.json").exists())
        with self.assertRaises(PermissionError):
            evaluate_split("test")


class ModelAndGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = rows("train")[1]
        cls.tok = tokenizer()

    def test_bidirectional_visibility(self):
        mx.random.seed(23)
        attention = FullAttention(128, 4)
        x = mx.random.normal((1, 3, 128))
        changed = mx.concatenate((x[:, :2, :], x[:, 2:, :] + 3), axis=1)
        a, b = attention(x), attention(changed)
        mx.eval(a, b)
        self.assertGreater(float(mx.max(mx.abs(a[:, 0, :] - b[:, 0, :])).item()), 1e-5)

    def test_neural_gradients_and_parameter_update(self):
        c = config()
        mx.random.seed(11)
        model = CandidateReader(c["model"], self.tok.vocab_size)
        encoded = tokenize(self.tok, self.row)
        gold = self.row["gold_candidate_index"] + 1
        objective = nn.value_and_grad(model, loss)
        value, grads = objective(model, mx.array([encoded["ids"]]), encoded["question_end"],
                                 encoded["candidates"], gold)
        mx.eval(value, grads)
        norm = sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(grads))
        self.assertTrue(math.isfinite(norm) and norm > 0)
        old = model.candidate_out.weight + mx.zeros_like(model.candidate_out.weight)
        mx.eval(old)
        optimizer = optim.AdamW(learning_rate=0.001)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        self.assertGreater(float(mx.max(mx.abs(old - model.candidate_out.weight)).item()), 0)
        pair_model = PairReader(c["model"], self.tok.vocab_size)
        pairs, _ = pair_inputs(self.tok, self.row["passage"], self.row["question"])
        pair_objective = nn.value_and_grad(pair_model, pair_loss)
        pair_value, pair_grads = pair_objective(pair_model, pairs, self.row["gold_candidate_index"])
        mx.eval(pair_value, pair_grads)
        self.assertGreater(sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(pair_grads)), 0)

    def test_checkpoint_restoration_and_predictions(self):
        c, p = config(), load_plan()
        run = json.loads((V1 / "runs" / "main" / "worker_result.json").read_text())
        saved = V1 / "runs" / "main" / f"checkpoint-step-{run['best_step']:04d}"
        a = CandidateReader(c["model"], self.tok.vocab_size)
        b = CandidateReader(c["model"], self.tok.vocab_size)
        aa = optim.AdamW(learning_rate=c["training"]["learning_rate"])
        bb = optim.AdamW(learning_rate=c["training"]["learning_rate"])
        load_checkpoint(saved, a, aa, expected_provenance=provenance())
        load_checkpoint(saved, b, bb, expected_provenance=provenance())
        encoded = tokenize(self.tok, self.row)
        self.assertEqual(predict(a, encoded, self.row["candidates"]),
                         predict(b, encoded, self.row["candidates"]))
        self.assertEqual(run["parameter_count"], p["expected_parameter_count"])

    def test_existing_resource_guards(self):
        safety = json.loads((HERE.parent / "config.v1.json").read_text())["safety"]
        base = {"available_bytes": 2_000_000_000, "swap_used_bytes": 0,
                "swap_out_total_bytes": 0, "process_rss_bytes": 100_000_000,
                "memory_pressure_free_percent": 40, "disk_free_bytes": 3_000_000_000}
        self.assertEqual(guard_reasons(base, base, safety), [])
        too_much_swap = {**base, "swap_used_bytes": safety["stop_swap_growth_bytes"] + 1}
        self.assertIn("swap growth above stop ceiling", guard_reasons(too_much_swap, base, safety))
        low_disk = {**base, "disk_free_bytes": safety["min_preflight_disk_free_bytes"] - 1}
        self.assertIn("disk headroom below stop floor", guard_reasons(low_disk, base, safety))
        low_pressure = {**base, "memory_pressure_free_percent": safety["stop_pressure_free_percent"] - 1}
        self.assertIn("memory pressure unavailable or below stop floor", guard_reasons(low_pressure, base, safety))


if __name__ == "__main__":
    unittest.main()
