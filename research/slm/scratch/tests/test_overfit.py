import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from checkpoint import digest
from export_curve import export
from model import ScratchTransformer, next_token_loss
from overfit_run import validate_experiment
from overfit_worker import gradient_l1_by_group, load_fixture
from prepare_overfit import BASE_CONFIG, DEFAULT_OUT, build
from run import preflight
from tokenizer import SPECIAL


class OverfitFixtureTests(unittest.TestCase):
    def test_byte_reproducible_synthetic_fixture(self):
        saved = json.loads((DEFAULT_OUT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = build(Path(tmp) / "v1")
            self.assertEqual(rebuilt, saved)
            self.assertEqual(digest(Path(tmp) / "v1/data.jsonl"), saved["data_sha256"])
        self.assertTrue(saved["synthetic_only"])
        self.assertEqual(saved["academic_qa_records"], 0)
        self.assertEqual(saved["sequence_count"], 6)

    def test_targets_are_exactly_next_tokens_and_fit_context(self):
        tokenizer, rows, encoded = load_fixture()
        context = json.loads(BASE_CONFIG.read_text())["model"]["context_length"]
        self.assertEqual(len(rows), 6)
        for row, ids in zip(rows, encoded):
            self.assertEqual(ids[0], SPECIAL["bos"])
            self.assertEqual(ids[-1], SPECIAL["eos"])
            self.assertEqual(tokenizer.decode(ids), row["text"])
            self.assertLessEqual(len(ids) - 1, context)
            x, y = ids[:-1], ids[1:]
            self.assertEqual(x[1:], y[:-1])
            self.assertEqual(y[-1], SPECIAL["eos"])

    def test_experiment_caps_and_hashes(self):
        config, reasons = validate_experiment()
        self.assertEqual(reasons, [])
        self.assertLessEqual(config["max_optimizer_steps"], 200)
        self.assertLessEqual(config["wall_time_cap_seconds"], 120)
        self.assertEqual(config["batch_size"], 1)

    def test_curve_export_preserves_step_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [{"step": 0, "full_set_loss": 7.5, "exact_reproductions": 0},
                    {"step": 10, "full_set_loss": 2.0, "exact_reproductions": 1,
                     "continuations_evaluated": True}]
            (run_dir / "curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(export(run_dir)["rows"], 2)
            self.assertIn("10,2.0,1,True", (run_dir / "curve.csv").read_text())


@unittest.skipUnless(preflight("smoke")["ready"], "MLX gradient test skipped by memory preflight")
class OverfitGradientTests(unittest.TestCase):
    def test_all_required_model_groups_receive_gradients(self):
        base = json.loads(BASE_CONFIG.read_text())
        _, _, encoded = load_fixture()
        ids = encoded[0]
        mx.random.seed(base["seed"])
        model = ScratchTransformer(base["model"], 2000)
        objective = nn.value_and_grad(model, next_token_loss)
        loss, grads = objective(model, mx.array([ids[:-1]], dtype=mx.int32),
                                mx.array([ids[1:]], dtype=mx.int32))
        mx.eval(loss, grads)
        norms = gradient_l1_by_group(grads)
        self.assertTrue(all(value > 0 for value in norms.values()), norms)
        self.assertTrue(bool(mx.isfinite(loss).item()))
        self.assertGreater(len(tree_flatten(grads)), 0)


if __name__ == "__main__":
    unittest.main()
