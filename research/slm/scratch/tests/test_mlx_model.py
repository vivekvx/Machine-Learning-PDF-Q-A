import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from checkpoint import load_checkpoint, save_checkpoint
from model import ScratchTransformer, next_token_loss, parameter_count
from run import CONFIG, preflight


READY = preflight("smoke")["ready"]
SMALL = {"layers": 1, "embedding_dim": 16, "attention_heads": 2,
         "context_length": 8, "feed_forward_multiplier": 2,
         "normalization_epsilon": 1e-5}


@unittest.skipUnless(READY, "MLX execution skipped: memory preflight is not safe")
class MLXModelTests(unittest.TestCase):
    def test_actual_parameter_count(self):
        config = json.loads(CONFIG.read_text())
        mx.random.seed(config["seed"])
        model = ScratchTransformer(config["model"], 2000)
        self.assertEqual(parameter_count(model), 1_332_352)

    def test_causal_prefix_invariance_and_shape(self):
        mx.random.seed(19)
        model = ScratchTransformer(SMALL, 260)
        left = mx.array([[5, 9, 11, 13]], dtype=mx.int32)
        right = mx.array([[5, 9, 99, 101]], dtype=mx.int32)
        a = model(left)
        b = model(right)
        mx.eval(a, b)
        self.assertEqual(a.shape, (1, 4, 260))
        self.assertTrue(bool(mx.all(mx.abs(a[:, :2] - b[:, :2]) < 1e-5).item()))
        with self.assertRaises(ValueError):
            model(mx.array([[5] * 9], dtype=mx.int32))

    def test_gradient_and_checkpoint_resume_are_reproducible(self):
        x = mx.array([[4, 5, 6, 7]], dtype=mx.int32)
        y = mx.array([[5, 6, 7, 8]], dtype=mx.int32)

        def fresh():
            mx.random.seed(23)
            model = ScratchTransformer(SMALL, 260)
            mx.eval(model.parameters())
            return model, optim.AdamW(learning_rate=0.001, weight_decay=0.01)

        def step(model, optimizer):
            loss, grads = nn.value_and_grad(model, next_token_loss)(model, x, y)
            flat = tree_flatten(grads)
            mx.eval(loss, grads)
            self.assertTrue(any(bool(mx.any(mx.abs(v) > 0).item()) for _, v in flat))
            self.assertTrue(bool(mx.isfinite(loss).item()))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            return float(loss.item())

        uninterrupted, full_opt = fresh()
        first_loss = step(uninterrupted, full_opt)
        second_loss = step(uninterrupted, full_opt)
        self.assertLess(second_loss, first_loss)
        source, source_opt = fresh()
        self.assertAlmostEqual(step(source, source_opt), first_loss, places=5)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp) / "ckpt", source, source_opt, step=1, cursor=1,
                            provenance={"test": "reproducibility"})
            resumed, resumed_opt = fresh()
            metadata = load_checkpoint(Path(tmp) / "ckpt", resumed, resumed_opt,
                                       expected_provenance={"test": "reproducibility"})
            self.assertEqual(metadata["step"], 1)
            self.assertAlmostEqual(step(resumed, resumed_opt), second_loss, places=5)
            for (_, expected), (_, actual) in zip(tree_flatten(uninterrupted.parameters()),
                                                  tree_flatten(resumed.parameters())):
                self.assertTrue(bool(mx.all(mx.abs(expected - actual) < 1e-6).item()))


if __name__ == "__main__":
    unittest.main()
