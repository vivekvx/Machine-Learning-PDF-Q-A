import json
import sys
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

SCRATCH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRATCH), str(SCRATCH / "real_text"), str(SCRATCH / "optimization")]
from model import ScratchTransformer, next_token_loss  # noqa: E402
from expanded_worker import load_verified  # noqa: E402
from audit import reevaluate, tensor_stats, window_sha  # noqa: E402
from trial import verified_setup  # noqa: E402
from analyze import analyze  # noqa: E402


class StabilityAuditTests(unittest.TestCase):
    def test_frozen_checkpoints_same_windows_and_finite_state(self):
        result = reevaluate()
        self.assertEqual(result["window_counts"], {"train": 33, "validation": 23})
        for step in ("60", "80"):
            entry = result["checkpoints"][step]
            self.assertLessEqual(entry["train_record_difference"], 1e-5)
            self.assertLessEqual(entry["validation_record_difference"], 1e-5)
            self.assertEqual(entry["optimizer_step"], int(step))
            self.assertTrue(entry["model"]["all_finite"])
            self.assertTrue(entry["optimizer"]["all_finite"])
        self.assertGreater(result["checkpoints"]["80"]["train"]["loss"], result["checkpoints"]["60"]["train"]["loss"])
        self.assertGreater(result["checkpoints"]["80"]["validation"]["loss"], result["checkpoints"]["60"]["validation"]["loss"])

    def test_next_token_shift_and_fixed_sample_order(self):
        spec, old, manifest, tok, train, val, train_windows, val_windows = verified_setup()
        self.assertEqual(spec["control_learning_rate"], old["learning_rate"])
        self.assertEqual(spec["trial_learning_rate"], 0.0001)
        self.assertEqual(spec["optimizer_steps"], 80)
        by_id = {row["document_id"]: row for row in train + val}
        for w in train_windows + val_windows:
            self.assertEqual(len(w["ids"][:-1]), old["sequence_length"])
            encoded = tok.encode(by_id[w["document_id"]]["text"], bos=True, eos=True)
            self.assertEqual(w["ids"], encoded[w["offset"]:w["offset"] + old["sequence_length"] + 1])
        self.assertEqual(set(w["document_id"] for w in train_windows) & set(w["document_id"] for w in val_windows), set())
        audit = json.loads((SCRATCH / "optimization/v1/frozen_checkpoint_reevaluation.json").read_text())
        self.assertEqual(window_sha(train_windows), audit["train_window_sha256"])
        self.assertEqual(window_sha(val_windows), audit["validation_window_sha256"])

    def test_gradient_and_adamw_update_are_finite_and_nonzero(self):
        tiny = {"layers": 1, "embedding_dim": 16, "attention_heads": 2, "context_length": 8,
                "feed_forward_multiplier": 2, "normalization_epsilon": 1e-5}
        mx.random.seed(314)
        model = ScratchTransformer(tiny, 260)
        opt = optim.AdamW(learning_rate=0.0001, weight_decay=0.01)
        x = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
        y = mx.array([[6, 7, 8, 9]], dtype=mx.int32)
        mx.eval(model.parameters())
        before = model(x)
        mx.eval(before)
        loss, grads = nn.value_and_grad(model, next_token_loss)(model, x, y)
        mx.eval(loss, grads)
        self.assertTrue(tensor_stats(grads)["all_finite"])
        self.assertGreater(tensor_stats(grads)["l2_norm"], 0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        after = model(x)
        mx.eval(after)
        self.assertTrue(tensor_stats(model.parameters())["all_finite"])
        self.assertGreater(float(mx.max(mx.abs(after - before)).item()), 0)
        self.assertEqual(int(dict(tree_flatten(opt.state))["step"].item()), 1)

    def test_paired_curve_and_exact_best_final_tensor_restore(self):
        result = analyze()
        self.assertEqual([r["step"] for r in result["fixed_curve"]], list(range(0, 81, 10)))
        self.assertEqual(result["control_best_step"], 60)
        self.assertEqual(result["trial_best_step"], 70)
        for step in ("70", "80"):
            restored = result["trial_checkpoint_restoration"][step]
            self.assertGreater(restored["exact_model_tensors"], 0)
            self.assertGreater(restored["exact_optimizer_tensors"], 0)
            self.assertEqual(restored["optimizer_step"], int(step))

    def test_eval_mode_is_numerically_identical_for_this_architecture(self):
        config, manifest, tok, train, val, train_windows, val_windows = load_verified()
        model_config = json.loads((SCRATCH / "config.v1.json").read_text())["model"]
        mx.random.seed(config["seed"])
        model = ScratchTransformer(model_config, tok.vocab_size)
        mx.eval(model.parameters())
        x = mx.array([val_windows[0]["ids"][:-1]], dtype=mx.int32)
        model.train()
        train_logits = model(x)
        model.eval()
        eval_logits = model(x)
        mx.eval(train_logits, eval_logits)
        self.assertTrue(bool(mx.all(train_logits == eval_logits).item()))


if __name__ == "__main__":
    unittest.main()
