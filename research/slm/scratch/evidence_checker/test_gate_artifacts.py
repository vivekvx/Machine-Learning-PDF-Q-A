"""Regression checks for immutable scores, checkpoint restoration and test gate."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent
sys.path.insert(0, str(SCRATCH))
from evidence_checker.v2.prepare import sha, verify  # noqa: E402


class GateArtifactTests(unittest.TestCase):
    def test_frozen_data_and_pointer_hash(self):
        self.assertEqual(verify()["train"]["records"], 7200)
        run = json.loads((HERE / "v2" / "run" / "run.json").read_text())
        worker = run["worker_result"]
        pointer_step = json.loads((SCRATCH / "pointer_qa" / "v2" / "validation" /
                                   "pointer_selection.json").read_text())["checkpoint_step"]
        pointer_file = SCRATCH / "pointer_qa" / "v2" / "runs" / "main" / f"checkpoint-step-{pointer_step:04d}" / "model.safetensors"
        self.assertEqual(sha(pointer_file), worker["pointer_checkpoint_sha256"])
        self.assertEqual(worker["restoration"]["bce_difference"], 0)
        self.assertEqual(worker["restoration"]["exact_tensors"], 2)
        self.assertFalse(worker["test_file_opened"])

    def test_old_threshold_sweep_has_no_feasible_gate(self):
        audit = json.loads((HERE / "v1_threshold_audit.json").read_text())
        old_scores = SCRATCH / "pointer_qa" / "answerability" / "v1" / "run" / "validation_predictions.jsonl"
        self.assertEqual(audit["source_predictions_sha256"], sha(old_scores))
        self.assertFalse(audit["calibration"]["frozen_gate_feasible"])
        self.assertFalse(audit["assessment"]["frozen_gate_feasible"])

    def test_combined_no_go_overrides_gold_only_selection(self):
        gate = json.loads((HERE / "v2" / "combined_gate.json").read_text())
        self.assertTrue(gate["gold_candidate_gate_passed"])
        self.assertFalse(gate["pointer_selected_gate_passed"])
        self.assertFalse(gate["combined_pipeline_approved"])
        self.assertFalse(gate["test_may_be_opened"])
        self.assertEqual(gate["pointer_selected"]["confusion"]["FP"], 23)
        self.assertEqual(gate["pointer_selected_error_groups"]["correction"]["unsupported_false_answers"], 14)
        self.assertFalse(gate["sealed_test_read"])


if __name__ == "__main__":
    unittest.main()
