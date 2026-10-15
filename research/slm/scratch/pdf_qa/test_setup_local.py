"""Focused, offline checks for the portable installer audit."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.slm.scratch.pdf_qa import setup_local


class SetupLocalTests(unittest.TestCase):
    def test_artifact_status_requires_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            expected = hashlib.sha256(b"frozen-test-bytes").hexdigest()
            self.assertEqual(setup_local.artifact_status(path, expected), "missing")
            path.write_bytes(b"frozen-test-bytes")
            self.assertEqual(setup_local.artifact_status(path, expected), "verified")
            path.write_bytes(b"changed")
            self.assertEqual(setup_local.artifact_status(path, expected), "checksum_mismatch")

    def test_audit_refuses_missing_pointer_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "selection.json"
            with patch.object(setup_local, "POINTER_SELECTION", missing), \
                 patch.object(setup_local, "environment_status", return_value="verified"), \
                 patch.object(setup_local, "artifact_status", return_value="verified"), \
                 patch.object(setup_local.sys, "platform", "darwin"), \
                 patch.object(setup_local.platform, "machine", return_value="arm64"):
                status = setup_local.audit()
            self.assertFalse(status["pointer_ready"])
            self.assertEqual(status["pointer"], {"selection_ledger": "missing"})

    def test_audit_keeps_pointer_and_qwen_readiness_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "selection.json"
            ledger.write_text(json.dumps({"checkpoint_sha256": "a", "tokenizer_sha256": "b"}))
            def status(path, _expected):
                return "missing" if path.name == "model.safetensors" and path.parent == setup_local.MODEL else "verified"
            with patch.object(setup_local, "POINTER_SELECTION", ledger), \
                 patch.object(setup_local, "environment_status", return_value="verified"), \
                 patch.object(setup_local, "artifact_status", side_effect=status), \
                 patch.object(setup_local.sys, "platform", "darwin"), \
                 patch.object(setup_local.platform, "machine", return_value="arm64"):
                result = setup_local.audit()
            self.assertTrue(result["pointer_ready"])
            self.assertFalse(result["qwen_ready"])


if __name__ == "__main__":
    unittest.main()
