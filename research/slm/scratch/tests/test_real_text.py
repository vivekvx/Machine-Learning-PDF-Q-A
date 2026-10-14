"""Focused offline checks for the pinned CPython real-text pilot."""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real_text"))
sys.path.insert(0, str(ROOT))
from real_text.prepare import V1, build_rows, clean_span, encode_windows, rights_gate, sha256  # noqa: E402
from tokenizer import ByteBPE  # noqa: E402
from real_text import worker  # noqa: E402
from real_text.quality_audit import audit  # noqa: E402


class RealTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((V1 / "config.json").read_text())
        cls.manifest = json.loads((V1 / "manifest.json").read_text())
        cls.rows = build_rows(cls.config)
        cls.tok = json.loads((V1 / "tokenizer.json").read_text())

    def test_pinned_official_source_and_rights_notice(self):
        self.assertTrue(rights_gate(self.config))
        for name, expected in self.manifest["source_sha256"].items():
            self.assertEqual(sha256(V1 / "raw" / name), expected)
        self.assertEqual(self.manifest["rights_status"], "official_documentation_prose_only_verified_with_pinned_license")
        with patch("real_text.prepare.sha256", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "Pinned source bytes differ"):
                rights_gate(self.config)

    def test_document_split_and_clean_prose(self):
        train = [r for r in self.rows if r["split"] == "train"]
        val = [r for r in self.rows if r["split"] == "validation"]
        self.assertEqual((len(train), len(val)), (3, 1))
        self.assertFalse({r["document_id"] for r in train} & {r["document_id"] for r in val})
        self.assertEqual(len({p["sha256"] for r in self.rows for p in r["paragraphs"]}), 33)
        self.assertFalse(any("http" in p["text"] or ".. " in p["text"] or ">>>" in p["text"]
                             for r in self.rows for p in r["paragraphs"]))
        self.assertEqual(self.manifest["split_document_text_sha256"]["train"],
                         {r["document_id"]: r["text_sha256"] for r in train})
        self.assertEqual(self.manifest["split_document_text_sha256"]["validation"],
                         {r["document_id"]: r["text_sha256"] for r in val})

    def test_extractor_rejects_furniture_and_code(self):
        for bad in ([".. code-block:: python"], ["See http://example.com"], [">>> print(1)"], ["| table |"]):
            with self.assertRaises(ValueError):
                clean_span(bad, 1, 1)

    def test_tokenizer_fit_only_on_training_and_roundtrip(self):
        train = [r for r in self.rows if r["split"] == "train"]
        val = [r for r in self.rows if r["split"] == "validation"]
        self.assertEqual(self.tok["training_document_ids"], [r["document_id"] for r in train])
        self.assertFalse({r["document_id"] for r in val} & set(self.tok["training_document_ids"]))
        rebuilt = ByteBPE.train([r["text"] for r in train], self.config["tokenizer_target_size"])
        self.assertEqual([list(m) for m in rebuilt.merges], self.tok["merges"])
        for row in self.rows:
            self.assertEqual(rebuilt.decode(rebuilt.encode(row["text"])), row["text"])
        self.assertEqual(rebuilt.vocab_size, self.tok["vocab_size"])

    def test_fixed_windows_do_not_cross_documents(self):
        tokenizer = ByteBPE(self.tok["merges"], self.tok["target_size"])
        for split in ("train", "validation"):
            rows = [r for r in self.rows if r["split"] == split]
            windows = encode_windows(tokenizer, rows, self.config["sequence_length"], self.config[f"max_{split}_windows"])
            self.assertEqual(len(windows), self.manifest["fixed_sequence_windows"][split])
            self.assertTrue(all(len(w["ids"]) == 129 and w["document_id"] in {r["document_id"] for r in rows} for w in windows))

    def test_manifest_hashes_and_training_cap(self):
        for name, expected in self.manifest["file_sha256"].items():
            self.assertEqual(sha256(V1 / name), expected)
        self.assertLessEqual(self.config["max_optimizer_steps"], 100)
        self.assertEqual(self.config["batch_size"], 1)
        self.assertLessEqual(self.config["sequence_length"], 256)

    def test_training_requires_watchdog_and_verified_inputs(self):
        config, manifest, tokenizer, train, val, train_windows, val_windows = worker.load_verified()
        self.assertEqual((len(train_windows), len(val_windows)), (21, 8))
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"SLMA_SAFETY_PARENT": ""}):
            with self.assertRaisesRegex(RuntimeError, "watchdog"):
                worker.run(Path(tmp))

    def test_quality_audit_is_reproducible(self):
        self.assertEqual(audit(), json.loads((V1 / "quality_audit.json").read_text()))
        self.assertEqual(audit()["paragraph_count"], 33)


if __name__ == "__main__":
    unittest.main()
