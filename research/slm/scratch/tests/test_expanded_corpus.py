"""Rights, split, leakage, tokenizer and sealed-test checks for CPython v2."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real_text"))
sys.path.insert(0, str(ROOT))
from real_text.expanded_prepare import V2, build_rows, leakage_audit, rights_gate, verify, windows  # noqa: E402
from real_text.expanded_worker import load_verified, restore_and_verify, run  # noqa: E402
from tokenizer import ByteBPE  # noqa: E402


class ExpandedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((V2 / "config.json").read_text())
        cls.manifest = json.loads((V2 / "manifest.json").read_text())

    def test_rights_and_rebuild(self):
        self.assertIsNone(rights_gate(self.config))
        self.assertTrue(verify()["verified"])
        wrong = dict(self.config)
        wrong["source_commit"] = "changed"
        with self.assertRaisesRegex(ValueError, "revision"):
            rights_gate(wrong)
        with patch("real_text.expanded_prepare.sha256", return_value="tampered"):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                rights_gate(self.config)

    def test_document_disjoint_and_test_sealed(self):
        rows = build_rows(self.config)
        self.assertEqual(len(rows), 12)
        self.assertEqual({split: len([r for r in rows if r["split"] == split]) for split in ("train", "validation", "test")},
                         {"train": 6, "validation": 3, "test": 3})
        self.assertEqual(len({r["document_id"] for r in rows}), 12)
        self.assertTrue(all(r["selection_method"] == "sealed_first_clean_paragraphs" for r in rows if r["split"] == "test"))
        quality = json.loads((V2 / "quality_samples.json").read_text())
        self.assertFalse(any(x["split"] == "test" for x in quality["samples"]))
        self.assertTrue(self.manifest["test_sealed_for_training_and_model_selection"])

    def test_leakage_detector_catches_exact_and_near_duplicates(self):
        rows = build_rows(self.config)
        self.assertFalse(leakage_audit(rows, self.config)["cross_split_near_paragraph_pairs"])
        sample = "A deterministic technical sentence with enough distinct words to form several matching five word shingles about data processing."
        fake = [{"split": "train", "document_id": "a", "text": sample,
                 "paragraphs": [{"text": sample, "source_lines": [1, 1]}]},
                {"split": "validation", "document_id": "b", "text": sample,
                 "paragraphs": [{"text": sample, "source_lines": [1, 1]}]}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            leakage_audit(fake, self.config)
        fake[1]["text"] = sample.replace("technical", "practical")
        fake[1]["paragraphs"][0]["text"] = fake[1]["text"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            leakage_audit(fake, self.config)

    def test_train_only_tokenizer_and_fixed_windows(self):
        config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
        tokdata = json.loads((V2 / "tokenizer.json").read_text())
        self.assertEqual(tokdata["training_document_ids"], [r["document_id"] for r in train])
        self.assertFalse(set(tokdata["training_document_ids"]) & {r["document_id"] for r in val})
        rebuilt = ByteBPE.train([r["text"] for r in train], config["tokenizer_target_size"])
        self.assertEqual([list(x) for x in rebuilt.merges], tokdata["merges"])
        self.assertEqual((len(train_windows), len(val_windows)), (33, 23))
        self.assertTrue(all(len(w["ids"]) == 129 for w in train_windows + val_windows))
        self.assertEqual(set(w["document_id"] for w in val_windows), {r["document_id"] for r in val})
        self.assertEqual(tokdata["vocab_size"], 1024)

    def test_training_loader_never_opens_sealed_test(self):
        original_read_text = Path.read_text
        original_read_bytes = Path.read_bytes

        def safe_text(path, *args, **kwargs):
            if path.name == "test.sealed.jsonl" or path.name in ("logging.rst", "re.rst", "io.rst"):
                raise AssertionError("Training loader accessed sealed test")
            return original_read_text(path, *args, **kwargs)

        def safe_bytes(path, *args, **kwargs):
            if path.name == "test.sealed.jsonl" or path.name in ("logging.rst", "re.rst", "io.rst"):
                raise AssertionError("Training loader accessed sealed test")
            return original_read_bytes(path, *args, **kwargs)

        with patch.object(Path, "read_text", safe_text), patch.object(Path, "read_bytes", safe_bytes):
            load_verified()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SLMA_SAFETY_PARENT": ""}):
            with self.assertRaisesRegex(RuntimeError, "watchdog"):
                run(Path(tmp))

    def test_saved_best_and_final_checkpoint_restore(self):
        run_dir = ROOT / "runs/v2-expanded-cpython"
        result_file = run_dir / "worker_result.json"
        if not result_file.exists():
            self.skipTest("Bounded v2 run has not been attempted yet")
        result = json.loads(result_file.read_text())
        config, manifest, tokenizer, train, val, train_windows, val_windows = load_verified()
        base = json.loads((ROOT / "config.v1.json").read_text())
        for key in ("best_checkpoint", "final_checkpoint"):
            _, checked = restore_and_verify(run_dir / result[key], base["model"], tokenizer,
                                            config, result["provenance"], val_windows[0])
            self.assertLessEqual(checked["max_absolute_probe_logit_difference"], 1e-6)


if __name__ == "__main__":
    unittest.main()
