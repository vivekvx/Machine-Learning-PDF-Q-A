import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRATCH = Path(__file__).resolve().parents[1]
REAL = SCRATCH / "real_text"
sys.path[:0] = [str(SCRATCH), str(REAL)]
from checkpoint import digest  # noqa: E402
from v3_prepare import V3, MEDICAL, read_archive, verify, similar_pairs, page_family  # noqa: E402
from v3_data import load_verified, load_plan, identity  # noqa: E402
from v3_analyze import analyze  # noqa: E402


class CorpusV3Tests(unittest.TestCase):
    def test_official_archive_license_and_manifest_hashes(self):
        config = json.loads((V3 / "config.json").read_text())
        files = read_archive(config)
        self.assertIn("Doc/license.rst", files)
        self.assertEqual(verify()["status"], "verified")
        manifest = json.loads((V3 / "manifest.json").read_text())
        self.assertEqual(digest(V3 / "source/cpython-e848e4b.tar.gz"), manifest["archive_sha256"])
        self.assertEqual(manifest["license"], "Python Software Foundation License Version 2")

    def test_split_grouping_leakage_and_minimum_corpus_gate(self):
        manifest = json.loads((V3 / "manifest.json").read_text())
        config = json.loads((V3 / "config.json").read_text())
        self.assertGreaterEqual(manifest["split_stats"]["train"]["unique_text_tokens"], config["min_unique_train_tokens_for_pilot"])
        self.assertLessEqual(manifest["split_stats"]["train"]["unique_text_tokens"], config["target_max_unique_train_tokens"])
        leakage = json.loads((V3 / "leakage_audit.json").read_text())
        self.assertEqual(leakage["cross_split_paragraph_matches"], [])
        self.assertEqual(leakage["cross_split_near_documents"], [])
        train = [json.loads(x) for x in (V3 / "train.jsonl").read_text().splitlines()]
        val = [json.loads(x) for x in (V3 / "validation.jsonl").read_text().splitlines()]
        self.assertFalse({r["document_id"] for r in train} & {r["document_id"] for r in val})
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in val})
        for row in train + val:
            self.assertFalse(MEDICAL.search(row["title"]))
            self.assertTrue(all(not MEDICAL.search(p["text"]) for p in row["paragraphs"]))

    def test_train_only_tokenizer_and_frozen_windows(self):
        config, manifest, tok, train, val, train_windows, val_windows = load_verified()
        plan = load_plan(config, train_windows, val_windows)
        self.assertEqual(len(plan["fixed_train"]), 64)
        self.assertGreaterEqual(len({w["document_id"] for w in plan["fixed_validation"]}), 10)
        self.assertEqual(len(plan["training_sample_order"]), 100)
        self.assertEqual(len({(w["document_id"], w["offset"]) for w in plan["training_sample_order"]}), 100)
        self.assertEqual(len({w["group_id"] for w in plan["training_sample_order"]}), 44)
        tokdata = json.loads((V3 / "tokenizer.json").read_text())
        self.assertEqual(tokdata["training_document_ids"], [r["document_id"] for r in train])
        for row in (train[:3] + val[:3]):
            self.assertEqual(tok.decode(tok.encode(row["text"])), row["text"])
        by_id = {r["document_id"]: r for r in train + val}
        for window in plan["fixed_train"] + plan["fixed_validation"] + plan["training_sample_order"]:
            encoded = tok.encode(by_id[window["document_id"]]["text"], bos=True, eos=True)
            self.assertEqual(window["ids"], encoded[window["offset"]:window["offset"] + 129])

    def test_training_loader_never_reads_sealed_test(self):
        original = Path.read_text

        def blocked(path, *args, **kwargs):
            if path.name == "test.sealed.jsonl":
                raise AssertionError("Sealed test was opened")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", blocked):
            config, manifest, tok, train, val, tw, vw = load_verified()
            load_plan(config, tw, vw)

    def test_training_is_opt_in_with_strict_step_and_wall_caps(self):
        config = json.loads((V3 / "pilot_config.v1.json").read_text())
        self.assertEqual(config["optimizer_steps_cap"], 100)
        self.assertEqual(config["target_token_budget"], 12800)
        self.assertEqual(config["wall_time_cap_seconds"], 180)
        self.assertEqual(config["learning_rate"], 0.0001)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["learning_rate_schedule"], "constant")

    def test_duplicate_detector_and_related_page_family(self):
        self.assertEqual(page_family("Doc/library/asyncio-eventloop.rst"), page_family("Doc/library/asyncio-protocol.rst"))
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        items = [{"document_id": "a", "text": words},
                 {"document_id": "b", "text": words},
                 {"document_id": "c", "text": words.replace(" mu", " nu")}]
        config = json.loads((V3 / "config.json").read_text())
        matches = similar_pairs(items, config)
        self.assertTrue(any(kind == "exact" for _, _, _, kind in matches))
        self.assertTrue(any(kind == "near" for _, _, _, kind in matches))

    def test_guarded_run_curve_and_checkpoint_restore(self):
        result = analyze()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["training_target_tokens_processed"], 12800)
        self.assertEqual(result["sampled_unique_windows"], 100)
        self.assertEqual(result["scheduled_loss_rises"], {"fixed_train": [], "document_balanced_validation": []})
        self.assertEqual(result["best_validation_step"], 100)
        restored = result["restored_checkpoints"]["100"]
        self.assertEqual(restored["exact_model_tensors"], 28)
        self.assertEqual(restored["exact_optimizer_tensors"], 58)
        self.assertEqual(restored["probe_max_abs_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
