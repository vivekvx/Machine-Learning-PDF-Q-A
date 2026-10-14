"""Reproducibility and target-masking checks for the bounded training sample."""
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


class FeasibilitySampleTests(unittest.TestCase):
    def test_sample_counts_splits_and_hashes(self):
        manifest = json.loads((ROOT / "feasibility_data/manifest.json").read_text())
        train_path = ROOT / "feasibility_data/train.jsonl"
        valid_path = ROOT / "feasibility_data/valid.jsonl"
        train, valid = rows(train_path), rows(valid_path)
        self.assertEqual((len(train), len(valid)), (32, 8))
        for path, label, sample, expected in ((train_path, "train", train, (24, 8)),
                                              (valid_path, "valid", valid, (4, 4))):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["file_sha256"][label])
            self.assertEqual({x["split"] for x in sample},
                             {"validation" if label == "valid" else "train"})
            self.assertEqual((sum(x["answerability"] == "answerable" for x in sample),
                              sum(x["answerability"] == "unanswerable" for x in sample)), expected)
            self.assertEqual(len({x["document_id"] for x in sample}), len(sample))
            self.assertTrue(all(x["full_tokens"] <= 384 for x in sample))
            self.assertTrue(all(x["prompt_tokens"] < x["full_tokens"] - 1 for x in sample))
        self.assertFalse({x["document_id"] for x in train} & {x["document_id"] for x in valid})
        baseline_docs = {x["document_id"] for x in rows(ROOT / "validation_gold_eval.jsonl")}
        self.assertFalse(baseline_docs & {x["document_id"] for x in valid})
        self.assertEqual(hashlib.sha256((ROOT / "curated_manifest.json").read_bytes()).hexdigest(),
                         manifest["source_curated_manifest_sha256"])

    def test_assistant_only_targets_and_template(self):
        from mlx_lm.utils import load_tokenizer
        from training_child import QwenMaskedDataset

        tokenizer = load_tokenizer(str(ROOT / ".cache/qwen3-0.6b-4bit"))
        for name in ("train", "valid"):
            dataset = QwenMaskedDataset(ROOT / f"feasibility_data/{name}.jsonl", tokenizer)
            for row in dataset._data:
                tokens, offset = dataset.process(row)
                self.assertEqual(offset, row["prompt_tokens"])
                self.assertEqual(len(tokens) - offset, row["completion_tokens"])
                target = row["messages"][-1]["content"]
                if row["answerability"] == "unanswerable":
                    self.assertEqual(target, "UNANSWERABLE")
                else:
                    self.assertNotEqual(target, "UNANSWERABLE")
                    self.assertIn(normalized(target), normalized(row["messages"][-2]["content"]))

    def test_baseline_files_untouched(self):
        expected = {
            "baseline_smoke.json": "5505f2950d121d697b8a04f5f3eb3ce6b65dc0f56d84cd979f14c87a5153a60e",
            "baseline_validation.json": "bd4acc9b9a82804fcf0714d362dff5fc59ca95407330676c3fc260af7c07eb7f",
            "baseline_faiss_validation.json": "c9abb0c7a86a3aed804f39a5bd37e378c9d18bd34e567b04a64131b76e6f98a4",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
