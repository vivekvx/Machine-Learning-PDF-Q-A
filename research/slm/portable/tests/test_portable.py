"""Dataset, tokenizer, masking, retrieval, metric, and artifact integrity gates."""
import hashlib
import json
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from scoring import aggregate, audit_prediction, exact_abstention, token_f1
from training_data import tokenized_row


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


class PortableDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = HERE / "data/v1"
        cls.manifest = json.loads((cls.data / "manifest.json").read_text())
        cls.train = rows(cls.data / "train.jsonl")
        cls.valid = rows(cls.data / "validation_gold.jsonl")
        cls.tfidf = rows(cls.data / "validation_tfidf_faiss.jsonl")

    def test_hashes_counts_and_document_isolation(self):
        for name, digest in self.manifest["file_sha256"].items():
            self.assertEqual(sha(self.data / name), digest)
        self.assertEqual(len(self.train), 352)
        self.assertEqual(len(self.valid), 80)
        self.assertEqual(Counter(x["answerability"] for x in self.train),
                         {"answerable": 256, "unanswerable": 96})
        self.assertEqual(Counter(x["answerability"] for x in self.valid),
                         {"answerable": 40, "unanswerable": 40})
        train_docs = {x["document_id"] for x in self.train}
        valid_docs = {x["document_id"] for x in self.valid}
        test_docs = {x["document_id"] for x in rows(ROOT / "curated/test_documents.jsonl")}
        self.assertEqual((len(train_docs), len(valid_docs)), (352, 80))
        self.assertFalse(train_docs & valid_docs or train_docs & test_docs or valid_docs & test_docs)
        self.assertEqual({x["split"] for x in self.train}, {"train"})
        self.assertEqual({x["split"] for x in self.valid}, {"validation"})
        decisions = {(x["split"], x["document_id"]): x["status"]
                     for x in rows(ROOT / "curated/document_decisions.jsonl")}
        self.assertTrue(all(decisions[(x["split"], x["document_id"])] == "include"
                            for x in self.train + self.valid))

    def test_evidence_exact_and_no_truncation(self):
        for split, sample in (("train", self.train), ("validation", self.valid)):
            docs = {d["document_id"]: d for d in rows(ROOT / f"curated/{split}_documents.jsonl")}
            questions = {q["question_id"]: q for q in rows(ROOT / f"curated/{split}_questions.jsonl")}
            for row in sample:
                p = row["selected_passages"][0]
                doc = docs[row["document_id"]]
                section = next(s for s in doc["sections"] if s["section_index"] == p["section_index"])
                self.assertEqual(p["text"], section["paragraphs"][p["paragraph_index"]])
                self.assertEqual(hashlib.sha256(p["text"].encode()).hexdigest(), p["sha256"])
                self.assertLessEqual(row["full_tokens"], 512)
                self.assertLessEqual(row["prompt_tokens"] + 64, 512)
                self.assertEqual(row["provenance"]["source_split"], split)
                q = questions[row["question_id"]]
                self.assertEqual(row["answerability"], q["answerability"])
                if row["answerability"] == "answerable":
                    self.assertEqual(p["source_type"], "qasper_gold_source_paragraph")
                    self.assertEqual(set(p["evidence_ids"]),
                                     {e["evidence_id"] for e in q["evidence_passages"]})
                    self.assertIn(norm(row["reference_answer"]), norm(p["text"]))
                    self.assertEqual(row["target_answer"],
                                     f"{row['reference_answer']} [{p['passage_id']}]")
                else:
                    self.assertEqual(p["source_type"], "document_excerpt_not_gold")
                    self.assertEqual(row["target_answer"], "UNANSWERABLE")
                    self.assertEqual(q["evidence_passages"], [])

    def test_assistant_only_labels_and_cross_backend_chat_tokens(self):
        from transformers import AutoTokenizer

        mlx = AutoTokenizer.from_pretrained(ROOT / ".cache/qwen3-0.6b-4bit", local_files_only=True)
        base = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",
                                            revision="c1899de289a04d12100db370d81485cdf75e47ca",
                                            local_files_only=True)
        for row in self.train + self.valid:
            full, offset, labels = tokenized_row(row, mlx)
            base_full, base_offset, base_labels = tokenized_row(row, base)
            self.assertEqual((full, offset, labels), (base_full, base_offset, base_labels))
            self.assertEqual(labels[:offset], [-100] * offset)
            self.assertTrue(all(x != -100 for x in labels[offset:]))
            self.assertEqual(len(labels), row["full_tokens"])

    def test_tfidf_condition_provenance_and_budget_audit(self):
        meta = json.loads((self.data / "validation_tfidf_faiss.manifest.json").read_text())
        self.assertEqual(meta["output_sha256"], sha(self.data / "validation_tfidf_faiss.jsonl"))
        self.assertEqual(meta["source_sha256"], sha(self.data / "validation_gold.jsonl"))
        self.assertEqual([r["question_id"] for r in self.tfidf],
                         [r["question_id"] for r in self.valid])
        self.assertEqual(len(self.tfidf), 80)
        docs = {d["document_id"]: d for d in rows(ROOT / "curated/validation_documents.jsonl")}
        skipped = 0
        recall = 0
        for row in self.tfidf:
            self.assertEqual(row["condition"], "tfidf_faiss")
            self.assertLessEqual(row["prompt_tokens"] + 64, 512)
            self.assertEqual(len(row["retrieval_candidates"]), 3)
            included = {p["passage_id"] for p in row["selected_passages"]}
            self.assertEqual(included,
                             {p["passage_id"] for p in row["retrieval_candidates"] if p["included"]})
            skipped += sum(not p["included"] for p in row["retrieval_candidates"])
            recall += row["gold_passage_retrieved"] is True
            for p in row["selected_passages"]:
                section = next(s for s in docs[row["document_id"]]["sections"]
                               if s["section_index"] == p["section_index"])
                self.assertEqual(p["text"], section["paragraphs"][p["paragraph_index"]])
        self.assertEqual(skipped, meta["candidate_passages_skipped_for_budget"])
        self.assertEqual(recall, meta["answerable_gold_recall_count"])
        retrieval_cfg = json.loads((HERE / "v1/retrieval.json").read_text())["conditions"]
        self.assertIsNone(retrieval_cfg["tfidf_faiss"]["embedding_model"])
        self.assertEqual(retrieval_cfg["minilm_faiss"]["embedding_model"],
                         "sentence-transformers/all-MiniLM-L6-v2")

    def test_scoring_separates_answer_quality_abstention_and_citation(self):
        row = next(r for r in self.valid if r["answerability"] == "answerable")
        memory = {"process_rss_bytes": 100, "model_peak_bytes": 200,
                  "system_available_bytes": 300, "system_swap_free_bytes": 400}
        exact = audit_prediction(row, row["target_answer"], 0.5, memory)
        self.assertEqual(exact["exact_match"], 1)
        self.assertEqual(exact["token_f1"], 1.0)
        self.assertTrue(exact["reference_citation_present"])
        self.assertTrue(exact["citation_is_source"])
        self.assertFalse(exact_abstention("The answer is unavailable"))
        self.assertTrue(exact_abstention(" UNANSWERABLE "))
        self.assertEqual(aggregate([exact])["answerable_count"], 1)
        self.assertEqual(token_f1("different", [row["reference_answer"]]), 0.0)
        page = audit_prediction(row, "Answer on page 4", 0.5, memory)
        self.assertTrue(page["invented_page_reference"])

    def test_backend_config_alignment_and_dependency_pins(self):
        experiment = json.loads((HERE / "v1/experiment.json").read_text())
        mlx = json.loads((HERE / "v1/mlx.json").read_text())
        gpu = json.loads((HERE / "v1/gpu_peft.json").read_text())
        self.assertEqual(experiment["maximum_sequence_tokens"], mlx["maximum_sequence_tokens"])
        self.assertEqual(experiment["maximum_sequence_tokens"], gpu["maximum_sequence_tokens"])
        self.assertEqual(mlx["lora_rank"], gpu["lora_rank"])
        self.assertEqual(mlx["lora_scale"], gpu["lora_alpha"] / gpu["lora_rank"])
        self.assertEqual(mlx["trainable_last_layers"], gpu["trainable_last_layers"])
        self.assertEqual({x.split(".")[-1] for x in mlx["target_modules"]},
                         set(gpu["target_modules"]))
        requirements = {name: version for name, version in
                        (line.strip().split("==") for line in (HERE / "requirements-gpu.txt").read_text().splitlines()
                         if line.strip() and not line.startswith("#"))}
        self.assertEqual(requirements, gpu["dependency_versions"])

    def test_training_entrypoints_default_to_preflight(self):
        for script, args in ((ROOT / "run_feasibility.py", []),
                             (HERE / "train.py", ["--backend", "mlx"])):
            result = subprocess.run([sys.executable, str(script), *args],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["mode"], "preflight_only")
        child = subprocess.run([sys.executable, str(ROOT / "training_child.py")],
                               capture_output=True, text=True, timeout=15)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("explicit guarded launch", child.stderr)

    def test_prior_results_and_five_step_records_unchanged(self):
        expected = {
            "baseline_smoke.json": "5505f2950d121d697b8a04f5f3eb3ce6b65dc0f56d84cd979f14c87a5153a60e",
            "baseline_validation.json": "bd4acc9b9a82804fcf0714d362dff5fc59ca95407330676c3fc260af7c07eb7f",
            "baseline_faiss_validation.json": "c9abb0c7a86a3aed804f39a5bd37e378c9d18bd34e567b04a64131b76e6f98a4",
            "training_feasibility_run.json": "2aec0d004e8cf8c85fe7374a43e1d94c6da27258eb9a2a2e9d87de28361eeeab",
            "training_steps.jsonl": "11debb023ed83d5911149f3185675b324a85c729153883a59c59fd90e5e9621d",
            "training_worker.log": "08d221f205b1b98ef49cc0669e5ed9fdf2a4078bc90a87db3b4d20b6fd6883af",
            "adapter_validation.json": "f68d942939e0b7eeec4e71f6fdc9d70664faa975790e111bbcea38ab2e1c4197",
            "feasibility_adapter/adapters.safetensors": "fe742fc4a562aa63163f6f6f79d59a2b0bf064b7485cc789b98a4220448810a3",
            "TRAINING_FEASIBILITY_REPORT.md": "65433020b3287eafa4aeb88ded249c1903bb7da6ff70cebb71204dac30595882",
        }
        for name, digest in expected.items():
            self.assertEqual(sha(ROOT / name), digest, name)


if __name__ == "__main__":
    unittest.main()
