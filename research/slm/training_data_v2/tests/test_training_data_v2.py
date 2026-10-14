"""Integrity and triage-rule tests; none claim semantic evidence verification."""
import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SLM = HERE.parent
sys.path.insert(0, str(HERE))
from build import rows, signals
from validate import validate


class TrainingDataV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = list(rows(SLM / "portable/data/v1/train.jsonl"))
        cls.questions = {r["question_id"]: r for r in rows(SLM / "curated/train_questions.jsonl")}
        cls.docs = {r["document_id"]: r for r in rows(SLM / "curated/train_documents.jsonl")}
        cls.audit = list(rows(HERE / "v1/selected_audit.jsonl"))

    def test_frozen_artifacts_and_blank_human_review(self):
        self.assertEqual(validate()["human_reviews_completed"], 0)

    def test_selected_answerable_evidence_is_source_linked(self):
        answerable = [r for r in self.audit if r["answerability"] == "answerable"]
        self.assertEqual(len(answerable), 256)
        self.assertTrue(all(r["selected_passage"]["source_match"] for r in answerable))
        self.assertTrue(all(r["selected_passage"]["evidence_ids"] for r in answerable))
        self.assertTrue(all(not r["categories"]["missing"] for r in self.audit))

    def test_missing_evidence_signal(self):
        row = copy.deepcopy(next(r for r in self.selected if r["answerability"] == "answerable"))
        row["selected_passages"][0]["text"] = ""
        check = signals(row, self.questions[row["question_id"]], self.docs[row["document_id"]])
        self.assertIn("missing_selected_passage", check["flags"])
        self.assertTrue(check["categories"]["missing"])

    def test_annotation_contradiction_is_review_signal(self):
        record = next(r for r in self.audit if r["question_id"] == "5679fabeadf680e35a4f7b092d39e8638dca6b4d")
        self.assertIn("possible_annotation_contradiction", record["flags"])
        self.assertEqual({a["yes_no"] for a in record["original_annotations"]}, {True, False})
        self.assertEqual(record["automatic_support_status"], "unverified_requires_human_review")

    def test_unanswerable_excerpt_is_not_gold(self):
        unanswerable = [r for r in self.audit if r["answerability"] == "unanswerable"]
        self.assertEqual(len(unanswerable), 96)
        self.assertTrue(all("unanswerable_excerpt_not_proof" in r["flags"] for r in unanswerable))
        self.assertTrue(all(r["selected_passage"]["source_type"] == "document_excerpt_not_gold" for r in unanswerable))

    def test_candidate_manifest_and_no_auto_addition(self):
        manifest = json.loads((HERE / "v1/manifest.json").read_text())
        self.assertEqual(manifest["additional_candidate_documents"], 122)
        self.assertEqual(manifest["all_train_question_rejections"]["full_context_exceeds_token_budget"], 15)
        self.assertEqual(len(list(rows(HERE / "v1/train_ids.jsonl"))), 352)
        self.assertFalse(manifest["training_executed"])


if __name__ == "__main__":
    unittest.main()
