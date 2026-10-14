"""Synthetic-only tests; do not score or alter previous baseline predictions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abstention_metrics import score, strict_marker, strict_output


class AbstentionMetricsTest(unittest.TestCase):
    def test_prose_refusal_is_semantic_not_exact(self):
        result = score([{"context_sufficient": False, "human_response_class": "abstain",
                         "raw_response": "I cannot answer from this evidence.",
                         "allowed_passage_ids": []}])
        self.assertEqual(result["semantic_abstention"]["counts"]["tp"], 1)
        self.assertEqual(result["exact_marker"]["counts"]["fn"], 1)
        self.assertEqual(result["strict_format_compliance"], 0.0)

    def test_exact_marker_requires_no_wrapper(self):
        self.assertTrue(strict_marker(" \nUNANSWERABLE\n"))
        self.assertFalse(strict_marker("Answer: UNANSWERABLE"))
        self.assertFalse(strict_marker("UNANSWERABLE because evidence is missing"))
        self.assertEqual(strict_output("Answer [section:2:paragraph:3]", {"section:2:paragraph:3"}), "cited_answer")
        self.assertEqual(strict_output("Answer [section:2:paragraph:3]", set()), "invalid")
        self.assertEqual(strict_output("Answer [section:2:paragraph:3] extra", {"section:2:paragraph:3"}), "invalid")
        self.assertEqual(strict_output("<think>test</think> UNANSWERABLE", set()), "invalid")

    def test_unjudged_records_rejected(self):
        with self.assertRaises(ValueError):
            score([{"context_sufficient": False, "raw_response": "UNANSWERABLE", "allowed_passage_ids": []}])
        with self.assertRaises(ValueError):
            score([{"human_response_class": "abstain", "raw_response": "UNANSWERABLE", "allowed_passage_ids": []}])
        with self.assertRaises(ValueError):
            score([{"context_sufficient": False, "human_response_class": "abstain", "raw_response": "UNANSWERABLE"}])

    def test_context_sufficiency_not_document_answerability(self):
        result = score([{"context_sufficient": False, "human_response_class": "abstain",
                         "raw_response": "UNANSWERABLE", "document_answerable": True,
                         "allowed_passage_ids": []},
                        {"context_sufficient": True, "human_response_class": "answer",
                         "raw_response": "42 [section:0:paragraph:0]", "document_answerable": True,
                         "allowed_passage_ids": ["section:0:paragraph:0"]}])
        self.assertEqual(result["semantic_abstention"]["accuracy"], 1.0)
        self.assertEqual(result["exact_marker"]["accuracy"], 1.0)
        self.assertEqual(result["strict_format_compliance"], 1.0)


if __name__ == "__main__":
    unittest.main()
