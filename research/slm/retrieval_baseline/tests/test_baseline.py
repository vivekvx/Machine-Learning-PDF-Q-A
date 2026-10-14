"""Focused checks for strict parsing, retrieval denominator, and source isolation."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SLM = HERE.parent
sys.path.insert(0, str(HERE))

from response_format import parse_response
from run_retrieval import retrieval_metrics


class ResponseFormatTests(unittest.TestCase):
    def test_exact_abstention_only(self):
        self.assertEqual(parse_response(" UNANSWERABLE\n", "stop")["format_status"], "exact_abstention")
        for malformed in ("unanswerable", "UNANSWERABLE.", "UNANSWERABLE because no evidence"):
            self.assertNotEqual(parse_response(malformed, "stop")["format_status"], "exact_abstention")

    def test_citation_and_unexpected_formats(self):
        self.assertEqual(parse_response("The method uses BERT [section:2:paragraph:3]", "stop")["format_status"],
                         "cited_answer")
        self.assertEqual(parse_response("The method uses BERT", "stop")["format_status"], "uncited_answer")
        self.assertEqual(parse_response("[section:2:paragraph:3] answer", "stop")["format_status"],
                         "malformed_citation")
        self.assertEqual(parse_response("<think>reasoning</think> UNANSWERABLE", "stop")["format_status"],
                         "control_or_thinking_token")
        self.assertEqual(parse_response("", "stop")["format_status"], "empty_generation")

    def test_truncation_is_independent_of_format(self):
        parsed = parse_response("partial answer", "length")
        self.assertTrue(parsed["generation_truncated"])
        self.assertEqual(parsed["format_status"], "uncited_answer")
        self.assertEqual(parse_response("UNANSWERABLE", None)["format_status"], "unknown_generation_stop")


class RetrievalMetricTests(unittest.TestCase):
    def test_answerable_only_denominator_and_packed_coverage(self):
        rows = [
            {"answerability": "answerable", "reference_citation": "section:0:paragraph:0",
             "retrieval_candidates": [{"passage_id": "section:0:paragraph:1", "included": True},
                                      {"passage_id": "section:0:paragraph:0", "included": False}],
             "gold_passage_retrieved": False, "selected_passages": [{}],
             "retrieval_latency_seconds": 0.1, "ranking_latency_seconds": 0.08},
            {"answerability": "unanswerable", "reference_citation": None,
             "retrieval_candidates": [{"passage_id": "section:0:paragraph:1", "included": True}],
             "gold_passage_retrieved": None, "selected_passages": [{}],
             "retrieval_latency_seconds": 0.2, "ranking_latency_seconds": 0.18},
        ]
        m = retrieval_metrics(rows)
        self.assertEqual(m["answerable_gold_questions"], 1)
        self.assertEqual(m["recall_at_1"], 0)
        self.assertEqual(m["recall_at_3"], 1)
        self.assertEqual(m["mrr_at_3"], 0.5)
        self.assertEqual(m["packed_evidence_coverage"], 0)

    def test_fixed_data_and_config_hash(self):
        import hashlib
        config = json.loads((HERE / "v1/config.json").read_text())
        data = SLM / config["validation_source"]
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(),
                         config["validation_source_sha256"])
        source = [json.loads(line) for line in data.read_text().splitlines()]
        self.assertEqual(len(source), 80)
        self.assertEqual(len({r["document_id"] for r in source}), 80)
        self.assertEqual({r["split"] for r in source}, {"validation"})

    def test_frozen_outputs_and_spotcheck(self):
        from validate import validate
        manifest = json.loads((HERE / "v1/artifact_manifest.json").read_text())
        self.assertEqual(validate(), manifest)
        spot = [json.loads(line) for line in (HERE / "v1/support_spotcheck.jsonl").read_text().splitlines()]
        self.assertEqual(len(spot), 18)
        for condition in ("gold", "tfidf_faiss", "minilm_faiss"):
            preds = json.loads((HERE / f"v1/predictions_{condition}.json").read_text())
            self.assertEqual(preds["status"], "complete")
            ids = {r["question_id"] for r in preds["predictions"]}
            chosen = [r for r in spot if r["condition"] == condition]
            self.assertEqual(len(chosen), 6)
            self.assertEqual(sum(r["label"] == "answerable" for r in chosen), 3)
            self.assertTrue(all(r["question_id"] in ids and r["reason"] for r in chosen))


if __name__ == "__main__":
    unittest.main()
