"""Focused offline tests; none opens sealed records or downloads a model."""
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from research.slm.scratch.pdf_qa import app as service
from research.slm.scratch.pdf_qa.pipeline import validate_citation


def fixture():
    pdf = fitz.open()
    for lines in (
        ["For quartz node, the method is hash join.", "For birch relay, the method is range scan.",
         "For quartz node, the metric is 48 ms.", "For birch relay, the metric is 96 ms."],
        ["For cedar cache, the method is bloom filter.", "For elm index, the method is merge sort.",
         "For cedar cache, the metric is 72 ms.", "For elm index, the metric is 34 ms."],
    ):
        page = pdf.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + i * 30), line)
    return pdf.tobytes()


class PDFQATests(unittest.TestCase):
    def setUp(self):
        self.upload = service.upload_bytes(fixture(), "original.pdf")
        self.index = service.DOCUMENTS[self.upload["document_id"]]

    def test_page_provenance_offsets_and_deduplication(self):
        self.assertEqual(self.upload["pages"], 2)
        retrieved = self.index.retrieve_candidates("What metric is recorded for elm index?")
        candidates = retrieved["candidates"]
        self.assertEqual(len({(x["page_number"], x["value_start"], x["value_end"]) for x in candidates}), len(candidates))
        gold = next(x for x in candidates if x["value"] == "34 ms")
        self.assertEqual(gold["page_number"], 2)
        self.assertTrue(validate_citation(self.index, gold))
        self.assertEqual(self.index.pages[1].page_content[gold["value_start"]:gold["value_end"]], "34 ms")

    def test_question_rank_has_no_gold_answer_input(self):
        gold = self.index.retrieve_candidates("What method is used for quartz node?")["candidates"]
        wrong = self.index.retrieve_candidates("What method is used for birch relay?")["candidates"]
        self.assertEqual(gold[0]["value"], "hash join")
        self.assertEqual(wrong[0]["value"], "range scan")

    def test_insufficient_candidates_is_not_semantic_abstention(self):
        pdf = fitz.open(); page = pdf.new_page()
        page.insert_text((72, 72), "For quartz node, the method is hash join.")
        upload = service.upload_bytes(pdf.tobytes(), "one.pdf")
        result = service.ask(service.DOCUMENTS[upload["document_id"]], "What method is used for quartz node?")
        self.assertEqual(result["status"], "INSUFFICIENT_CANDIDATES")
        self.assertIsNone(result["neural_prediction"])

    def test_missing_fact_veto_and_obsolete_veto(self):
        candidates = self.index.retrieve_candidates("What method is used for quartz node?")["candidates"]
        selected = next(x for x in candidates if x["value"] == "hash join")
        self.assertEqual(service.evidence_decision(self.index, "What method is used for missing node?", selected, candidates)[0],
                         "INSUFFICIENT_EVIDENCE")
        self.assertEqual(service.evidence_decision(self.index, "What metric is recorded for quartz node?", selected, candidates)[0],
                         "INSUFFICIENT_EVIDENCE")
        pdf = fitz.open(); page = pdf.new_page()
        page.insert_text((72, 72), "For cobalt proxy, the method was linear scan; it is obsolete.\n"
                                    "For cobalt proxy, the current method is radix sort.\n"
                                    "For amber gateway, the metric is 19 ms.\n"
                                    "For ivory parser, the method is token merge.")
        upload = service.upload_bytes(pdf.tobytes(), "correction.pdf")
        index = service.DOCUMENTS[upload["document_id"]]
        ranked = index.retrieve_candidates("What is the current method for cobalt proxy?")["candidates"]
        old = next(x for x in ranked if x["value"] == "linear scan")
        self.assertEqual(service.evidence_decision(index, "What is the current method for cobalt proxy?", old, ranked)[0],
                         "INSUFFICIENT_EVIDENCE")

    def test_http_upload_and_error_handling_without_external_api(self):
        client = TestClient(service.app)
        response = client.post("/api/upload", files={"file": ("fixture.pdf", fixture(), "application/pdf")})
        self.assertEqual(response.status_code, 200)
        document_id = response.json()["document_id"]
        with patch.object(service, "pointer_request", side_effect=RuntimeError("guard refused")):
            ask = client.post("/api/ask", json={"document_id": document_id,
                                                   "question": "What method is used for quartz node?"})
        self.assertEqual(ask.status_code, 503)
        self.assertNotIn("groq", service.app.openapi().get("paths", {}))
        self.assertEqual(client.get("/").status_code, 200)

    def test_frozen_worker_hash_and_answer_length_contract(self):
        script = """import json,sys
sys.path.insert(0,'research/slm/scratch')
from pdf_qa.pointer_worker import choose_four,sha,SCRATCH
from pointer_qa.v2.train import tokenizer
from working_qa.v2.prepare import pair_inputs
selection=json.loads((SCRATCH/'pointer_qa/v2/validation/pointer_selection.json').read_text())
assert sha(SCRATCH/'pointer_qa/v2/runs/main/checkpoint-step-2000/model.safetensors') == selection['checkpoint_sha256']
assert sha(SCRATCH/'pointer_qa/v1/tokenizer.json') == selection['tokenizer_sha256']
tok=tokenizer(); p,c=pair_inputs(tok,'For amber gateway, the method is [ binary search ].','What method is used for amber gateway?')
assert c[0]['value']=='binary search' and p[0]['value_end']-p[0]['value_start']>8
print('frozen hashes and eight-token cap verified')"""
        python = Path(__file__).resolve().parents[1] / ".venv/bin/python"
        result = subprocess.run([str(python), "-c", script], text=True, capture_output=True,
                                cwd=Path(__file__).resolve().parents[4], timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sealed_attempt_cannot_repeat(self):
        # The sealed fixture is deliberately kept out of portable installs.
        if not (Path(__file__).parent / "benchmark/v1/sealed.jsonl").is_file():
            self.skipTest("sealed benchmark retained locally, not shipped with the service")
        from research.slm.scratch.pdf_qa.benchmark.evaluate import evaluate
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            evaluate("sealed")


if __name__ == "__main__":
    unittest.main()
