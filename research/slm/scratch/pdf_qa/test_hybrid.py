"""Focused hybrid contract tests. Qwen weights and sealed examples are never loaded."""
import json
import subprocess
import unittest
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from research.slm.scratch.pdf_qa import app as service
from research.slm.scratch.pdf_qa import hybrid_service
from research.slm.scratch.pdf_qa.hybrid import classify_intent, plan, validate_citations
from research.slm.scratch.pdf_qa.qwen_worker import (relocate_leading_summary_citations,
    source_text, summary_format_issues)
from research.slm.scratch.pdf_qa.support_audit import audit as audit_support
from src.pdf_loader import PageDocument
from research.slm.scratch.pdf_qa.pipeline import PDFIndex


def fixture():
    pdf = fitz.open()
    pages = (
        "Introduction\nThe Quartz paper studies low latency retrieval. The baseline is lexical search.\n"
        "For quartz node, the method is hash join. For birch relay, the method is range scan.",
        "Methods\nThe system uses a two stage retrieval procedure. For cedar cache, the method is bloom filter.\n"
        "For elm index, the method is merge sort.",
        "Results\nFor quartz node, the metric is 48 ms. For birch relay, the metric is 96 ms.\n"
        "The proposed system reduced latency compared with the lexical baseline.",
        "Conclusion\nThe authors report that evaluation covered small datasets only. They identify scaling as a limitation."
    )
    for page_text in pages:
        page = pdf.new_page()
        for i, line in enumerate(page_text.split("\n")):
            page.insert_text((72, 72 + i * 22), line)
    return pdf.tobytes()


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)
        self.upload = self.client.post("/api/upload", files={"file": ("fixture.pdf", fixture(), "application/pdf")}).json()
        self.index = service.DOCUMENTS[self.upload["document_id"]]

    def test_intent_routing(self):
        questions = {
            "Give me a summary of this paper": "whole_paper_summary",
            "Summarize the methods": "section_summary",
            "What is the difference between the methods?": "comparison",
            "Explain the methodology": "explanation",
            "what metric is recorded for quartz node?": "factual",
            "Tell me a joke": "unsupported",
            "What problem are the researchers trying to solve?": "section_summary",
            "What were the main experimental results?": "section_summary",
        }
        for question, intent in questions.items():
            self.assertEqual(classify_intent(question), intent)

    def test_section_coverage_and_exact_page_offsets(self):
        chosen = plan(self.index, "Give me a summary of this paper")
        self.assertEqual(chosen["intent"], "whole_paper_summary")
        self.assertGreaterEqual(chosen["coverage"]["pages_sampled"], 3)
        for source in chosen["sources"]:
            page = next(p for p in self.index.pages if p.page_number == source["page_number"])
            self.assertEqual(page.page_content[source["page_start"]:source["page_end"]], source["text"])
            self.assertEqual(source["document_id"], self.index.document_id)

    def test_long_paper_overview_ignores_references_and_appendix(self):
        intro = "The study asks how document retrieval can stay accurate under noisy input. " * 5
        method = "The method groups source spans, reranks passages, and checks citation identifiers. " * 5
        results = "The main result is a measured reduction in retrieval errors on a local sample. " * 5
        conclusion = "The authors conclude that the approach needs evaluation on more diverse documents. " * 5
        pages = [
            PageDocument("Technical paper\n" + intro + "\n1\nIntroduction\n" + intro, 1, "sample.pdf"),
            PageDocument("2\nMethod\n" + method, 2, "sample.pdf"),
            PageDocument("3\nMain Results\n" + results, 3, "sample.pdf"),
            PageDocument("4\nConclusion\n" + conclusion, 4, "sample.pdf"),
            PageDocument("References\n[1] " + "irrelevant citations and paper titles. " * 20, 5, "sample.pdf"),
            PageDocument("Appendix Contents\n" + "table of contents and numbers. " * 20, 6, "sample.pdf"),
        ]
        index = PDFIndex.from_pages(b"original-synthetic-test-only", "sample.pdf", pages)
        summary = plan(index, "Give me a summary of this paper")
        self.assertEqual([s["page_number"] for s in summary["sources"]], [1, 2, 3, 4])
        self.assertEqual(summary["coverage"]["back_matter_pages_excluded"], 2)
        results_only = plan(index, "What were the main experimental results?")
        self.assertEqual([s["page_number"] for s in results_only["sources"]], [3])
        for source in summary["sources"]:
            page = pages[source["page_number"] - 1]
            self.assertEqual(page.page_content[source["page_start"]:source["page_end"]], source["text"])

    def test_summary_avoids_table_heading_and_uses_result_prose(self):
        pages = [
            PageDocument("Opening abstract. " + "The paper studies retrieval errors in a local system. " * 10,
                         1, "sample.pdf"),
            PageDocument("Our method reconstructs dependencies and audits handoffs between agents.\n" * 12,
                         2, "sample.pdf"),
            PageDocument("Method\nAgent Step Agent Step Agent\n50.0\n24.1\n64.3\n80.4\n76.0\n" * 5,
                         3, "sample.pdf"),
            PageDocument("Results\n" + "The proposed system achieves higher attribution accuracy than the lexical baseline.\n" * 10,
                         4, "sample.pdf"),
            PageDocument("Conclusion\n" + "The authors conclude that a larger sample is needed. " * 10,
                         5, "sample.pdf"),
        ]
        index = PDFIndex.from_pages(b"synthetic-table-fixture", "sample.pdf", pages)
        sources = plan(index, "Give me a summary of this paper")["sources"]
        self.assertEqual([s["page_number"] for s in sources], [1, 2, 4, 5])
        self.assertNotIn("Agent Step Agent", " ".join(s["text"] for s in sources))
        for source in sources:
            page = pages[source["page_number"] - 1]
            self.assertEqual(page.page_content[source["page_start"]:source["page_end"]], source["text"])

    def test_summary_prefers_outcome_over_benchmark_setup(self):
        pages = [
            PageDocument("Introduction\n" + "The paper asks how local retrieval can improve reliability.\n" * 12,
                         1, "sample.pdf"),
            PageDocument("Methods\n" + "Our method audits source use before a final decision is made.\n" * 12,
                         2, "sample.pdf"),
            PageDocument("Results\nBenchmark and baselines. Our test set contains many prompts. "
                         "We compare several systems under two settings.\n" * 12, 3, "sample.pdf"),
            PageDocument("Main observations. Our system improves measured accuracy from 51 to 83 "
                         "on the held-out development set. This result exceeds the lexical baseline.\n" * 8,
                         4, "sample.pdf"),
            PageDocument("Conclusion\n" + "The authors conclude that broader evaluation is still needed.\n" * 12,
                         5, "sample.pdf"),
        ]
        index = PDFIndex.from_pages(b"synthetic-outcome-fixture", "sample.pdf", pages)
        result = next(s for s in plan(index, "Give me a summary of this paper")["sources"]
                      if s["section"] == "Results")
        self.assertEqual(result["page_number"], 4)
        self.assertIn("improves measured accuracy", result["text"])
        self.assertEqual(pages[3].page_content[result["page_start"]:result["page_end"]], result["text"])

    def test_summary_format_rejects_headings_and_citation_lists(self):
        good = "- The system checks how earlier evidence is used by later agents. [E2]\n" \
               "- The method improves attribution accuracy on the reported benchmark. [E3]"
        bad = "- [E2] Agent Step Agent Step Agent\n- [E3] Experimental Setup\n**[E2]**"
        self.assertEqual(summary_format_issues(good), [])
        self.assertIn("bullet_is_heading_or_fragment", summary_format_issues(bad))
        self.assertIn("standalone_citation_list", summary_format_issues(bad))

    def test_citation_relocation_keeps_raw_claim_and_source_id(self):
        raw = "- [E1] The study examines errors across cooperating software agents.\n" \
              "- The result improved measured accuracy on the sample. [E3]"
        shown, changed = relocate_leading_summary_citations(raw)
        self.assertTrue(changed)
        self.assertEqual(shown, "- The study examines errors across cooperating software agents. [E1]\n"
                         "- The result improved measured accuracy on the sample. [E3]")
        self.assertEqual(summary_format_issues(shown), [])

    def test_v14_keeps_runtime_stops_and_requires_measured_start_headroom(self):
        old = json.loads((hybrid_service.HERE / "hybrid_config.v9.json").read_text())
        new = hybrid_service.CONFIG
        for key in ("min_preflight_pressure_free_percent",
                    "stop_available_bytes", "stop_pressure_free_percent", "stop_swap_growth_bytes",
                    "min_disk_free_bytes", "stop_process_rss_bytes", "worker_timeout_seconds"):
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["min_preflight_available_bytes"], 2_000_000_000)
        self.assertEqual(new["mlx_free_cache_limit_bytes"], 64 * 1024 * 1024)
        self.assertEqual(new["prefill_step_size"], 128)

    def test_installed_mlx_lm_supports_bounded_prefill(self):
        script = ("import inspect; from mlx_lm.generate import generate_step; "
                  "assert 'prefill_step_size' in inspect.signature(generate_step).parameters; "
                  "print('bounded prefill supported')")
        result = subprocess.run([str(hybrid_service.QWEN_PYTHON), "-c", script],
                                capture_output=True, text=True, cwd=hybrid_service.ROOT, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bounded prefill supported", result.stdout)

    def test_prompt_sources_expose_section_role_without_changing_evidence_id(self):
        rendered = source_text([{"id": "E2", "section": "Method", "text": "The system audits handoffs."}])
        self.assertEqual(rendered, "[E2] (Method) The system audits handoffs.")

    def test_raw_memory_upload_and_session_removal(self):
        uploaded = self.client.post("/api/upload-memory", content=fixture(),
            headers={"content-type": "application/pdf", "x-pdf-filename": "local.pdf"})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        document_id = uploaded.json()["document_id"]
        self.assertIn(document_id, service.DOCUMENTS)
        removed = self.client.delete(f"/api/documents/{document_id}")
        self.assertEqual(removed.status_code, 200)
        self.assertNotIn(document_id, service.DOCUMENTS)

    def test_preview_is_passages_not_an_answer_and_never_loads_model(self):
        with patch.object(hybrid_service, "preflight", side_effect=AssertionError("no model in preview")):
            response = self.client.post("/api/evidence-preview", json={
                "document_id": self.upload["document_id"],
                "question": "Give me a summary of this paper"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["label"], "source_passages_only_not_a_model_answer")
        self.assertNotIn("answer", result)
        self.assertGreaterEqual(len(result["sources"]), 3)

    def test_placeholder_summary_is_not_marked_as_a_finished_answer(self):
        fake = {"status": "complete", "answer": "claim [E1]. This is not a readable summary.",
                "citations": {"valid": True, "cited_ids": ["E1"], "unknown_ids": [], "has_citation": True},
                "model_load_ms": 1, "peak_mlx_bytes": 100, "process_rss_bytes": 100,
                "calls": [{"finish_reason": "stop"}], "watchdog": {}, "offline_network_blocked": True,
                "model_revision": "test", "model_sha256": "test", "prompt_version": "test",
                "final_raw_source_ids": ["E1"]}
        with patch.object(hybrid_service, "preflight", return_value={"ready": True, "errors": [], "resources": hybrid_service.snapshot()}), \
             patch.object(hybrid_service, "invoke_worker", return_value=fake), \
             patch.object(hybrid_service, "reasons", return_value=[]):
            response = self.client.post("/api/hybrid/ask", json={"document_id": self.upload["document_id"],
                "question": "Give me a summary of this paper", "mode": "hybrid"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "FORMAT_UNVERIFIED")

    def test_citation_verifier_rejects_fabricated_ids(self):
        sources = plan(self.index, "What metric is recorded for quartz node?")["sources"]
        self.assertTrue(validate_citations("48 ms [E1]", sources)["valid"])
        invalid = validate_citations("48 ms [E999]", sources)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["unknown_ids"], ["E999"])

    def test_lexical_support_warning_is_not_human_entailment(self):
        sources = [{"id": "E1", "text": "2to3's arguments are files or directories to transform."},
                   {"id": "E2", "text": "2to3 transforms Python 2.x source code into Python 3.x code."}]
        wrong = audit_support("2to3 transforms Python 2.x into Python 3.x code [E1]", sources)
        self.assertTrue(wrong["requires_manual_support_review"])
        correct = audit_support("2to3 transforms Python 2.x into Python 3.x code [E2]", sources)
        self.assertFalse(correct["requires_manual_support_review"])
        pooled = audit_support("2to3 transforms Python 2.x into Python 3.x code. Arguments are files or directories. [E1] [E2]", sources)
        self.assertTrue(any(x["reason"] == "pooled_citations_do_not_identify_per_sentence_support"
                            for x in pooled["warnings"]))

    def test_guard_fails_closed(self):
        snap = {"available_bytes": 650_000_000, "pressure_free_percent": 9,
                "disk_free_bytes": 500_000_000, "swap_used_bytes": 700_000_000,
                "swap_out_bytes": 700_000_000, "process_rss_bytes": 3_000_000_000}
        before = {**snap, "swap_used_bytes": 0, "swap_out_bytes": 0}
        self.assertGreaterEqual(len(hybrid_service.reasons(snap, before)), 5)

    def test_readiness_reports_resource_block_without_model_loading(self):
        snap = {"available_bytes": 1_780_000_000, "pressure_free_percent": 44,
                "disk_free_bytes": 49_000_000_000, "swap_used_bytes": 5_140_000_000,
                "swap_out_bytes": 0, "process_rss_bytes": None}
        with patch.object(hybrid_service, "snapshot", return_value=snap), \
             patch.object(hybrid_service, "preflight", side_effect=AssertionError("no model checks")):
            result = self.client.get("/api/model-readiness")
        self.assertEqual(result.status_code, 200)
        data = result.json()
        self.assertFalse(data["resource_ready_now"])
        self.assertEqual(data["available_bytes"], 1_780_000_000)
        self.assertEqual(data["required_available_bytes"], 2_000_000_000)
        self.assertTrue(any("available memory" in reason for reason in data["reasons"]))

    def test_http_modes_and_repeated_document_without_model_loading(self):
        calls = []
        def fake_worker(payload, before):
            calls.append(payload)
            return {"status": "complete", "answer": "The method is hash join [E1].",
                    "citations": {"valid": True, "cited_ids": ["E1"], "unknown_ids": [], "has_citation": True},
                    "model_load_ms": 1, "peak_mlx_bytes": 100, "process_rss_bytes": 100,
                    "calls": [{"finish_reason": "stop"}], "watchdog": {}, "offline_network_blocked": True,
                    "model_revision": "test", "model_sha256": "test", "prompt_version": "test",
                    "final_raw_source_ids": ["E1"]}
        request = {"document_id": self.upload["document_id"], "question": "What method is used for quartz node?"}
        with patch.object(hybrid_service, "preflight", return_value={"ready": True, "errors": [], "resources": hybrid_service.snapshot()}), \
             patch.object(hybrid_service, "invoke_worker", side_effect=fake_worker), \
             patch.object(hybrid_service, "reasons", return_value=[]):
            first = self.client.post("/api/hybrid/ask", json={**request, "mode": "qwen"})
            second = self.client.post("/api/hybrid/ask", json={**request, "mode": "qwen"})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(first.json()["pointer"]["used"], False)
        self.assertEqual(first.json()["sources"][0]["document_id"], self.upload["document_id"])
        bad = self.client.post("/api/hybrid/ask", json={**request, "mode": "cloud"})
        self.assertEqual(bad.status_code, 422)
        self.assertNotIn("groq", json.dumps(service.app.openapi()).lower())

    def test_summary_bypasses_pointer(self):
        with patch.object(hybrid_service, "preflight", return_value={"ready": False, "errors": ["memory"], "resources": {}}), \
             patch.object(service, "pointer_request", side_effect=AssertionError("pointer must be bypassed")):
            result = self.client.post("/api/hybrid/ask", json={"document_id": self.upload["document_id"],
                "question": "Give me a summary of this paper", "mode": "hybrid"})
        self.assertEqual(result.status_code, 503)
        self.assertIn("preflight", result.json()["detail"].lower())

    def test_corrupt_and_empty_pdf_are_rejected_cleanly(self):
        corrupt = self.client.post("/api/upload", files={"file": ("bad.pdf", b"%PDF-invalid", "application/pdf")})
        self.assertEqual(corrupt.status_code, 422)
        self.assertNotIn("/Users/", corrupt.text)
        blank = fitz.open(); blank.new_page()
        empty = self.client.post("/api/upload", files={"file": ("empty.pdf", blank.tobytes(), "application/pdf")})
        self.assertEqual(empty.status_code, 422)
        self.assertIn("OCR is not available", empty.json()["detail"])

    def test_isolated_worker_denies_network_connections(self):
        script = ("from research.slm.scratch.pdf_qa.qwen_worker import block_network; "
                  "import socket; block_network(); "
                  "\ntry: socket.create_connection(('example.com', 443))\n"
                  "except RuntimeError as error: assert 'disabled' in str(error); print('network blocked')\n"
                  "else: raise AssertionError('network connection was not blocked')")
        result = subprocess.run([str(hybrid_service.QWEN_PYTHON), "-c", script],
                                capture_output=True, text=True, cwd=hybrid_service.ROOT, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("network blocked", result.stdout)


if __name__ == "__main__":
    unittest.main()
