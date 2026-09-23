"""Focused regression checks for the academic RAG improvements.

Run with: python3 -m unittest test_rag_improvements.py
"""

import unittest
from unittest.mock import patch

from src.chunking import TextChunk, _section_heading_at
from src.generator import _build_prompt, _generate_extractive
from src.query_intelligence import classify_and_expand, classify_intent
from src.retriever import retrieve


class FakeVectorStore:
    is_ready = True
    total_vectors = 3


class AcademicRAGTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            TextChunk(
                "Abstract\nThis review synthesizes methods and findings in retrieval augmented generation.",
                page_number=1,
                source="research-paper.pdf",
                chunk_index=0,
                metadata={"section_heading": "Abstract", "file_name": "research-paper.pdf"},
            ),
            TextChunk(
                "Methods\nThe authors compare chunk sizes and query expansion strategies.",
                page_number=3,
                source="research-paper.pdf",
                chunk_index=1,
                metadata={"section_heading": "Methods", "file_name": "research-paper.pdf"},
            ),
            TextChunk(
                "Results\nSection-aware retrieval improves grounded synthesis across the paper.",
                page_number=5,
                source="research-paper.pdf",
                chunk_index=2,
                metadata={"section_heading": "Results", "file_name": "research-paper.pdf"},
            ),
        ]

    def test_requested_query_taxonomy(self):
        expected = {
            "Summarize this document.": "broad_summary",
            "What is the methodology?": "methodology",
            "What are the key findings?": "findings",
            "What metrics or results are reported?": "metrics",
            "Compare the approaches.": "comparison",
            "What information is missing?": "unknown",
            "Who is the corresponding author?": "specific_fact",
        }
        for question, query_type in expected.items():
            with self.subTest(question=question):
                self.assertEqual(classify_intent(question), query_type)

    def test_broad_summary_expands_and_retrieves_more(self):
        info = classify_and_expand("Summarize this document.", index_size=30)
        self.assertEqual(info.intent, "broad_summary")
        self.assertGreater(info.recommended_top_k, 6)
        self.assertGreater(len(info.expanded_queries), 1)

    def test_prompt_contains_real_chunk_text_and_page_labels(self):
        system, prompt = _build_prompt(
            "Summarize this document.", self.chunks, [0.8, 0.7, 0.6], "broad_summary"
        )
        self.assertIn("academic PDF question-answering assistant", system)
        self.assertIn("Section-aware retrieval improves", prompt)
        self.assertIn("Page 5", prompt)
        self.assertIn("Query type:\nbroad_summary", prompt)

    def test_extractive_fallback_cites_pages(self):
        answer, model = _generate_extractive("What are the key findings?", self.chunks)
        self.assertEqual(model, "extractive-tfidf")
        self.assertRegex(answer, r"\(p\. \d+\)")

    def test_section_heading_tracks_research_paper_sections(self):
        text = "Abstract\nOverview\n\nMethods\nProcedure\n\nConclusion\nImplications"
        self.assertEqual(_section_heading_at(text, text.index("Abstract")), "Abstract")
        self.assertEqual(_section_heading_at(text, text.index("Methods")), "Methods")
        self.assertEqual(_section_heading_at(text, text.index("Conclusion")), "Conclusion")

    def test_broad_retrieval_keeps_low_similarity_context_and_expands_structure(self):
        observed_queries = []

        def fake_search(query, _store, _top_k):
            observed_queries.append(query)
            return [(self.chunks[0], 0.12), (self.chunks[1], 0.11), (self.chunks[2], 0.10)]

        with patch("src.retriever._search_single", side_effect=fake_search):
            result = retrieve(
                "Summarize this document.",
                FakeVectorStore(),
                top_k=3,
                expanded_queries=["Summarize this document."],
                is_broad=True,
                query_type="broad_summary",
            )

        self.assertEqual(len(result.chunks), 3)
        self.assertTrue(result.fallback_triggered)
        self.assertIn("abstract research purpose overview", observed_queries)
        self.assertIn("conclusion discussion implications summary", observed_queries)


if __name__ == "__main__":
    unittest.main()
