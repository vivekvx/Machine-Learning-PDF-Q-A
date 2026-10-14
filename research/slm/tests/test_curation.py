import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from curate_dataset import classify_document, resolve_question, duplicate_pairs  # noqa: E402
from metrics import exact_match, token_f1, is_abstention  # noqa: E402
from run_baseline import prompt_for  # noqa: E402


class CurationTests(unittest.TestCase):
    def test_filter_never_uses_heldout_answers_or_body(self):
        d = {"title": "Machine translation", "abstract": "We evaluate multilingual models.",
             "qas": {"answers": [{"answer": [{"free_form_answer": "clinical"}]}]},
             "full_text": {"paragraphs": [["medical"]]}}
        self.assertEqual(classify_document(d)[0], "include")
        d["abstract"] = "We study patients in clinical records."
        self.assertEqual(classify_document(d)[0], "exclude")
        d["abstract"] = "We mention clinical applications."
        self.assertEqual(classify_document(d)[0], "quarantine")

    def test_conflict_and_missing_evidence_quarantine(self):
        q = {"annotations": [{"unanswerable": True}, {"unanswerable": False, "free_form_answer": "x", "evidence": ["p"]}]}
        self.assertEqual(resolve_question(q)[1], ["conflicting_answerability"])
        q["annotations"] = [{"unanswerable": False, "free_form_answer": "x", "evidence": []}]
        self.assertEqual(resolve_question(q)[1], ["missing_evidence"])

    def test_near_duplicate_cross_split(self):
        common = {"title_grams": {"abcde", "bcdef"}, "abstract_words": {"one", "two"}, "body_sha256": "x"}
        a = {**common, "split": "train", "document_id": "a"}
        b = {**common, "split": "validation", "document_id": "b", "body_sha256": "y"}
        self.assertEqual(duplicate_pairs([a, b])[0]["reason"], "near_duplicate_title_abstract")

    def test_metrics(self):
        self.assertEqual(exact_match("The answer!", ["answer"]), 0)
        self.assertAlmostEqual(token_f1("red blue", ["blue red"]), 1)
        self.assertTrue(is_abstention("UNANSWERABLE"))

    def test_prompt_stays_within_sequence_budget(self):
        class StubTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return " ".join(m["content"] for m in messages)
            def encode(self, text):
                return text.split()
        row = {"question_id": "q", "document_id": "d", "title": "Paper", "question": "What?",
               "selected_passages": [{"section_index": 0, "evidence_id": "e", "text": "word " * 400}]}
        _, count, used = prompt_for(row, StubTokenizer(), 256, 32)
        self.assertLessEqual(count + 32, 256)
        self.assertLess(len(used[0]["text"]), 2000)


if __name__ == "__main__":
    unittest.main()
