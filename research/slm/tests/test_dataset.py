import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_dataset", ROOT / "prepare_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DatasetTests(unittest.TestCase):
    def test_exclusion_scans_sections_and_annotations(self):
        doc = {"title": "NLP", "abstract": "", "full_text": {"paragraphs": [["patient data"]]},
               "qas": {"question": [], "answers": []}}
        self.assertEqual(module.exclusion_matches(doc), ["patient"])
        doc["full_text"]["paragraphs"] = [["language modeling"]]
        doc["qas"]["answers"] = [{"answer": [{"evidence": ["clinical trial"]}]}]
        self.assertEqual(module.exclusion_matches(doc), ["clinical"])
        doc["qas"]["answers"] = []
        doc["full_text"]["section_name"] = ["Biomedical methods"]
        self.assertEqual(module.exclusion_matches(doc), ["biomedical"])

    def test_annotations_and_mixed_labels_preserved(self):
        group = {"answer": [{"unanswerable": False, "evidence": ["paragraph"]},
                            {"unanswerable": True, "evidence": []}],
                 "annotation_id": ["a1", "a2"], "worker_id": ["w1", "w2"]}
        doc = {"id": "paper", "qas": {"question": ["Q?"], "question_id": ["q1"], "answers": [group]}}
        q = module.question_records(doc, "train")[0]
        self.assertEqual(q["answerability"], "mixed")
        self.assertEqual([a["annotation_id"] for a in q["annotations"]], ["a1", "a2"])

    def test_generated_pilot(self):
        if not (ROOT / "manifest.json").exists():
            self.skipTest("run prepare_dataset.py first")
        spec = importlib.util.spec_from_file_location("validate_dataset", ROOT / "validate_dataset.py")
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        validator.validate()


if __name__ == "__main__":
    unittest.main()
