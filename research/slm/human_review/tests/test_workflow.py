"""Synthetic decisions are created only in temporary directories, never in saved ledgers."""
import gzip
import hashlib
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from acl_metadata import ACL_PUBLISHER, acl_license, load_selected, resolve
from cohort import construct, write_snapshot
from review_core import SAMPLE, append_review, latest_reviews, load_sample, review_events, sha256, summarize
from rights_core import LEDGER, append_rights, read_jsonl
from server import Handler, ThreadingHTTPServer


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reviews = self.root / "reviews.jsonl"
        self.rights = self.root / "rights.jsonl"
        self.sample = load_sample()
        self.ledger = {r["document_id"]: r for r in read_jsonl(LEDGER)}

    def evidence(self, row, reviewer, *, judgment=None, abstention=None, full_document="Yes"):
        answerable = row["answerability"] == "answerable"
        return {"question_id": row["question_id"], "reviewer_id": reviewer,
                "judgment": judgment or ("Supported" if answerable else "Unsupported"),
                "abstention_appropriate": abstention or ("No" if answerable else "Yes"),
                "document_answerability_consistent": "Yes", "full_document_reviewed": full_document,
                "medical_domain": "No", "notes": "Synthetic fixture only; not a human decision"}

    def rights_event(self, row, reviewer="rights_test"):
        return {"document_id": row["document_id"], "reviewer_id": reviewer,
                "decision": "permissive_verified", "source_version_matches": "Yes",
                "third_party_material_cleared": "Yes", "training_and_redistribution_allowed": "Yes",
                "license_evidence_url": "https://aclanthology.org/faq/copyright/",
                "attribution": "Synthetic test attribution", "notes": "Synthetic fixture only; not rights clearance"}

    def test_review_persistence_and_corrections(self):
        row = self.sample[0]
        first = append_review(self.evidence(row, "reviewer_one"), decisions_path=self.reviews)
        correction = self.evidence(row, "reviewer_one", judgment="Needs Review")
        append_review(correction, decisions_path=self.reviews)
        events = review_events(self.reviews)
        self.assertEqual(len(events), 2)
        self.assertNotEqual(first["event_id"], events[1]["event_id"])
        latest = latest_reviews(events, {r["question_id"] for r in self.sample}, sha256(SAMPLE))
        self.assertEqual(latest[(row["question_id"], "reviewer_one")]["judgment"], "Needs Review")
        self.assertEqual(self.sample[0]["human_support_judgment"], "")

    def test_review_validation_and_agreement(self):
        row = self.sample[0]
        with self.assertRaises(ValueError):
            append_review(self.evidence(row, "bad id"), decisions_path=self.reviews)
        append_review(self.evidence(row, "reviewer_one"), decisions_path=self.reviews)
        append_review(self.evidence(row, "reviewer_two"), decisions_path=self.reviews)
        summary = summarize(decisions_path=self.reviews)
        self.assertEqual(summary["reviewed_question_count"], 1)
        self.assertEqual(summary["agreement"][0]["support"]["raw_agreement"], 1.0)
        self.assertIn(row["question_id"], summary["agreed_question_ids"])

    def test_acl_exact_match_and_license_scope(self):
        row = load_selected()[0]
        bib = self.root / "small.bib.gz"
        block = f'''@inproceedings{{test-acl,\n    title = "{row['title']}",\n    author = "Test, Alice",\n    year = "2019",\n    publisher = "{ACL_PUBLISHER}",\n    url = "https://aclanthology.org/D19-1581/"\n}}\n'''
        with gzip.open(bib, "wt", encoding="utf-8") as stream:
            stream.write(block)
        matched = resolve([row], bib)[0]
        self.assertEqual(matched["match_method"], "unique_normalized_exact_title")
        self.assertEqual(matched["acl_id"], "D19-1581")
        self.assertEqual(matched["acl_publication_license_class"], "verified_permissive_acl_publication")
        self.assertTrue(matched["qasper_text_training_rights"].startswith("unresolved"))
        third_party = {"publisher": "Third Party", "year": "2019"}
        self.assertEqual(acl_license(third_party)[0], "unresolved")
        restrictive = {"publisher": ACL_PUBLISHER, "year": "2015"}
        self.assertEqual(acl_license(restrictive)[0], "restrictive_acl_publication")

    def test_rights_policy_blocks_uncleared_and_third_party(self):
        good = next(r for r in self.sample if self.ledger[r["document_id"]]["acl_publication_license_class"] == "verified_permissive_acl_publication")
        bad = self.rights_event(good)
        bad["source_version_matches"] = "Uncertain"
        with self.assertRaises(ValueError):
            append_rights(bad, decisions_path=self.rights)
        third_party = next(r for r in self.sample if self.ledger[r["document_id"]]["acl_publication_license_class"] == "unresolved")
        with self.assertRaises(ValueError):
            append_rights(self.rights_event(third_party), decisions_path=self.rights)

    def test_reproducible_cohort_and_no_auto_inclusion(self):
        candidates, decisions = construct(review_path=self.reviews, rights_path=self.rights)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(len(decisions), 352)
        self.assertEqual(sum("not_in_64_question_review_sample" in x["reasons"] for x in decisions), 288)
        self.assertTrue(all("individual_source_rights_not_cleared" in x["reasons"] for x in decisions))
        row = next(r for r in self.sample if self.ledger[r["document_id"]]["acl_publication_license_class"] == "verified_permissive_acl_publication" and
                   "possible_annotation_contradiction" not in json.loads(r["automatic_flags_json"]))
        for reviewer in ("reviewer_one", "reviewer_two"):
            append_review(self.evidence(row, reviewer), decisions_path=self.reviews)
        self.assertEqual(len(construct(review_path=self.reviews, rights_path=self.rights)[0]), 0)
        append_rights(self.rights_event(row), decisions_path=self.rights)
        one, _ = construct(review_path=self.reviews, rights_path=self.rights)
        self.assertEqual([x["question_id"] for x in one], [row["question_id"]])
        self.assertEqual(one, construct(review_path=self.reviews, rights_path=self.rights)[0])
        manifest = write_snapshot(self.root / "snapshot", review_path=self.reviews, rights_path=self.rights)
        self.assertEqual(manifest["candidate_count"], 1)
        with self.assertRaises(FileExistsError):
            write_snapshot(self.root / "snapshot", review_path=self.reviews, rights_path=self.rights)

    def test_unanswerable_requires_full_document_and_no_medical_concern(self):
        row = next(r for r in self.sample if r["answerability"] == "unanswerable" and
                   self.ledger[r["document_id"]]["acl_publication_license_class"] == "verified_permissive_acl_publication")
        for reviewer in ("reviewer_one", "reviewer_two"):
            append_review(self.evidence(row, reviewer, full_document="No"), decisions_path=self.reviews)
        append_rights(self.rights_event(row), decisions_path=self.rights)
        self.assertEqual(construct(review_path=self.reviews, rights_path=self.rights)[0], [])
        for reviewer in ("reviewer_one", "reviewer_two"):
            event = self.evidence(row, reviewer)
            event["medical_domain"] = "Uncertain" if reviewer == "reviewer_two" else "No"
            append_review(event, decisions_path=self.reviews)
        self.assertEqual(construct(review_path=self.reviews, rights_path=self.rights)[0], [])
        append_review(self.evidence(row, "reviewer_two"), decisions_path=self.reviews)
        self.assertEqual(len(construct(review_path=self.reviews, rights_path=self.rights)[0]), 1)

    def test_local_http_interface_saves_separate_event(self):
        class TestHandler(Handler):
            sample_path = SAMPLE
            decisions_path = self.reviews
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse(); body = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn(self.sample[0]["question"], body)
        self.assertIn("Original annotation", body)
        form = self.evidence(self.sample[0], "reviewer_one")
        form["i"] = "0"
        form["csrf_token"] = TestHandler.csrf_token
        conn.request("POST", "/review", body=urlencode(form), headers={"Content-Type": "application/x-www-form-urlencoded"})
        response = conn.getresponse(); response.read()
        self.assertEqual(response.status, 303)
        self.assertEqual(len(review_events(self.reviews)), 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
