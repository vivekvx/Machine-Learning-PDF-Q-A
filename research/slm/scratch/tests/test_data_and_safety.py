import json
import tempfile
import unittest
from pathlib import Path

from checkpoint import digest
from data import DATA, load_data, windows
from prepare_data import CONFIG, build
from run import guard_reasons
from tokenizer import ByteBPE, SPECIAL


class DataAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_train_validation_isolation_and_vocab(self):
        tokenizer, records, manifest = load_data(self.config)
        self.assertEqual(tokenizer.vocab_size, 2000)
        self.assertEqual((len(records["train"]), len(records["validation"])), (96, 24))
        self.assertFalse({r["record_id"] for r in records["train"]} &
                         {r["record_id"] for r in records["validation"]})
        self.assertFalse({r["text"] for r in records["train"]} &
                         {r["text"] for r in records["validation"]})
        self.assertEqual(manifest["tokenizer_trained_on"], "train.jsonl only")
        self.assertEqual(len(windows(records["train"], tokenizer, 256)), 4)
        self.assertEqual(len(windows(records["validation"], tokenizer, 256)), 2)

    def test_tokenizer_round_trip(self):
        tokenizer = ByteBPE.load(DATA / "tokenizer.json")
        for text in ("Record 0001. The blue robot moves east.\n", "Unicode: café ☕ 日本語"):
            ids = tokenizer.encode(text, bos=True, eos=True)
            self.assertEqual(ids[0], SPECIAL["bos"])
            self.assertEqual(ids[-1], SPECIAL["eos"])
            self.assertEqual(tokenizer.decode(ids), text)

    def test_data_rebuild_is_byte_identical(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = build(Path(tmp) / "v1")
            self.assertEqual(rebuilt, manifest)
            for name, expected in manifest["file_sha256"].items():
                self.assertEqual(digest(Path(tmp) / "v1" / name), expected)

    def test_preflight_and_stop_thresholds(self):
        safety = self.config["safety"]
        before = {"available_bytes": 2_000_000_000, "swap_used_bytes": 1_000_000_000,
                  "swap_free_bytes": 1_000_000_000, "swap_out_total_bytes": 2_000_000_000,
                  "disk_free_bytes": 20_000_000_000,
                  "memory_pressure_free_percent": 30, "process_rss_bytes": 100_000_000}
        self.assertEqual(guard_reasons(before, before, safety, preflight=True), [])
        # vm.swapusage free is unused room in already allocated swap files,
        # not a fixed macOS swap limit. It remains observable, not a gate.
        low_allocated_swap = {**before, "swap_free_bytes": 0}
        self.assertEqual(guard_reasons(low_allocated_swap, before, safety, preflight=True), [])
        self.assertEqual(guard_reasons(low_allocated_swap, before, safety), [])
        growing = {**before, "swap_used_bytes": 1_700_000_000}
        self.assertIn("swap growth above stop ceiling", guard_reasons(growing, before, safety))
        swapping = {**before, "swap_out_total_bytes": 2_700_000_000}
        self.assertIn("swap-out activity above preflight ceiling",
                      guard_reasons(swapping, before, safety, preflight=True))
        self.assertIn("swap-out activity above stop ceiling",
                      guard_reasons(swapping, before, safety))
        disk_low = {**before, "disk_free_bytes": 500_000_000}
        self.assertIn("disk headroom below preflight floor",
                      guard_reasons(disk_low, before, safety, preflight=True))
        self.assertIn("disk headroom below stop floor", guard_reasons(disk_low, before, safety))
        pressure_low = {**before, "memory_pressure_free_percent": 5}
        self.assertIn("memory pressure unavailable or below preflight floor",
                      guard_reasons(pressure_low, before, safety, preflight=True))
        self.assertIn("memory pressure unavailable or below stop floor",
                      guard_reasons(pressure_low, before, safety))


if __name__ == "__main__":
    unittest.main()
