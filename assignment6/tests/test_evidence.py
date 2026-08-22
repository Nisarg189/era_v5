"""The evidence bundle, and the throughput reconstruction it depends on."""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdes import evidence as ev, perf as perf_mod                  # noqa: E402


def rec(seqs, seq_len, loss):
    return {"n_sequences": seqs, "sequence_length": seq_len,
            "loss_tokens_by_lane": {"web": loss}, "tokens_by_lane": {"web": seqs * seq_len}}


class EvidenceBundle(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.e = ev.Evidence(self.dir)

    def test_one_failure_fails_the_bundle(self):
        self.e.add("a", True, "f1", "ok")
        self.assertTrue(self.e.all_passed)
        self.e.add("b", False, "f2", "not ok")
        self.assertFalse(self.e.all_passed)

    def test_json_records_every_check_with_its_evidence(self):
        self.e.add("a", True, ["f1", "f2"], "compared x with y", {"x": 1, "y": 1})
        payload = self.e.write_json(os.path.join(self.dir, "evidence.json"))
        self.assertEqual(payload["n_checks"], 1)
        self.assertEqual(payload["n_failed"], 0)
        self.assertEqual(payload["checks"][0]["evidence"], ["f1", "f2"])
        self.assertEqual(payload["checks"][0]["values"], {"x": 1, "y": 1})
        with open(os.path.join(self.dir, "evidence.json")) as f:
            self.assertEqual(json.load(f)["schema"], "era-v5-session6-evidence/1")

    def test_markdown_has_a_row_for_every_check(self):
        for i in range(4):
            self.e.add(f"req {i}", i != 2, "f", "detail", {"i": i})
        path = self.e.write_md(os.path.join(self.dir, "evidence.md"), {"run_id": "t"})
        with open(path) as f:
            text = f.read()
        for i in range(4):
            self.assertIn(f"req {i}", text)
        self.assertIn("FAILURES PRESENT", text)
        self.assertEqual(text.count("| **PASS** |"), 3)
        self.assertEqual(text.count("| **FAIL** |"), 1)

    def test_a_pipe_in_a_detail_cannot_break_the_table(self):
        self.e.add("a", True, "f", "x | y")
        path = self.e.write_md(os.path.join(self.dir, "e.md"), {})
        with open(path) as f:
            row = [l for l in f if l.startswith("| a |")][0]
        self.assertIn("x \\| y", row, "the detail pipe must be escaped")
        self.assertEqual(row.replace("\\|", "").count("|"), 5,
                         "only the four cell separators may remain")


class ThroughputReconstruction(unittest.TestCase):
    def test_agreeing_counts_pass(self):
        records = [rec(4, 128, 500) for _ in range(10)]
        report = {"raw_token_positions": 4 * 128 * 10, "loss_bearing_tokens": 5000}
        self.assertTrue(perf_mod.reconstruct_check(report, records)["ok"])

    def test_an_inflated_claim_is_caught(self):
        records = [rec(4, 128, 500) for _ in range(10)]
        report = {"raw_token_positions": 999999, "loss_bearing_tokens": 5000}
        out = perf_mod.reconstruct_check(report, records)
        self.assertFalse(out["ok"])
        self.assertFalse(out["raw_matches"])
        self.assertTrue(out["loss_matches"])

    def test_the_three_rates_are_ordered(self):
        m = perf_mod.PerfMeter()
        m.raw_tokens, m.loss_tokens, m.accepted_tokens = 1000, 800, 600
        m.pad_tokens, m.context_tokens, m.steps = 200, 200, 4
        r = m.report(wall=1.0)
        self.assertGreater(r["raw_tokens_per_second"], r["useful_loss_bearing_tokens_per_second"])
        self.assertGreater(r["useful_loss_bearing_tokens_per_second"], r["accepted_tokens_per_second"])
        self.assertAlmostEqual(r["packing_occupancy"], 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
