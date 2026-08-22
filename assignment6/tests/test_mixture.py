"""Mixture quotas, protected floors, warmup and compliance."""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdes import mixture                                           # noqa: E402


class Quotas(unittest.TestCase):
    def setUp(self):
        self.sched = mixture.Schedule(1_000_000, 512)

    def test_every_step_allocates_exactly_the_batch(self):
        alloc = mixture.QuotaAllocator()
        for _ in range(200):
            counts, _ = alloc.allocate(self.sched.stages[1]["mixture"], 16)
            self.assertEqual(sum(counts.values()), 16)
            self.assertTrue(all(v >= 0 for v in counts.values()))

    def test_carried_remainders_make_the_average_converge(self):
        """A lane owed 3.84 sequences a step must average 3.84 over many steps."""
        alloc = mixture.QuotaAllocator(floors={})
        mix = self.sched.stages[1]["mixture"]
        total = {l: 0 for l in mixture.LANES}
        n = 400
        for _ in range(n):
            counts, _ = alloc.allocate(mix, 16)
            for l, v in counts.items():
                total[l] += v
        for l in mixture.LANES:
            self.assertAlmostEqual(total[l] / (n * 16), mix[l], places=2,
                                   msg=f"{l} drifted from its share")

    def test_a_sub_sequence_floor_still_appears(self):
        """A one percent floor on a sixteen sequence batch is 0.16 sequences."""
        alloc = mixture.QuotaAllocator({"agentic": 0.01})
        mix = {l: 0.0 for l in mixture.LANES}
        mix["web"] = 1.0
        seen = 0
        for _ in range(400):
            counts, _ = alloc.allocate(mix, 16)
            seen += counts["agentic"]
        self.assertGreater(seen, 0, "a fractional floor must not round away to nothing")
        self.assertAlmostEqual(seen / (400 * 16), 0.01, places=2)

    def test_a_floor_lifts_a_starved_lane(self):
        alloc = mixture.QuotaAllocator(mixture.PROTECTED_FLOORS)
        mix = {l: 0.0 for l in mixture.LANES}
        mix["web"] = 1.0
        total = {l: 0 for l in mixture.LANES}
        for _ in range(200):
            counts, _ = alloc.allocate(mix, 16)
            for l, v in counts.items():
                total[l] += v
        for lane, floor in mixture.PROTECTED_FLOORS.items():
            self.assertGreaterEqual(total[lane] / (200 * 16), floor - 0.01, lane)

    def test_stage_boundaries_are_blended_not_stepped(self):
        s2 = self.sched.stages[2]
        at_start, _, w0 = self.sched.mixture_at(s2["token_start"] + 1)
        after, _, w1 = self.sched.mixture_at(s2["token_start"] + s2["warmup_tokens"] + 1)
        prev = self.sched.stages[1]["mixture"]
        self.assertGreater(w0, 0.0, "the first token of a stage must still be in warmup")
        self.assertEqual(w1, 0.0)
        self.assertLess(abs(at_start["code"] - prev["code"]),
                        abs(after["code"] - prev["code"]),
                        "the blended mixture must start near the previous stage")

    def test_stages_tile_the_whole_run(self):
        self.assertEqual(self.sched.stages[0]["token_start"], 0)
        self.assertEqual(self.sched.stages[-1]["token_end"], 1_000_000)
        for a, b in zip(self.sched.stages, self.sched.stages[1:]):
            self.assertEqual(a["token_end"], b["token_start"])

    def test_every_stage_mixture_sums_to_one(self):
        for s in self.sched.stages:
            self.assertAlmostEqual(sum(s["mixture"].values()), 1.0, places=6, msg=s["stage"])

    def test_sequence_length_follows_the_session_5_progression(self):
        lens = [s["sequence_length"] for s in self.sched.stages]
        self.assertEqual(lens[0], lens[1], "S0 and S1 share a sequence length")
        self.assertLess(lens[1], lens[2])
        self.assertEqual(max(lens), self.sched.max_seq_len)
        self.assertEqual(lens[4], self.sched.max_seq_len, "S4 is the long context stage")

    def test_compliance_reports_the_error(self):
        rows = mixture.compliance({"web": 50, "code": 50}, {"web": 60, "code": 40})
        self.assertAlmostEqual(rows["web"]["abs_error"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
