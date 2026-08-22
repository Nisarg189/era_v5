"""The invariant the whole system exists for: a stream restored from its saved
state produces exactly the batches it would have produced without the interruption,
and a stream that restores only the model does not."""
import copy, os, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import PAD, EOS, fake_docs, make_shard                # noqa: E402

from tdes import mixture, registry as reg_mod                      # noqa: E402
from tdes.dataloader import DataStream                             # noqa: E402
from tdes.ledger import batch_fingerprint                          # noqa: E402
from tdes.opus import Opus                                         # noqa: E402

TOKHASH = "t" * 64
LANE_KIND = {"web": "pretrain", "code": "pretrain", "indic": "pretrain",
             "stem": "sft", "reasoning": "sft", "agentic": "agentic"}


def deterministic_scorer(item):
    """A stand-in for the model based scorer, so this test has no model in it.

    The score still varies by sample, so selection is a real decision and not a
    pass-through, and it is a pure function of the sample so it replays exactly.
    """
    h = sum(int(x) for x in item["ids"][:32])
    return ((h % 200) / 100.0 - 1.0), 1.0, 1.0


def build_registry(tmp):
    reg = reg_mod.Registry(tmp, TOKHASH)
    rng = np.random.default_rng(4)
    for lane, kind in LANE_KIND.items():
        m, t = make_shard(tmp, f"train-{lane}", fake_docs(lane, kind, 60, rng), lane)
        ok, fails, _ = reg.admit(m, t)
        assert ok, (lane, fails)
    return reg


def make_stream(reg, seed=1234, schedule=None):
    sched = schedule or mixture.Schedule(400_000, 128)
    opus = Opus(deterministic_scorer, mixture.PROTECTED_FLOORS, "test-proxy")
    s = DataStream(reg, sched, opus, PAD, EOS, ranks=2, microbatch=4, accumulation=2,
                   seed=seed)
    s.attach_allocator(mixture.QuotaAllocator(mixture.PROTECTED_FLOORS))
    return s


def run(stream, n):
    out = []
    for _ in range(n):
        info = stream.build_step("ckpt-test")
        for mb in info["microbatches"]:
            out.append((info["global_step"], mb["microbatch_id"],
                        batch_fingerprint(mb["packs"])))
    return out


class ResumeIsExact(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = build_registry(self.tmp)

    def test_a_restored_stream_continues_identically(self):
        a = make_stream(self.reg)
        run(a, 5)
        saved = copy.deepcopy(a.state())
        rest_of_a = run(a, 5)

        b = make_stream(self.reg)
        b.load_state(saved)
        rest_of_b = run(b, 5)

        self.assertEqual(len(rest_of_a), len(rest_of_b))
        self.assertEqual(rest_of_a, rest_of_b,
                         "a stream restored from its saved state must be byte identical")

    def test_a_fresh_loader_does_not_continue_the_stream(self):
        """The control: without the loader state, the run silently repeats data."""
        a = make_stream(self.reg)
        run(a, 5)
        saved = copy.deepcopy(a.state())
        expected = run(a, 1)

        naive = make_stream(self.reg)
        got = run(naive, 1)
        self.assertNotEqual(expected[0][2], got[0][2])
        self.assertEqual(got[0][0], 0, "a fresh loader restarts at step 0")
        self.assertEqual(saved["step"], 5)

    def test_the_saved_state_carries_all_four_pieces(self):
        a = make_stream(self.reg)
        run(a, 6)
        st = a.state()
        for key in ("cursors", "epochs", "allocator", "tokens_seen", "step"):
            self.assertIn(key, st)
        self.assertGreater(st["tokens_seen"], 0)
        self.assertTrue(any(v > 0 for v in st["cursors"].values()))
        self.assertTrue(any(abs(v) > 0 for v in st["allocator"]["carry"].values()),
                        "the quota remainder must be part of the saved state")

    def test_dropping_the_quota_carry_changes_the_stream(self):
        """Each saved field is load bearing, not decoration."""
        a = make_stream(self.reg)
        run(a, 5)
        saved = copy.deepcopy(a.state())
        expected = run(a, 3)

        broken = copy.deepcopy(saved)
        broken["allocator"]["carry"] = {l: 0.0 for l in broken["allocator"]["carry"]}
        b = make_stream(self.reg)
        b.load_state(broken)
        self.assertNotEqual(expected, run(b, 3))

    def test_dropping_the_cursors_changes_the_stream(self):
        a = make_stream(self.reg)
        run(a, 5)
        saved = copy.deepcopy(a.state())
        expected = run(a, 3)

        broken = copy.deepcopy(saved)
        broken["cursors"] = {l: 0 for l in broken["cursors"]}
        b = make_stream(self.reg)
        b.load_state(broken)
        self.assertNotEqual(expected, run(b, 3))

    def test_replaying_the_same_interval_twice_agrees(self):
        a = make_stream(self.reg)
        run(a, 4)
        saved = copy.deepcopy(a.state())
        first = run(a, 4)

        for _ in range(2):
            b = make_stream(self.reg)
            b.load_state(saved)
            self.assertEqual(first, run(b, 4))

    def test_a_fork_with_a_changed_mixture_diverges(self):
        a = make_stream(self.reg)
        run(a, 4)
        saved = copy.deepcopy(a.state())
        parent = run(a, 3)

        forked_schedule = mixture.Schedule(400_000, 128)
        for st in forked_schedule.stages:
            st["mixture"]["indic"] = round(st["mixture"]["indic"] + 0.10, 4)
            st["mixture"]["web"] = round(st["mixture"]["web"] - 0.10, 4)
        f = make_stream(self.reg, schedule=forked_schedule)
        f.load_state(saved)
        self.assertNotEqual(parent, run(f, 3))

    def test_every_step_serves_the_full_batch(self):
        a = make_stream(self.reg)
        for _ in range(8):
            info = a.build_step("ckpt-test")
            n = sum(len(mb["packs"]) for mb in info["microbatches"])
            self.assertEqual(n, a.seqs_per_step)

    def test_the_selector_never_sees_held_out_data(self):
        for lane in LANE_KIND:
            for it in make_stream(self.reg).pool[lane]:
                self.assertFalse(it["doc_id"].startswith("held-"))


if __name__ == "__main__":
    unittest.main()


class RepetitionAccounting(unittest.TestCase):
    """Repeated passes are counted, and the count survives a restore.

    The demonstration run is too short for any lane to be consumed twice, so the
    mechanism is proved here instead of being left unproved.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = reg_mod.Registry(self.tmp, TOKHASH)
        rng = np.random.default_rng(8)
        for lane, kind in LANE_KIND.items():
            n = 4 if lane == "agentic" else 60      # a lane small enough to wrap
            m, t = make_shard(self.tmp, f"train-{lane}", fake_docs(lane, kind, n, rng), lane)
            ok, fails, _ = self.reg.admit(m, t)
            assert ok, fails

    def test_a_small_lane_wraps_and_counts_the_pass(self):
        s = make_stream(self.reg)
        seen = set()
        for _ in range(30):
            info = s.build_step("ckpt-test")
            for mb in info["microbatches"]:
                for p in mb["packs"]:
                    for mem in p.members:
                        if mem["lane"] == "agentic":
                            seen.add(mem["sample_id"])
        self.assertGreater(s.epochs["agentic"], 0, "a four document lane must wrap")
        self.assertEqual(s.state()["epochs"]["agentic"], s.epochs["agentic"])

    def test_the_pass_number_survives_a_restore(self):
        s = make_stream(self.reg)
        run(s, 35)
        saved = copy.deepcopy(s.state())
        self.assertGreater(saved["epochs"]["agentic"], 0)
        b = make_stream(self.reg)
        b.load_state(saved)
        self.assertEqual(b.epochs, s.epochs)
        self.assertEqual(run(b, 4), run(s, 4))

    def test_repeated_pass_is_attached_to_every_drawn_item(self):
        s = make_stream(self.reg)
        for _ in range(25):
            s.build_step("ckpt-test")
        items = s.draw("agentic", 2000)
        self.assertTrue(items)
        self.assertTrue(all(i["repeated_pass"] >= 1 for i in items))
        self.assertGreater(max(i["repeated_pass"] for i in items), 1)
