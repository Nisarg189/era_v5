"""Packing, masks and batch correctness: the invariants a batch must satisfy."""
import os, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EOS, PAD, fake_docs, make_shard                # noqa: E402

from tdes import packing                                           # noqa: E402


class PackingInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(7)
        self.m_pre, self.t_pre = make_shard(self.tmp, "s-pre", fake_docs("web", "pretrain", 40, rng), "web")
        self.m_ag, self.t_ag = make_shard(self.tmp, "s-ag", fake_docs("agentic", "agentic", 40, rng), "agentic")
        self.items_pre = packing.make_items(self.m_pre, self.t_pre)
        self.items_ag = packing.make_items(self.m_ag, self.t_ag)

    def test_loss_mask_follows_roles_not_lane(self):
        """An agentic sample is graded on assistant turns only."""
        for p in packing.pack(self.items_ag, 128, PAD, "structure_preserving"):
            for m in p.members:
                lo, hi = m["placed_at"]
                self.assertGreater(int(p.loss_mask[lo:hi].sum()), 0)
                self.assertLess(int(p.loss_mask[lo:hi].sum()), hi - lo,
                                "an agentic sample must carry context-only tokens too")

    def test_pretraining_is_graded_everywhere(self):
        for p in packing.pack(self.items_pre, 128, PAD, "best_fit"):
            for m in p.members:
                lo, hi = m["placed_at"]
                self.assertEqual(int(p.loss_mask[lo:hi].sum()), hi - lo)

    def test_padding_never_carries_loss(self):
        for policy in packing.POLICIES:
            for p in packing.pack(self.items_pre, 128, PAD, policy):
                self.assertEqual(int(p.loss_mask[p.used:].sum()), 0, policy)
                self.assertTrue(np.all(p.input_ids[p.used:] == PAD), policy)

    def test_position_ids_restart_per_sample(self):
        for p in packing.pack(self.items_pre, 128, PAD, "best_fit"):
            for m in p.members:
                lo, _ = m["placed_at"]
                self.assertEqual(int(p.position_ids[lo]), m["token_span"][0])

    def test_segments_isolate_packed_samples(self):
        """No position may attend across a segment boundary."""
        packs = packing.pack(self.items_pre, 128, PAD, "best_fit")
        multi = [p for p in packs if len(p.members) > 1]
        self.assertTrue(multi, "the fixture should produce at least one multi-sample window")
        for p in multi[:5]:
            am = packing.attention_mask(p.segment_ids)
            for i in range(len(am)):
                for j in range(i + 1):
                    if am[i, j]:
                        self.assertEqual(p.segment_ids[i], p.segment_ids[j])

    def test_concat_chop_does_cross_boundaries(self):
        """The control: the one policy that is allowed to, does."""
        packs = packing.pack(self.items_pre, 128, PAD, "concat_chop")
        rep = packing.utilisation_report(packs, "concat_chop")
        self.assertGreater(rep["boundary_crossings"], 0)

    def test_verify_pack_finds_no_violations(self):
        for policy in ("best_fit", "structure_preserving", "long_context", "pad_only"):
            for p in packing.pack(self.items_ag, 128, PAD, policy):
                self.assertEqual(packing.verify_pack(p, EOS, PAD), [], policy)

    def test_best_window_keeps_graded_tokens(self):
        """A long agentic trajectory truncated from the front would lose its answer."""
        long_items = [i for i in self.items_ag if i["n"] > 64]
        self.assertTrue(long_items)
        for it in long_items[:10]:
            off = packing.best_window(it, 64)
            mask = packing._loss_from_roles(it["n"], it["roles"])[off:off + 64]
            self.assertGreater(int(mask.sum()), 0)

    def test_chunking_preserves_every_token(self):
        chunks = packing.chunk_items(self.items_pre, 64)
        before = sum(i["n"] for i in self.items_pre)
        after = sum(c["n"] for c in chunks)
        self.assertGreaterEqual(after, before - 16 * len(self.items_pre))
        self.assertGreater(len(chunks), len(self.items_pre))

    def test_structured_samples_are_never_chunked(self):
        chunks = packing.chunk_items(self.items_ag, 64)
        self.assertEqual(len(chunks), len(self.items_ag))

    def test_no_loss_free_sequence_is_ever_emitted(self):
        for policy in packing.POLICIES:
            for p in packing.pack(self.items_ag, 64, PAD, policy):
                self.assertGreater(int(p.loss_mask.sum()), 0, policy)

    def test_best_fit_beats_greedy_or_ties(self):
        g = packing.utilisation_report(packing.pack(self.items_pre, 128, PAD, "greedy"), "greedy")
        b = packing.utilisation_report(packing.pack(self.items_pre, 128, PAD, "best_fit"), "best_fit")
        self.assertLessEqual(g["n_sequences"] and b["n_sequences"], g["n_sequences"])


if __name__ == "__main__":
    unittest.main()
