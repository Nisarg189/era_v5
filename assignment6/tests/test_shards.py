"""Shards, manifests, hashes and deduplication."""
import copy, os, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import EOS, fake_docs, make_shard                     # noqa: E402

from tdes import shards                                            # noqa: E402


class ShardIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rng = np.random.default_rng(3)
        self.m, self.t = make_shard(self.tmp, "s0", fake_docs("web", "pretrain", 30, self.rng), "web")

    def test_a_fresh_shard_verifies(self):
        ok, detail = shards.verify_shard(self.tmp, self.m)
        self.assertTrue(ok, detail)

    def test_one_altered_token_is_detected(self):
        tampered = shards.load_tokens(self.tmp, self.m).copy()
        tampered[len(tampered) // 2] = (int(tampered[len(tampered) // 2]) + 1) % 500
        other = tempfile.mkdtemp()
        np.save(os.path.join(other, self.m["token_file"]), tampered)
        ok, detail = shards.verify_shard(other, self.m)
        self.assertFalse(ok, "an edited token file must not verify")
        self.assertIn("content hash", detail)

    def test_a_reordered_sample_index_is_detected(self):
        """The hash covers the index as well as the tokens."""
        m2 = copy.deepcopy(self.m)
        m2["samples"] = list(reversed(m2["samples"]))
        ok, _ = shards.verify_shard(self.tmp, m2)
        self.assertFalse(ok)

    def test_truncation_is_detected(self):
        other = tempfile.mkdtemp()
        np.save(os.path.join(other, self.m["token_file"]),
                shards.load_tokens(self.tmp, self.m)[:-50])
        ok, _ = shards.verify_shard(other, self.m)
        self.assertFalse(ok)

    def test_exact_duplicates_are_removed(self):
        docs = fake_docs("web", "pretrain", 12, np.random.default_rng(5))
        m, t = make_shard(tempfile.mkdtemp(), "dup", docs + docs, "web")
        self.assertEqual(m["dedup_report"]["samples_in"], 24)
        self.assertGreaterEqual(m["dedup_report"]["exact_duplicates_removed"], 12)
        self.assertEqual(m["n_samples"], m["dedup_report"]["samples_kept"])

    def test_repack_keeps_spans_consistent(self):
        toks = shards.load_tokens(self.tmp, self.m)
        for s in self.m["samples"]:
            self.assertEqual(s["spans"][0][0], s["start"])
            self.assertEqual(s["spans"][-1][1], s["end"])
            self.assertEqual(s["n_tokens"], s["end"] - s["start"])
            self.assertLessEqual(s["end"], len(toks))

    def test_every_sample_ends_with_eos(self):
        toks = shards.load_tokens(self.tmp, self.m)
        for s in self.m["samples"]:
            self.assertEqual(int(toks[s["end"] - 1]), EOS)

    def test_loss_token_count_matches_the_spans(self):
        for s in self.m["samples"]:
            self.assertEqual(s["n_loss_tokens"],
                             sum(b - a for a, b, r in s["spans"] if r == "text"))


if __name__ == "__main__":
    unittest.main()
