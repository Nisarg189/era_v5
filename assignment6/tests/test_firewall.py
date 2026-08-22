"""The admission gate and the evaluation firewall."""
import copy, os, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import fake_docs, make_shard                          # noqa: E402

from tdes import registry as reg_mod                               # noqa: E402

TOKHASH = "t" * 64


class Firewall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(21)
        self.train_docs = fake_docs("web", "pretrain", 40, rng)
        self.test_docs = fake_docs("held", "pretrain", 20, np.random.default_rng(99))
        self.m_test, self.t_test = make_shard(self.tmp, "test-0", self.test_docs, "web",
                                              split="test", never_train=True)
        self.reg = reg_mod.Registry(self.tmp, TOKHASH)
        self.reg.register_heldout(self.m_test, self.t_test)

    def test_a_clean_shard_is_admitted(self):
        m, t = make_shard(self.tmp, "clean-0", self.train_docs, "web")
        ok, fails, hits = self.reg.admit(m, t)
        self.assertTrue(ok, fails)
        self.assertEqual(hits, [])

    def test_an_exact_overlap_is_blocked(self):
        m, t = make_shard(self.tmp, "dirty-0", self.train_docs[:20] + self.test_docs[:5], "web")
        ok, fails, hits = self.reg.admit(m, t)
        self.assertFalse(ok)
        self.assertGreaterEqual(len(hits), 5)
        self.assertTrue(any(h["match"] == "exact_content_hash" for h in hits))

    def test_a_near_duplicate_is_blocked(self):
        """Contamination survives small edits, so exact hashing alone is not enough."""
        edited = copy.deepcopy(self.test_docs[:4])
        for d in edited:
            d["doc_id"] = d["doc_id"] + "-edited"
            d["segments"][0]["text"] = d["segments"][0]["text"] + " 42 43 44"
        m, t = make_shard(self.tmp, "near-0", self.train_docs[:20] + edited, "web")
        ok, fails, hits = self.reg.admit(m, t)
        self.assertFalse(ok)
        self.assertTrue(any(h["match"] == "ngram_fingerprint" for h in hits))

    def test_a_blocked_shard_does_not_enter_the_registry(self):
        m, t = make_shard(self.tmp, "dirty-1", self.train_docs[:10] + self.test_docs[:3], "web")
        self.reg.admit(m, t)
        self.assertNotIn("dirty-1", [x["shard_id"] for x in self.reg.trainable()])

    def test_a_wrong_tokenizer_hash_is_blocked(self):
        m, t = make_shard(self.tmp, "wrongtok", self.train_docs, "web",
                          extra={"tokenizer_hash": "z" * 64})
        ok, fails, _ = self.reg.admit(m, t)
        self.assertFalse(ok)
        self.assertTrue(any("tokenizer hash" in f for f in fails))

    def test_an_unusable_licence_is_blocked(self):
        m, t = make_shard(self.tmp, "badlic", self.train_docs, "web",
                          extra={"license_tier": "unknown"})
        ok, fails, _ = self.reg.admit(m, t)
        self.assertFalse(ok)
        self.assertTrue(any("licence" in f for f in fails))

    def test_missing_cleaning_lineage_is_blocked(self):
        m, t = make_shard(self.tmp, "nolineage", self.train_docs, "web",
                          extra={"cleaning_pipeline_hash": ""})
        ok, fails, _ = self.reg.admit(m, t)
        self.assertFalse(ok)
        self.assertTrue(any("lineage" in f for f in fails))

    def test_never_train_documents_are_indexed(self):
        self.assertEqual(len(self.reg.never_train_docs()), self.m_test["n_samples"])

    def test_test_shards_are_never_trainable(self):
        for m in self.reg.by_split("test"):
            self.assertNotIn(m, self.reg.trainable())

    def test_probe_shards_are_registered_but_not_trainable(self):
        m, t = make_shard(self.tmp, "probe-0", self.train_docs, "web", extra={"probe": True})
        ok, _, _ = self.reg.admit(m, t)
        self.assertTrue(ok)
        self.assertNotIn("probe-0", [x["shard_id"] for x in self.reg.trainable()])


if __name__ == "__main__":
    unittest.main()
