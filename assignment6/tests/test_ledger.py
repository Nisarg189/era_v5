"""Ledger invariants: append-only offsets, batch identity, gap and repeat detection."""
import json, os, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import PAD, fake_docs, make_shard                     # noqa: E402

from tdes import ledger as L, packing                              # noqa: E402


def rec(step, rank, mb, **kw):
    r = {"global_step": step, "rank": rank, "microbatch_id": mb,
         "n_sequences": 4, "sequence_length": 128,
         "loss_tokens_by_lane": {"web": 400}, "tokens_by_lane": {"web": 512}}
    r.update(kw)
    return r


class LedgerBehaviour(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "l", "consumption.main.jsonl")
        self.open_ledgers = []

    def tearDown(self):
        for led in self.open_ledgers:
            led.close()

    def ledger(self):
        led = L.Ledger(self.path)
        self.open_ledgers.append(led)
        return led

    def test_offsets_are_the_line_count(self):
        led = self.ledger()
        for i in range(20):
            self.assertEqual(led.append(rec(i, 0, f"m{i}")), i)
        self.assertEqual(led.offset, 20)
        self.assertEqual(len(led.read_all()), 20)

    def test_a_record_survives_without_a_clean_close(self):
        """Every append is flushed and synced, so a crash cannot lose the tail."""
        led = self.ledger()
        led.append(rec(0, 0, "m0"))
        with open(self.path) as f:
            self.assertEqual(len(f.readlines()), 1)

    def test_reading_from_an_offset_returns_the_tail(self):
        led = self.ledger()
        for i in range(10):
            led.append(rec(i, 0, f"m{i}"))
        self.assertEqual([r["global_step"] for r in led.read_from(7)], [7, 8, 9])
        self.assertEqual([r["global_step"] for r in led.read_range(2, 5)], [2, 3, 4])
        self.assertEqual(led.at(4)["microbatch_id"], "m4")

    def test_repeated_microbatches_are_caught(self):
        rs = [rec(0, 0, "a"), rec(1, 0, "b"), rec(1, 0, "b")]
        for i, r in enumerate(rs):
            r["ledger_offset"] = i
        problems = L.check_contiguous(rs)
        self.assertTrue(any("repeated" in p for p in problems))

    def test_skipped_steps_are_caught(self):
        rs = [rec(0, 0, "a"), rec(1, 0, "b"), rec(4, 0, "c")]
        for i, r in enumerate(rs):
            r["ledger_offset"] = i
        problems = L.check_contiguous(rs)
        self.assertTrue(any("gap" in p for p in problems))

    def test_a_correct_stream_has_no_problems(self):
        rs = [rec(s, r, f"s{s}r{r}") for s in range(6) for r in range(2)]
        for i, r in enumerate(rs):
            r["ledger_offset"] = i
        self.assertEqual(L.check_contiguous(rs), [])

    def test_branch_book_records_the_divergence_point(self):
        book = L.BranchBook(os.path.join(os.path.dirname(self.path), "branches.json"))
        book.open_branch("main")
        book.open_branch("exp", parent="main", forked_at_step=8, forked_at_offset=32,
                         from_checkpoint="ckpt-main-00008", reason="test")
        with open(book.path) as f:
            saved = json.load(f)
        self.assertEqual(saved["exp"]["parent_branch"], "main")
        self.assertEqual(saved["exp"]["forked_at_offset"], 32)
        self.assertIsNone(saved["main"]["parent_branch"])


class BatchIdentity(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(11)
        m, t = make_shard(tmp, "s", fake_docs("web", "pretrain", 24, rng), "web")
        self.packs = packing.pack(packing.make_items(m, t), 128, PAD, "best_fit")[:4]

    def test_the_same_batch_hashes_the_same(self):
        self.assertEqual(L.batch_fingerprint(self.packs), L.batch_fingerprint(self.packs))

    def test_a_different_order_is_a_different_batch(self):
        self.assertNotEqual(L.batch_fingerprint(self.packs),
                            L.batch_fingerprint(list(reversed(self.packs))))

    def test_the_same_tokens_under_a_different_mask_is_a_different_batch(self):
        """The fingerprint covers the mask, not only the tokens."""
        before = L.batch_fingerprint(self.packs)
        self.packs[0].loss_mask[3] = 1 - self.packs[0].loss_mask[3]
        self.assertNotEqual(before, L.batch_fingerprint(self.packs))

    def test_changed_position_ids_change_the_hash(self):
        before = L.batch_fingerprint(self.packs)
        self.packs[0].position_ids[5] += 1
        self.assertNotEqual(before, L.batch_fingerprint(self.packs))


if __name__ == "__main__":
    unittest.main()
