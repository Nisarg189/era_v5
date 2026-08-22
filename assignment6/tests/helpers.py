"""Fixtures for the invariant tests.

The tests build shards from generated token arrays rather than from the network,
so they run offline, in about a second, and test the system rather than the data.
The demonstration in run_demo.py is what exercises the real corpus.
"""
import os, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdes import shards                                            # noqa: E402

VOCAB, EOS, PAD = 512, 1, 2


def fake_docs(lane, kind, n, rng, min_len=40, max_len=300):
    docs = []
    for i in range(n):
        length = int(rng.integers(min_len, max_len))
        text = " ".join(str(int(x)) for x in rng.integers(10, VOCAB - 1, length))
        if kind == "pretrain":
            segs = [{"role": "text", "text": text}]
        elif kind == "sft":
            cut = len(text) // 3
            segs = [{"role": "prompt", "text": text[:cut]},
                    {"role": "response", "text": text[cut:]}]
        else:
            cut = len(text) // 4
            segs = [{"role": "system", "text": text[:cut]},
                    {"role": "user", "text": text[cut:2 * cut]},
                    {"role": "tool", "text": text[2 * cut:3 * cut]},
                    {"role": "assistant", "text": text[3 * cut:]}]
        docs.append({"doc_id": f"{lane}-{i:04d}", "lane": lane, "kind": kind,
                     "source": "test", "source_row_id": str(i), "segments": segs,
                     "chars": len(text)})
    return docs


class WordTokenizer:
    """A deterministic stand-in: each whitespace token maps to an integer."""

    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self.Enc([int(w) % VOCAB for w in text.split() if w.isdigit()])

    def token_to_id(self, t):
        return {"<eos>": EOS, "<pad>": PAD}.get(t, 0)

    def get_vocab_size(self):
        return VOCAB

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def make_shard(tmpdir, shard_id, docs, lane, split="train", never_train=False, extra=None):
    tok = WordTokenizer()
    toks, samples = shards.tokenize_documents(tok, docs, EOS)
    kept, rep = shards.dedup(toks, samples)
    toks, kept = shards.repack(toks, kept)
    meta = {"split": split, "never_train": never_train, "capability_lane": lane,
            "source": "test", "language": "en", "script": "Latn", "license_tier": "A",
            "tokenizer_hash": "t" * 64, "cleaning_pipeline_hash": "c" * 64,
            "dedup_status": "pass", "dedup_report": rep,
            "contamination_status": "unscanned", "eval_overlap": False,
            "provenance_tier": "test", "parent_shard_ids": [],
            "data_kind": docs[0]["kind"]}
    meta.update(extra or {})
    m = shards.write_shard(tmpdir, shard_id, toks, kept, meta)
    return m, toks
