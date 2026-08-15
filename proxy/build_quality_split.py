#!/usr/bin/env python3
"""Split the Hindi lane into a bulk pool and a top-quality reserve pool.

Experiment E2 holds back an arbitrary slice of Hindi and finds that timing alone
buys nothing. E2b repeats it with a reserve that is actually better data, which is
what the session means by reserving the strongest tier A material.

The quality signal is the Session 4 edu_score in substance, rescaled because its
clamps saturate on Sangraha verified. The reserve it selects is long, lexically
rich, well-structured prose, which is what an edu classifier selects in practice.

The last 3% of documents are excluded, because the evaluation tail is drawn from
there and the reserve must not leak into it.

Output: data/indic_bulk.train.npy and data/indic_topq.train.npy

    python3 build_quality_split.py [--reserve-frac 0.15]
"""
import argparse, json, os, re

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

VAL_EXCLUDE = 0.03


def components(text):
    """The three signals behind the Session 4 edu_score: lexical diversity,
    sentence structure and length."""
    words = text.split()
    n = len(words)
    if n == 0:
        return None
    sents = [s for s in re.split(r"[।॥.!?]", text) if s.strip()]
    # root type-token ratio: lexical richness with the length bias removed,
    # since a plain types/tokens ratio falls mechanically as documents grow
    return (len(set(words)) / (n ** 0.5), (n / len(sents)) if sents else 0.0, n)


def rank_scores(docs):
    """Rank documents by quality.

    The Session 4 edu_score is reused in substance but not in form. Its clamps
    (min(diversity*2, 1), min(n/150, 1), a flat structure bonus over a wide band)
    all saturate on Sangraha verified, which is already cleaned: the median
    document scores a flat 1.000, so the original score cannot order this corpus
    at all. The same three signals are therefore rescaled to the corpus's own
    distribution, which restores the ordering the score was meant to express.
    """
    feats = [components(d) for d in docs]
    div = np.array([f[0] if f else 0.0 for f in feats])
    sent = np.array([f[1] if f else 0.0 for f in feats])
    length = np.array([f[2] if f else 0 for f in feats], dtype=float)

    def pct(v):                       # rank-transform to 0..1, distribution-free
        order = v.argsort().argsort()
        return order / max(1, len(v) - 1)

    # sentence structure scores best near the middle of the corpus's own range
    struct = 1.0 - np.abs(pct(sent) - 0.5) * 2
    return 0.40 * pct(div) + 0.35 * struct + 0.25 * pct(np.log1p(length))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-frac", type=float, default=0.15)
    a = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(DATA, "proxy_tokenizer.json"))
    eos = tok.token_to_id("<eos>")

    docs = open(os.path.join(DATA, "indic.txt")).read().split("\n\n")
    cut = int(len(docs) * (1 - VAL_EXCLUDE))
    docs = docs[:cut]
    print(f"{len(docs):,} documents in scope (last {VAL_EXCLUDE:.0%} excluded as evaluation tail)")

    sc = rank_scores(docs)
    scored = sorted(((sc[i], i) for i in range(len(docs))), reverse=True)
    qs = [s for s, _ in scored]
    print(f"quality score: max {qs[0]:.3f}  p90 {qs[len(qs)//10]:.3f}  median {qs[len(qs)//2]:.3f}  "
          f"min {qs[-1]:.3f}")

    # take the highest-scoring documents until the reserve holds reserve_frac of chars
    total_chars = sum(len(d) for d in docs)
    want = total_chars * a.reserve_frac
    top, got = set(), 0
    for s, i in scored:
        top.add(i)
        got += len(docs[i])
        if got >= want:
            break
    print(f"reserve: {len(top):,} docs, {got/1e6:.1f}M chars ({got/total_chars:.1%}), "
          f"score cutoff {scored[len(top)-1][0]:.3f}")

    for name, keep in [("indic_topq", top), ("indic_bulk", set(range(len(docs))) - top)]:
        sel = [docs[i] for i in sorted(keep)]
        ids = []
        for i in range(0, len(sel), 2000):
            for enc in tok.encode_batch(sel[i:i + 2000]):
                ids.extend(enc.ids)
                ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        np.save(os.path.join(DATA, f"{name}.train.npy"), arr)
        cf = [components(d) for d in sel]
        cf = [c for c in cf if c]
        print(f"{name:<12} {len(sel):>7,} docs  {len(arr)/1e6:>7.2f}M tokens  "
              f"mean root-TTR {sum(c[0] for c in cf)/len(cf):.2f}  "
              f"mean words {sum(c[2] for c in cf)/len(cf):.0f}")

    with open(os.path.join(DATA, "quality_split.json"), "w") as f:
        json.dump({"reserve_frac": a.reserve_frac, "val_excluded": VAL_EXCLUDE,
                   "cutoff_score": scored[len(top) - 1][0], "reserve_docs": len(top)}, f, indent=1)


if __name__ == "__main__":
    main()
