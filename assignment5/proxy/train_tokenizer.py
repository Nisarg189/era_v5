#!/usr/bin/env python3
"""Train the single tokenizer shared by every proxy run.

The tokenizer is a controlled variable here, not the object of study, so one
tokenizer is trained once on a balanced sample of all four lanes and then frozen.
It is character-level (not byte-level) for the reason established in Session 2:
Devanagari costs about three UTF-8 bytes per character, so a byte-level vocabulary
would silently give the Hindi lane roughly a third of the content per token and
confound every mixture comparison.

Results are still reported in bits per byte, which removes the tokenizer from the
comparison entirely.

    python3 train_tokenizer.py [--vocab 16384]
"""
import argparse, json, os, random

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
LANES = ["web", "code", "indic", "math"]
SAMPLE_MB = 25          # per lane, for training the vocabulary


def lane_sample(lane, mb, seed=0):
    """A deterministic sample of whole documents from one lane."""
    with open(os.path.join(DATA, f"{lane}.txt")) as f:
        docs = f.read().split("\n\n")
    rnd = random.Random(seed)
    rnd.shuffle(docs)
    out, total = [], 0
    for d in docs:
        out.append(d)
        total += len(d)
        if total >= mb * 1e6:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16384)
    a = ap.parse_args()

    corpus = []
    for lane in LANES:
        s = lane_sample(lane, SAMPLE_MB)
        print(f"{lane:<6} {len(s):>6,} docs  {sum(len(x) for x in s)/1e6:.1f} MB")
        corpus += s
    random.Random(1).shuffle(corpus)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # split on whitespace, keeping the leading space with the word (Session 2 shape)
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=["<unk>", "<eos>"],
        show_progress=True,
        # every character seen at least this often becomes part of the base alphabet
        initial_alphabet=[],
        min_frequency=2,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    path = os.path.join(DATA, "proxy_tokenizer.json")
    tok.save(path)
    print("saved", path, "vocab", tok.get_vocab_size())

    # fertility + unk rate per lane, on text the tokenizer did not train on
    stats = {"vocab_size": tok.get_vocab_size(), "lanes": {}}
    unk_id = tok.token_to_id("<unk>")
    for lane in LANES:
        held = "\n\n".join(lane_sample(lane, 2, seed=99))
        ids = tok.encode(held).ids
        n_unk = sum(1 for i in ids if i == unk_id)
        stats["lanes"][lane] = {
            "chars": len(held),
            "bytes": len(held.encode()),
            "tokens": len(ids),
            "words": len(held.split()),
            "tokens_per_word": round(len(ids) / max(1, len(held.split())), 3),
            "bytes_per_token": round(len(held.encode()) / max(1, len(ids)), 3),
            "unk_rate": round(n_unk / max(1, len(ids)), 6),
        }
        print(lane, stats["lanes"][lane])
    with open(os.path.join(DATA, "tokenizer_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
