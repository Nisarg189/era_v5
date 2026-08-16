#!/usr/bin/env python3
"""Tokenize each lane once into a flat uint16 array, and split off a held-out tail.

Every run reads these arrays, so tokenization never happens inside a training run
and every arm of every experiment sees byte-identical data.

For each lane, the last VAL_FRAC of the token stream is held out and never
trained on. The byte length of the held-out text is recorded so that validation
loss can be converted to bits per byte.

    python3 tokenize_corpus.py
"""
import json, os

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
LANES = ["web", "code", "indic", "math"]
VAL_FRAC = 0.02
CHUNK_DOCS = 2000


def main():
    tok = Tokenizer.from_file(os.path.join(DATA, "proxy_tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    unk = tok.token_to_id("<unk>")
    meta = {"val_frac": VAL_FRAC, "eos_id": eos, "vocab_size": tok.get_vocab_size(), "lanes": {}}

    for lane in LANES:
        with open(os.path.join(DATA, f"{lane}.txt")) as f:
            docs = f.read().split("\n\n")
        ids, n_unk = [], 0
        for i in range(0, len(docs), CHUNK_DOCS):
            for enc in tok.encode_batch(docs[i:i + CHUNK_DOCS]):
                ids.extend(enc.ids)
                ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        n_unk = int((arr == unk).sum())

        n_val = int(len(arr) * VAL_FRAC)
        train, val = arr[:-n_val], arr[-n_val:]
        # bytes behind the held-out tokens, for bits per byte
        val_bytes = len(tok.decode(val.tolist()).encode())

        np.save(os.path.join(DATA, f"{lane}.train.npy"), train)
        np.save(os.path.join(DATA, f"{lane}.val.npy"), val)
        meta["lanes"][lane] = {
            "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "val_bytes": val_bytes, "unk_tokens": n_unk,
            "unk_rate": round(n_unk / len(arr), 8),
            "bytes_per_val_token": round(val_bytes / max(1, len(val)), 3),
        }
        print(f"{lane:<6} train {len(train)/1e6:>7.2f}M  val {len(val)/1e6:.2f}M  "
              f"unk {n_unk}  bytes/token {meta['lanes'][lane]['bytes_per_val_token']}")

    with open(os.path.join(DATA, "token_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
