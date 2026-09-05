"""
Experiment A: is the Kronecker codec invertible, and where does it stop being so?

Claim 5 of the assignment needs the codec to be reversible: an embedding must map
back to exactly one token. The learned projection is lossy, but the codec that feeds
it is not. This experiment proves the codec is an exact bijection for every token
whose UTF-8 length fits the byte window, by encoding and decoding the real 9,975-token
Session 2 vocabulary and checking each token returns unchanged.

It also measures the one and only way the codec loses information: two tokens that
agree on their first pos_dim bytes collide to the same code. The collision count is
reported per script and per window (32, 48, 64), which is the count the Session 7
lesson explicitly asks the assignment to produce.
"""

import json, os, sys, unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rkron.codec import KroneckerCodec

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_vocab():
    for c in ["/root/a7/data/vocab.json",
              "/mnt/user-data/uploads/assignments/webapp/assignment2/tokenizer/vocab.json",
              os.path.join(_HERE, "..", "..", "webapp", "assignment2", "tokenizer", "vocab.json")]:
        if os.path.isfile(c):
            return c
    return c


VOCAB = _resolve_vocab()
OUT = os.path.join(_HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)


def script_of(tok):
    for ch in tok:
        if ch.isspace():
            continue
        o = ord(ch)
        if o < 128:
            return "latin/ascii"
        if 0x0900 <= o <= 0x097F:
            return "devanagari"
        if 0x0A80 <= o <= 0x0AFF:
            return "gujarati"
        if 0x0C00 <= o <= 0x0C7F:
            return "telugu"
        try:
            name = unicodedata.name(ch)
            return name.split(" ")[0].lower()
        except ValueError:
            return "other"
    return "space/empty"


def audit(tokens, pos_dim):
    codec = KroneckerCodec(pos_dim=pos_dim)
    per = defaultdict(lambda: {"n": 0, "fits_window": 0, "roundtrip_ok": 0})
    code_map = {}
    collisions = defaultdict(int)
    coll_examples = []
    for tok in tokens:
        sc = script_of(tok)
        per[sc]["n"] += 1
        blen = len(tok.encode("utf-8"))
        if blen <= pos_dim:
            per[sc]["fits_window"] += 1
        if codec.roundtrip_ok(tok):
            per[sc]["roundtrip_ok"] += 1
        key = codec.encode(tok).tobytes()
        if key in code_map:
            collisions[sc] += 1
            if len(coll_examples) < 8 and code_map[key] != tok:
                coll_examples.append((code_map[key], tok, sc))
        else:
            code_map[key] = tok
    return {
        "pos_dim": pos_dim,
        "per_script": {k: dict(v) for k, v in per.items()},
        "collisions_per_script": dict(collisions),
        "total_collisions": sum(collisions.values()),
        "distinct_codes": len(code_map),
        "n_tokens": len(tokens),
        "collision_examples": coll_examples,
    }


def main():
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    tokens = list(vocab.keys()) if isinstance(vocab, dict) else list(vocab)
    print(f"real Session 2 vocabulary: {len(tokens)} tokens")

    results = {"n_tokens": len(tokens), "windows": {}}
    for pd in [32, 48, 64]:
        a = audit(tokens, pd)
        results["windows"][pd] = a
        rt = sum(v["roundtrip_ok"] for v in a["per_script"].values())
        fit = sum(v["fits_window"] for v in a["per_script"].values())
        print(f"\npos_dim={pd}: {a['total_collisions']} collisions, "
              f"{a['distinct_codes']}/{a['n_tokens']} distinct codes; "
              f"{fit} tokens fit window; round-trip exact on {rt}/{a['n_tokens']}")
        for sc, v in sorted(a["per_script"].items(), key=lambda kv: -kv[1]["n"]):
            c = a["collisions_per_script"].get(sc, 0)
            print(f"    {sc:16s} n={v['n']:5d}  fits={v['fits_window']:5d}  "
                  f"roundtrip_ok={v['roundtrip_ok']:5d}  collisions={c}")
        if a["collision_examples"]:
            print("    example collisions:",
                  ", ".join(f"{x!r}={y!r}" for x, y, _ in a["collision_examples"][:4]))

    # headline invertibility claim: exact round-trip for every token that fits the window
    exact = True
    for pd, a in results["windows"].items():
        for sc, v in a["per_script"].items():
            if v["roundtrip_ok"] != v["fits_window"]:
                exact = False
    results["exact_inverse_for_all_in_window"] = exact
    print(f"\nExact inverse for every token within the window: {exact}")

    with open(os.path.join(OUT, "invertibility.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("-> artifacts/invertibility.json")


if __name__ == "__main__":
    main()
