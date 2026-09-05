"""
Build a small language-modelling corpus and vocabulary from the real ERA_V5 course
data already on disk: fineweb-edu (web/English), Sangraha (Hindi), NuminaMath (math),
GitHub Python (code) and Hermes function-calling (agentic), plus the Session 2
English/Hindi/Gujarati/Telugu samples. This is the same six-lane mixture Sessions 5
and 6 used, so the proxy is trained on the corpus the model is actually planned around.

Tokenisation is word-level with a character fallback. A candidate word becomes a
vocabulary token only if its UTF-8 length is <= 32 bytes, so the Kronecker codec is
lossless over the whole vocabulary and the reversible head has a well-defined inverse
for every token. Longer strings are emitted as their characters. The residual
>32-byte collision question is problem 3's, not problem 5's, and is kept out on purpose.
"""

import json, os, re, glob
import numpy as np

def _first_existing(candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


# Resolve the course data whether this runs in the cloud sandbox (files staged under
# /mnt/user-data/uploads) or in the repository on a laptop (assignment7 sits beside
# assignment6, two levels above webapp). An env override wins if set.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # assignment7/
A6 = os.environ.get("ERA_A6_CACHE") or _first_existing([
    "/mnt/user-data/uploads/assignments/era_v5/assignment6/data/cache",
    os.path.join(_HERE, "..", "assignment6", "data", "cache"),
])
A2 = os.environ.get("ERA_A2_DATA") or _first_existing([
    "/mnt/user-data/uploads/assignments/webapp/assignment2/data",
    os.path.join(_HERE, "..", "..", "webapp", "assignment2", "data"),
])

# unicode-aware pretokeniser: a run of word characters, or one whitespace, or one other char
_PRETOK = re.compile(r"\w+|\s|[^\w\s]", re.UNICODE)


def _row_to_text(row):
    """Pull a text string out of a cache row, whatever schema it uses."""
    if row.get("text"):
        return row["text"]
    if row.get("code"):
        return row["code"]
    if row.get("problem") or row.get("solution"):
        return (row.get("problem", "") + "\n" + row.get("solution", "")).strip()
    conv = row.get("conversations")
    if isinstance(conv, list):
        parts = [m.get("value", "") for m in conv if isinstance(m, dict)]
        return "\n".join(p for p in parts if p)
    return None


def _rows_text(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for r in d.get("rows", []):
        t = _row_to_text(r.get("row", {}))
        if t:
            out.append(t)
    return out


def load_corpus(max_chars_per_lane=350_000, seed=0):
    """Return {lane: concatenated_text}. Lanes mirror the V5 mixture."""
    rng = np.random.default_rng(seed)
    lanes = {}

    def take(texts, cap):
        rng.shuffle(texts)
        buf, n = [], 0
        for t in texts:
            t = t.strip()
            if not t:
                continue
            buf.append(t)
            n += len(t)
            if n >= cap:
                break
        return "\n".join(buf)[:cap]

    web = []
    for f in sorted(glob.glob(f"{A6}/HuggingFaceFW_fineweb-edu*")):
        web += _rows_text(f)
    if os.path.exists(f"{A2}/en.txt"):
        web.append(open(f"{A2}/en.txt", encoding="utf-8").read())
    lanes["web_en"] = take(web, max_chars_per_lane)

    hin = []
    for f in sorted(glob.glob(f"{A6}/ai4bharat_sangraha*")):
        hin += _rows_text(f)
    for extra in ("hi.txt", "gu.txt", "te.txt"):
        p = f"{A2}/{extra}"
        if os.path.exists(p):
            hin.append(open(p, encoding="utf-8").read())
    lanes["indic"] = take(hin, max_chars_per_lane)

    math = []
    for f in sorted(glob.glob(f"{A6}/AI-MO_NuminaMath*")):
        math += _rows_text(f)
    lanes["math"] = take(math, max_chars_per_lane)

    code = []
    for f in sorted(glob.glob(f"{A6}/codeparrot_github-code*")):
        code += _rows_text(f)
    lanes["code"] = take(code, max_chars_per_lane)

    agent = []
    for f in sorted(glob.glob(f"{A6}/NousResearch_hermes*")):
        agent += _rows_text(f)
    lanes["agentic"] = take(agent, max_chars_per_lane)

    return {k: v for k, v in lanes.items() if v}


def pretokens(text):
    return _PRETOK.findall(text)


def build_vocab(lanes, target_vocab=1200):
    from collections import Counter
    counts = Counter()
    chars = set()
    for text in lanes.values():
        for tok in pretokens(text):
            chars.add(tok if len(tok) == 1 else None)
            for ch in tok:
                chars.add(ch)
            if len(tok.encode("utf-8")) <= 32:
                counts[tok] += 1
    chars.discard(None)
    # every single character is always in the vocabulary (guarantees full coverage)
    vocab = sorted(chars)
    have = set(vocab)
    # then the most frequent multi-character words up to the target size
    for tok, _ in counts.most_common():
        if len(vocab) >= target_vocab:
            break
        if tok not in have and len(tok) > 1:
            vocab.append(tok)
            have.add(tok)
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


def encode(text, stoi):
    ids = []
    for tok in pretokens(text):
        if tok in stoi:
            ids.append(stoi[tok])
        else:
            for ch in tok:
                if ch in stoi:
                    ids.append(stoi[ch])
    return ids


def make_streams(target_vocab=1200, val_frac=0.1, seed=0, max_chars_per_lane=350_000):
    lanes = load_corpus(max_chars_per_lane=max_chars_per_lane, seed=seed)
    vocab, stoi = build_vocab(lanes, target_vocab=target_vocab)
    train_ids, val_ids, lane_stats, val_by_lane = [], [], {}, {}
    for lane, text in lanes.items():
        ids = encode(text, stoi)
        cut = int(len(ids) * (1 - val_frac))
        train_ids += ids[:cut]
        val_ids += ids[cut:]
        val_by_lane[lane] = np.array(ids[cut:], dtype=np.int64)
        lane_stats[lane] = {"chars": len(text), "tokens": len(ids)}
    return {
        "vocab": vocab,
        "stoi": stoi,
        "train": np.array(train_ids, dtype=np.int64),
        "val": np.array(val_ids, dtype=np.int64),
        "val_by_lane": val_by_lane,
        "lane_stats": lane_stats,
    }


def get_batch(ids, batch_size, block, rng):
    ix = rng.integers(0, len(ids) - block - 1, size=batch_size)
    x = np.stack([ids[i:i + block] for i in ix])
    y = np.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x, y
