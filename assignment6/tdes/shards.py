#!/usr/bin/env python3
"""Immutable tokenised shards, and the manifests that describe them.

A shard is written once and then sealed.  Its content hash is computed over the
token bytes and the sample index together, so neither the tokens nor their
segment boundaries can move without the hash changing.  Correcting a shard does
not edit it; it produces a new shard whose manifest names the old one as parent.

One storage layout serves all three data types, because the difference between
them is not how tokens are stored but which spans carry loss:

    tokens.npy    one flat uint16 array of token ids for the whole shard
    manifest      a sample index of (start, end, role) spans into that array

A pretraining document is one span with role ``text``.  An SFT sample is a
``prompt`` span and a ``response`` span.  An agentic trajectory is a run of
``system``, ``user``, ``assistant`` and ``tool`` spans in their original order.
Roles decide the loss mask later, so the mask is derived from the data rather
than assumed from the lane.
"""
import hashlib, json, os

import numpy as np

from .corpus import LOSS_ROLES

DTYPE = np.uint16


def _sha256_bytes(*chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


# ------------------------------------------------------------------ tokenisation
def tokenize_documents(tok, docs, eos_id):
    """Turn documents into one flat token array plus a sample index.

    Every sample ends with EOS.  The EOS belongs to the final span of the sample,
    so a pretraining document that is later concatenated with another still tells
    the model where it stopped.
    """
    flat, samples = [], []
    for d in docs:
        start_sample = len(flat)
        spans = []
        for seg in d["segments"]:
            s0 = len(flat)
            flat.extend(tok.encode(seg["text"]).ids)
            if len(flat) > s0:
                spans.append([s0, len(flat), seg["role"]])
        if not spans:
            continue
        flat.append(eos_id)
        spans[-1][1] = len(flat)
        samples.append({
            "sample_id": d["doc_id"],
            "doc_id": d["doc_id"],
            "source_row_id": d["source_row_id"],
            "kind": d["kind"],
            "start": start_sample,
            "end": len(flat),
            "n_tokens": len(flat) - start_sample,
            "n_loss_tokens": sum(b - a for a, b, r in spans if r in LOSS_ROLES),
            "spans": spans,
        })
    return np.asarray(flat, dtype=DTYPE), samples


# ------------------------------------------------------------------ dedup fingerprints
def _ngram_hashes(ids, n=8, stride=4):
    """Stable 64-bit hashes of token n-grams, used for dedup and contamination."""
    out = set()
    for i in range(0, max(0, len(ids) - n + 1), stride):
        window = np.asarray(ids[i:i + n], dtype=DTYPE).tobytes()
        out.add(int.from_bytes(hashlib.blake2b(window, digest_size=8).digest(), "big"))
    return out


def sample_fingerprint(tokens, sample, n=8, stride=4):
    return _ngram_hashes(tokens[sample["start"]:sample["end"]], n=n, stride=stride)


def dedup(tokens, samples, jaccard_threshold=0.8):
    """Exact and near duplicate removal over token n-grams.

    Exact duplicates are caught by hashing the whole token span.  Near duplicates
    are caught by n-gram overlap.  Returns (kept_samples, report).
    """
    seen_exact, kept, exact_drop, near_drop = {}, [], 0, 0
    prints = []
    for s in samples:
        span = tokens[s["start"]:s["end"]].tobytes()
        h = hashlib.sha256(span).hexdigest()
        if h in seen_exact:
            exact_drop += 1
            continue
        fp = sample_fingerprint(tokens, s)
        dup = False
        for prev in prints:
            if not fp or not prev:
                continue
            inter = len(fp & prev)
            if inter and inter / len(fp | prev) >= jaccard_threshold:
                dup = True
                break
        if dup:
            near_drop += 1
            continue
        seen_exact[h] = True
        prints.append(fp)
        kept.append(s)
    return kept, {"exact_duplicates_removed": exact_drop,
                  "near_duplicates_removed": near_drop,
                  "jaccard_threshold": jaccard_threshold,
                  "samples_in": len(samples), "samples_kept": len(kept)}


def repack(tokens, samples):
    """Rebuild a compact token array after samples were dropped.

    Spans are absolute offsets into the array, so dropping a sample means every
    later offset must move.  Doing this here keeps shard files free of holes.
    """
    flat, out = [], []
    for s in samples:
        shift = len(flat) - s["start"]
        flat.extend(tokens[s["start"]:s["end"]].tolist())
        t = dict(s)
        t["start"], t["end"] = s["start"] + shift, s["end"] + shift
        t["spans"] = [[a + shift, b + shift, r] for a, b, r in s["spans"]]
        out.append(t)
    return np.asarray(flat, dtype=DTYPE), out


# ------------------------------------------------------------------ the shard object
def write_shard(shard_dir, shard_id, tokens, samples, meta):
    """Write the token array and return the sealed manifest."""
    os.makedirs(shard_dir, exist_ok=True)
    token_path = os.path.join(shard_dir, f"{shard_id}.npy")
    np.save(token_path, tokens)

    index_bytes = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    content_hash = _sha256_bytes(tokens.tobytes(), index_bytes)

    manifest = {
        "shard_id": shard_id,
        "content_hash": content_hash,
        "token_file": os.path.basename(token_path),
        "n_tokens": int(tokens.size),
        "n_samples": len(samples),
        "n_loss_tokens": int(sum(s["n_loss_tokens"] for s in samples)),
        "samples": samples,
    }
    manifest.update(meta)
    return manifest


def load_tokens(shard_dir, manifest):
    return np.load(os.path.join(shard_dir, manifest["token_file"]))


def verify_shard(shard_dir, manifest):
    """Recompute the content hash from what is on disk and compare.

    Returns (ok, detail).  This is the check that makes a shard immutable in
    practice rather than by convention: it catches an edited token file, a
    reordered sample index and a truncated write alike.
    """
    path = os.path.join(shard_dir, manifest["token_file"])
    if not os.path.exists(path):
        return False, "token file missing"
    tokens = np.load(path)
    index_bytes = json.dumps(manifest["samples"], sort_keys=True, separators=(",", ":")).encode()
    got = _sha256_bytes(tokens.tobytes(), index_bytes)
    if got != manifest["content_hash"]:
        return False, f"content hash {got[:16]} does not match manifest {manifest['content_hash'][:16]}"
    if int(tokens.size) != manifest["n_tokens"]:
        return False, "token count does not match manifest"
    return True, "content hash matches"


def write_manifest(manifest_dir, manifest):
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, f"{manifest['shard_id']}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return path
