#!/usr/bin/env python3
"""Packing policies, and the masks that make a packed sequence mean the right thing.

Six policies, matching Session 6 section 5.  Each returns fixed-length sequences
carrying four parallel arrays, because token ids alone do not tell the optimiser
where learning is supposed to happen:

    input_ids     the tokens
    loss_mask     1 where the token is graded, 0 where it is only context
    position_ids  restart at 0 for each packed sample, so a sample that lands
                  second in a window is not told it begins at position 900
    segment_ids   which packed sample a position belongs to, which is what turns
                  a causal mask into a block diagonal one

The loss mask is derived from the segment roles recorded in the shard, never from
the lane.  Pretraining text is graded everywhere.  An SFT sample is graded on the
response only.  An agentic trajectory is graded on the assistant turns, with the
system prompt, the user request and the tool observations left as context.

Utilisation is reported as three separate numbers rather than one, because they
answer different questions.  Occupancy is how much of the window is not padding.
Loss coverage is how much of the window is actually graded.  The gap between them
is real text the GPU paid for and the model did not learn from.
"""
import numpy as np

from .corpus import LOSS_ROLES

POLICIES = ["pad_only", "concat_chop", "greedy", "best_fit",
            "structure_preserving", "long_context"]

# Which policy each data type is served with.  Plain text tolerates concatenation.
# Anything with internal structure must not be cut or allowed to leak.
POLICY_FOR_KIND = {"pretrain": "best_fit", "sft": "structure_preserving",
                   "agentic": "structure_preserving"}



def make_items(shard_manifest, tokens, samples=None):
    """Turn shard samples into packable items with roles relative to the item."""
    out = []
    for s in (samples if samples is not None else shard_manifest["samples"]):
        ids = tokens[s["start"]:s["end"]]
        roles = [(a - s["start"], b - s["start"], r) for a, b, r in s["spans"]]
        out.append({
            "shard_id": shard_manifest["shard_id"],
            "sample_id": s["sample_id"],
            "doc_id": s["doc_id"],
            "lane": shard_manifest["capability_lane"],
            "kind": s["kind"],
            "ids": np.asarray(ids, dtype=np.int64),
            "roles": roles,
            "n": int(len(ids)),
        })
    return out


def best_window(item, seq_len):
    """Where to start reading a sample that does not fit the window.

    Truncating a long agentic trajectory from the front leaves a window holding the
    system prompt, the user request and nothing the model is graded on, which costs
    full compute and teaches nothing.  The offset that carries the most graded
    tokens is chosen instead, so the assistant turns survive the cut.  For plain
    pretraining every token is graded and this returns zero.
    """
    n = item["n"]
    if n <= seq_len:
        return 0
    mask = _loss_from_roles(n, item["roles"])
    cum = np.concatenate([[0], np.cumsum(mask)])
    covered = cum[seq_len:] - cum[:n - seq_len + 1]
    return int(np.argmax(covered))


def chunk_items(items, seq_len):
    """Split plain pretraining documents into window-sized spans.

    A four thousand token web document is not one training sequence, it is many.
    Truncating it to the first window would throw away most of the document and
    would also make the lane under-deliver against its mixture quota, because the
    quota is counted in sequences.  Chunking keeps every token and keeps each span
    isolated, which concatenate-and-chop does not.

    Structured samples are never chunked: an SFT pair or an agent trajectory cut in
    half is not two training examples.
    """
    out = []
    for it in items:
        if it["kind"] != "pretrain" or it["n"] <= seq_len:
            out.append(it)
            continue
        for off in range(0, it["n"], seq_len):
            piece = it["ids"][off:off + seq_len]
            if len(piece) < 16:
                continue
            child = dict(it)
            child["ids"] = piece
            child["n"] = len(piece)
            child["roles"] = [(0, len(piece), "text")]
            child["sample_id"] = f"{it['sample_id']}#{off}"
            child["chunk_offset"] = off
            out.append(child)
    return out


def _loss_from_roles(n, roles):
    m = np.zeros(n, dtype=np.int64)
    for a, b, r in roles:
        if r in LOSS_ROLES:
            m[max(0, a):min(n, b)] = 1
    return m


class Pack:
    """One fixed-length packed sequence."""

    def __init__(self, seq_len, pad_id):
        self.seq_len, self.pad_id = seq_len, pad_id
        self.input_ids = np.full(seq_len, pad_id, dtype=np.int64)
        self.loss_mask = np.zeros(seq_len, dtype=np.int64)
        self.position_ids = np.zeros(seq_len, dtype=np.int64)
        self.segment_ids = np.full(seq_len, -1, dtype=np.int64)
        self.members = []
        self.used = 0

    def room(self):
        return self.seq_len - self.used

    def add(self, item, take=None, offset=0, isolate=True):
        """Place part or all of an item.  ``offset`` is where in the item to start."""
        take = min(take if take is not None else item["n"] - offset, self.room())
        if take <= 0:
            return 0
        ids = item["ids"][offset:offset + take]
        lo, hi = self.used, self.used + take
        self.input_ids[lo:hi] = ids
        full_mask = _loss_from_roles(item["n"], item["roles"])
        self.loss_mask[lo:hi] = full_mask[offset:offset + take]
        seg = len(self.members) if isolate else 0
        self.segment_ids[lo:hi] = seg
        self.position_ids[lo:hi] = np.arange(offset, offset + take) if isolate else np.arange(lo, hi)
        self.members.append({
            "shard_id": item["shard_id"], "sample_id": item["sample_id"],
            "doc_id": item["doc_id"], "lane": item["lane"], "kind": item["kind"],
            "token_span": [int(offset), int(offset + take)],
            "placed_at": [int(lo), int(hi)], "segment": int(seg),
            "truncated": bool(offset + take < item["n"]),
            "repeated_pass": int(item.get("repeated_pass", 1)),
        })
        self.used = hi
        return take

    def stats(self):
        n_pad = int(self.seq_len - self.used)
        return {
            "sequence_length": self.seq_len,
            "occupied_tokens": int(self.used),
            "pad_tokens": n_pad,
            "loss_tokens": int(self.loss_mask.sum()),
            "context_only_tokens": int(self.used - int(self.loss_mask.sum())),
            "occupancy": round(self.used / self.seq_len, 6),
            "loss_coverage": round(float(self.loss_mask.sum()) / self.seq_len, 6),
            "n_members": len(self.members),
            "segments": int(len({m["segment"] for m in self.members})),
            "truncated_members": sum(1 for m in self.members if m["truncated"]),
        }


# ------------------------------------------------------------------ the six policies
def pack(items, seq_len, pad_id, policy):
    """Pack items, then drop any sequence that ended up carrying no loss.

    A window with no graded token costs the same compute as any other and produces
    no gradient.  Dropping it is not hiding a problem: the count is reported by
    ``utilisation_report`` as ``dropped_loss_free_sequences``.
    """
    packs = _dispatch(items, seq_len, pad_id, policy)
    keep = [p for p in packs if int(p.loss_mask.sum()) > 0]
    dropped = len(packs) - len(keep)
    for p in keep:
        p.dropped_siblings = dropped
    return keep


def _dispatch(items, seq_len, pad_id, policy):
    if policy == "pad_only":
        return _pad_only(items, seq_len, pad_id)
    if policy == "concat_chop":
        return _concat_chop(items, seq_len, pad_id)
    if policy == "greedy":
        return _fit(items, seq_len, pad_id, order=None)
    if policy == "best_fit":
        return _fit(items, seq_len, pad_id, order="desc")
    if policy == "structure_preserving":
        return _structure_preserving(items, seq_len, pad_id)
    if policy == "long_context":
        return _long_context(items, seq_len, pad_id)
    raise ValueError(f"unknown packing policy {policy!r}")


def _pad_only(items, seq_len, pad_id):
    """One sample per window.  Structure is perfectly safe, compute is wasted."""
    packs = []
    for it in items:
        p = Pack(seq_len, pad_id)
        p.add(it, take=min(it["n"], seq_len), offset=best_window(it, seq_len))
        packs.append(p)
    return packs


def _concat_chop(items, seq_len, pad_id):
    """Join the token stream and cut fixed windows out of it.

    Efficient, and it cuts samples wherever the window ends.  Acceptable for plain
    pretraining text.  Never used for structured data, which is why this policy is
    not in POLICY_FOR_KIND.
    """
    packs, p = [], Pack(seq_len, pad_id)
    for it in items:
        offset = 0
        while offset < it["n"]:
            if p.room() == 0:
                packs.append(p)
                p = Pack(seq_len, pad_id)
            offset += p.add(it, offset=offset, isolate=False)
    if p.used:
        packs.append(p)
    return packs


def _fit(items, seq_len, pad_id, order):
    """First fit in arrival order (greedy), or best fit on a length-sorted queue.

    Both keep samples whole and isolate them, so the only difference between the
    two is the order items are offered in.  That difference is the whole result:
    the same items reach very different utilisation.
    """
    queue = sorted(items, key=lambda x: -x["n"]) if order == "desc" else list(items)
    packs = []
    for it in queue:
        n = min(it["n"], seq_len)
        target = None
        if order == "desc":
            best = None
            for p in packs:
                r = p.room()
                if r >= n and (best is None or r < best.room()):
                    best = p
            target = best
        else:
            for p in packs:
                if p.room() >= n:
                    target = p
                    break
        if target is None:
            target = Pack(seq_len, pad_id)
            packs.append(target)
        target.add(it, take=n, offset=best_window(it, seq_len))
    return packs


def _structure_preserving(items, seq_len, pad_id):
    """Best fit, but a sample is never split across windows.

    A sample longer than the window is placed alone and truncated, and the
    truncation is recorded rather than hidden, because a silently cut agent
    trajectory teaches the model to stop mid tool call.
    """
    long_items = [i for i in items if i["n"] > seq_len]
    short_items = [i for i in items if i["n"] <= seq_len]
    packs = _fit(short_items, seq_len, pad_id, order="desc")
    for it in long_items:
        p = Pack(seq_len, pad_id)
        p.add(it, take=seq_len, offset=best_window(it, seq_len))
        packs.append(p)
    return packs


def _long_context(items, seq_len, pad_id):
    """Fill the expensive windows first with the longest available material.

    A long-context window costs quadratically in attention, so an unused position
    here is worth far more than an unused position in a short window.  Long items
    anchor each window and short items fill the tail.
    """
    pool = sorted(items, key=lambda x: -x["n"])
    packs = []
    while pool:
        p = Pack(seq_len, pad_id)
        anchor = pool.pop(0)
        p.add(anchor, take=min(anchor["n"], seq_len), offset=best_window(anchor, seq_len))
        i = 0
        while i < len(pool):
            if pool[i]["n"] <= p.room():
                p.add(pool.pop(i), take=None)
            else:
                i += 1
        packs.append(p)
    return packs


# ------------------------------------------------------------------ reporting
def utilisation_report(packs, policy):
    if not packs:
        return {"policy": policy, "n_sequences": 0}
    st = [p.stats() for p in packs]
    total_positions = sum(s["sequence_length"] for s in st)
    occupied = sum(s["occupied_tokens"] for s in st)
    loss = sum(s["loss_tokens"] for s in st)
    return {
        "policy": policy,
        "n_sequences": len(packs),
        "total_positions": total_positions,
        "occupied_tokens": occupied,
        "pad_tokens": total_positions - occupied,
        "loss_bearing_tokens": loss,
        "context_only_tokens": occupied - loss,
        "occupancy": round(occupied / total_positions, 6),
        "loss_coverage": round(loss / total_positions, 6),
        "wasted_positions": total_positions - occupied,
        "mean_segments_per_sequence": round(sum(s["segments"] for s in st) / len(st), 4),
        "truncated_samples": sum(s["truncated_members"] for s in st),
        "boundary_crossings": sum(max(0, s["n_members"] - s["segments"]) for s in st),
        "dropped_loss_free_sequences": getattr(packs[0], "dropped_siblings", 0),
    }


def attention_mask(segment_ids):
    """Causal, and additionally blocked between segments.

    Position i may attend to position j when j <= i and both belong to the same
    packed sample.  Without the second condition the model learns that one
    document is a natural continuation of an unrelated one.
    """
    n = len(segment_ids)
    idx = np.arange(n)
    causal = idx[:, None] >= idx[None, :]
    same = segment_ids[:, None] == segment_ids[None, :]
    return causal & same


def verify_pack(p, eos_id, pad_id):
    """Invariants every packed sequence must satisfy.  Returns list of violations."""
    bad = []
    n = p.seq_len
    if len(p.input_ids) != n or len(p.loss_mask) != n or len(p.position_ids) != n:
        bad.append("array lengths disagree with the sequence length")
    if p.used < n:
        if not np.all(p.input_ids[p.used:] == pad_id):
            bad.append("tail is not padding")
        if p.loss_mask[p.used:].sum():
            bad.append("padding carries loss")
    if p.loss_mask[:p.used].size and p.loss_mask.sum() == 0:
        bad.append("packed sequence carries no loss at all")
    for m in p.members:
        lo, hi = m["placed_at"]
        if not np.all(p.segment_ids[lo:hi] == m["segment"]):
            bad.append(f"segment ids disagree for {m['sample_id']}")
        if m["segment"] >= 0 and p.position_ids[lo] != m["token_span"][0]:
            bad.append(f"position ids do not restart for {m['sample_id']}")
    if np.any(p.loss_mask[p.segment_ids == -1]):
        bad.append("loss applied outside any segment")
    return bad
