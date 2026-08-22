#!/usr/bin/env python3
"""Append-only ledgers: the run's memory.

A ledger is a file that is only ever appended to.  Nothing in it is edited and
nothing is deleted.  A correction is a new record, never a rewrite of an old one,
because the value of the file is precisely that no one could go back and adjust it
once the consequences were known.

Two ledgers are kept, and they point in opposite directions.

The consumption ledger records what was served: one record per microbatch, naming
the shards, the token spans, the loss mask hash, the lane, the stage and the
selector decision.  Its length is the offset a checkpoint binds itself to.

The learning ledger records what came back: per shard, the loss the model actually
paid on those tokens, how that changed across passes, and where the difficulty was
concentrated.

Each branch gets its own consumption file, so an offset is unambiguous.  Forking
writes a divergence record naming the parent branch, the step and the offset, which
is what makes a comparison between two branches an experiment rather than a guess.
"""
import hashlib, json, os

import numpy as np


def array_hash(*arrays):
    """A stable digest over integer arrays, insensitive to platform word size."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def batch_fingerprint(packs):
    """The identity of a packed microbatch.

    Covers the tokens, the loss mask, the position ids and the segment ids, so a
    batch that carries the same tokens under a different mask is a different batch.
    This is the value compared after a resume and after a replay.
    """
    return array_hash(
        np.stack([p.input_ids for p in packs]) if packs else np.zeros((0, 0)),
        np.stack([p.loss_mask for p in packs]) if packs else np.zeros((0, 0)),
        np.stack([p.position_ids for p in packs]) if packs else np.zeros((0, 0)),
        np.stack([p.segment_ids for p in packs]) if packs else np.zeros((0, 0)),
    )


class Ledger:
    """One append-only JSONL file, with an offset that is simply the line count."""

    def __init__(self, path, resume=False):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not resume and os.path.exists(path):
            os.remove(path)
        self._n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        self._f = open(path, "a")

    def append(self, record):
        """Write one record and return the offset it was written at.

        The file is flushed and synced on every append.  A ledger that loses its
        tail in a crash cannot be used to recover from that crash.
        """
        record = dict(record, ledger_offset=self._n)
        self._f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self._n += 1
        return self._n - 1

    @property
    def offset(self):
        return self._n

    def close(self):
        self._f.close()

    # ------------------------------------------------------------------ reading
    def read_all(self):
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_from(self, offset):
        return [r for r in self.read_all() if r["ledger_offset"] >= offset]

    def read_range(self, lo, hi):
        return [r for r in self.read_all() if lo <= r["ledger_offset"] < hi]

    def at(self, offset):
        for r in self.read_all():
            if r["ledger_offset"] == offset:
                return r
        return None


class BranchBook:
    """Which branches exist, and exactly where each one diverged."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.branches = {}

    def open_branch(self, branch_id, parent=None, forked_at_step=None,
                    forked_at_offset=None, from_checkpoint=None, reason=""):
        self.branches[branch_id] = {
            "branch_id": branch_id, "parent_branch": parent,
            "forked_at_step": forked_at_step, "forked_at_offset": forked_at_offset,
            "forked_from_checkpoint": from_checkpoint, "reason": reason,
        }
        self.save()
        return self.branches[branch_id]

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.branches, f, indent=1)
        return self.path


def consumption_path(ledger_dir, branch_id):
    return os.path.join(ledger_dir, f"consumption.{branch_id}.jsonl")


# ------------------------------------------------------------------ ledger audits
def check_contiguous(records):
    """No batch skipped and none repeated, across the whole consumption ledger.

    A resume that restarts the loader too early repeats work and a resume that
    restarts it too late skips work.  Both look healthy in a loss curve, so the
    ledger is where they are caught.
    """
    problems = []
    seen = {}
    for r in records:
        key = (r["global_step"], r["rank"], r["microbatch_id"])
        if key in seen:
            problems.append(f"repeated microbatch {key} at offsets {seen[key]} and {r['ledger_offset']}")
        seen[key] = r["ledger_offset"]
    steps = sorted({r["global_step"] for r in records})
    for a, b in zip(steps, steps[1:]):
        if b != a + 1:
            problems.append(f"gap in global steps between {a} and {b}")
    return problems


def lane_token_totals(records, key="loss_tokens"):
    out = {}
    for r in records:
        for lane, n in r.get(f"{key}_by_lane", {}).items():
            out[lane] = out.get(lane, 0) + n
    return out
