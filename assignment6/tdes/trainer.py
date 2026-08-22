#!/usr/bin/env python3
"""The training loop, and the four things that can happen to it.

Resume continues the same run from the last checkpoint and the ledger offset that
checkpoint was bound to.  Replay restores an older checkpoint and feeds the same
historical stream again.  Fork restores a checkpoint and starts a new branch on
purpose.  Audit reconstructs what trained a range of steps.

A checkpoint saves more than the model.  It saves the optimiser, the scheduler
position, the random number generator states, the selector's scoring snapshot, the
dataloader state and the ledger offset.  A checkpoint without a data position
cannot resume a run, only restart one.

The crash is deliberate and it happens in the middle of a step, after some
microbatches have been served and recorded but before the optimiser has applied
anything and before any new checkpoint exists.  That is the realistic case and the
hard one: the ledger has run ahead of the model, and recovery has to go back to the
model's position rather than the ledger's end.
"""
import copy, hashlib, json, os, random, time

import numpy as np
import torch

from . import packing
from .dataloader import microbatch_record, microbatch_tensors
from .ledger import batch_fingerprint
from .model import cosine_lr

DATALOADER_VERSION = "1.0.0"


class CrashInjected(RuntimeError):
    """Raised to simulate a hardware or process failure mid step."""


# ------------------------------------------------------------------ checkpoints
def state_digest(state):
    """A hash over a state dict, so a checkpoint can be identified without keeping it."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes()
                 if hasattr(v, "detach") else repr(v).encode())
    return h.hexdigest()


def save_checkpoint(path, *, checkpoint_id, step, tokens_seen, ledger_offset, branch_id,
                    run_id, model, optimizer, stream_state, tokenizer_hash, segment,
                    with_optimizer=True):
    """Write a checkpoint.

    The scorer snapshot is not stored: it is by definition the model at this
    checkpoint, so keeping a second copy of the same weights would double the file
    for nothing.  The optimiser is stored only for the checkpoints a resume could
    land on.  Replay and fork rebuild the data stream and never step the optimiser,
    so an older checkpoint needs the model, the dataloader state, the generator
    states and the ledger offset, and nothing else.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "checkpoint_id": checkpoint_id,
        "global_step": step,
        "tokens_seen": tokens_seen,
        "ledger_offset": ledger_offset,
        "branch_id": branch_id,
        "run_id": run_id,
        "segment": segment,
        "tokenizer_hash": tokenizer_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if with_optimizer else None,
        "has_optimizer": bool(with_optimizer),
        "model_state_digest": state_digest(model.state_dict()),
        "dataloader_state": stream_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }, path)
    return path


def checkpoint_summary(ckpt):
    return {k: ckpt[k] for k in ("checkpoint_id", "global_step", "tokens_seen",
                                 "ledger_offset", "branch_id", "run_id", "segment",
                                 "tokenizer_hash", "has_optimizer", "model_state_digest")}


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ the trainer
class Trainer:
    def __init__(self, model, stream, ledger, trace, perf, *, run_id, branch_id,
                 tokenizer_hash, ckpt_dir, lr=3e-3, total_steps=100, warmup=8,
                 ckpt_interval=32, log=print, segment=0, keep_checkpoints=6,
                 keep_optimizer=2):
        self.model, self.stream, self.ledger = model, stream, ledger
        self.trace, self.perf = trace, perf
        self.run_id, self.branch_id = run_id, branch_id
        self.tokenizer_hash, self.ckpt_dir = tokenizer_hash, ckpt_dir
        self.lr, self.total_steps, self.warmup = lr, total_steps, warmup
        self.ckpt_interval = ckpt_interval
        self.keep_checkpoints, self.keep_optimizer = keep_checkpoints, keep_optimizer
        self.log = log
        self.segment = segment
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                           weight_decay=0.1)
        self.last_checkpoint = "none"
        self.checkpoints = []
        self.scorer_snapshot = copy.deepcopy(model.state_dict())
        self.step_losses = []
        self.pruned = []

    # -------------------------------------------------------------- state moves
    def restore(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # the scoring snapshot is the model at this checkpoint, by definition
        self.scorer_snapshot = copy.deepcopy(ckpt["model"])
        self.stream.load_state(ckpt["dataloader_state"])
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        self.last_checkpoint = ckpt["checkpoint_id"]
        return ckpt

    def save(self, step):
        cid = f"ckpt-{self.branch_id}-{step:05d}"
        path = os.path.join(self.ckpt_dir, cid + ".pt")
        save_checkpoint(
            path, checkpoint_id=cid, step=step, tokens_seen=self.stream.tokens_seen,
            ledger_offset=self.ledger.offset, branch_id=self.branch_id, run_id=self.run_id,
            model=self.model, optimizer=self.optimizer,
            stream_state=self.stream.state(), tokenizer_hash=self.tokenizer_hash,
            segment=self.segment)
        self.last_checkpoint = cid
        self.scorer_snapshot = copy.deepcopy(self.model.state_dict())
        self.checkpoints.append({
            "checkpoint_id": cid, "path": path, "global_step": step,
            "ledger_offset": self.ledger.offset, "tokens_seen": self.stream.tokens_seen,
            "branch_id": self.branch_id, "segment": self.segment,
            "model_state_digest": state_digest(self.model.state_dict()),
            "bytes": os.path.getsize(path), "has_optimizer": True, "retained": True})
        self.log(f"[PASS] checkpoint_saved {cid} step={step} "
                 f"ledger_offset={self.ledger.offset} tokens={self.stream.tokens_seen}")
        self._retain()
        return cid

    def _retain(self):
        """Bound what a demonstration run leaves on disk.

        Optimiser state is dropped from all but the most recent few checkpoints, and
        the oldest model files are removed once newer ones exist.  The metadata for
        every checkpoint ever written is kept, including the digest of its weights,
        so a pruned checkpoint is still named and identified in the index.
        """
        live = [c for c in self.checkpoints if c["retained"]]
        for c in live[:-self.keep_optimizer]:
            if not c["has_optimizer"]:
                continue
            blob = torch.load(c["path"], map_location="cpu", weights_only=False)
            blob["optimizer"], blob["has_optimizer"] = None, False
            torch.save(blob, c["path"])
            c["has_optimizer"] = False
            c["bytes"] = os.path.getsize(c["path"])
        for c in live[:-self.keep_checkpoints]:
            if os.path.exists(c["path"]):
                os.remove(c["path"])
            c["retained"] = False
            self.pruned.append(c["checkpoint_id"])

    # -------------------------------------------------------------- one step
    def train_step(self, dry_run=False, crash_after_microbatch=None):
        """Run one optimiser step.  Returns (step_info, records).

        ``dry_run`` builds and records nothing: it only rebuilds the batches, which
        is what replay and the resume comparison need.
        """
        self.perf.watch.start("loader")
        info = self.stream.build_step(self.last_checkpoint)
        self.perf.watch.stop()

        records, mbs = [], info["microbatches"]
        if dry_run:
            for mb in mbs:
                records.append(microbatch_record(info, mb, self.tokenizer_hash,
                                                 DATALOADER_VERSION, self.run_id,
                                                 self.branch_id, self.last_checkpoint))
            return info, records

        lr = cosine_lr(info["global_step"], self.total_steps, self.lr, self.warmup)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)

        step_loss, n_mb = 0.0, max(1, len(mbs))
        for i, mb in enumerate(mbs):
            ids, pos, seg, lm = microbatch_tensors(mb, torch)
            self.perf.watch.start("compute")
            per, m, loss = self.model.token_losses(ids, pos, seg, lm)
            (loss / n_mb).backward()
            self.perf.watch.stop()
            step_loss += float(loss.detach()) / n_mb

            self._attribute(info, mb, ids, pos, per.detach(), m.detach())
            self.perf.add_batch(mb["packs"])

            rec = microbatch_record(info, mb, self.tokenizer_hash, DATALOADER_VERSION,
                                    self.run_id, self.branch_id, self.last_checkpoint)
            rec["segment"] = self.segment
            rec["mean_loss"] = round(float(loss), 6)
            rec["learning_rate"] = round(lr, 8)
            self.ledger.append(rec)
            records.append(rec)

            if crash_after_microbatch is not None and i == crash_after_microbatch:
                raise CrashInjected(
                    f"injected failure during step {info['global_step']} after "
                    f"microbatch {mb['microbatch_id']}")

        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.optimizer.step()
        self.trace.record_grad_norm(info["global_step"], gn)
        self.perf.steps += 1
        self.step_losses.append({"global_step": info["global_step"], "loss": round(step_loss, 6),
                                 "grad_norm": round(gn, 5), "lr": round(lr, 8),
                                 "stage": info["curriculum_stage"],
                                 "sequence_length": info["sequence_length"]})
        return info, records

    def _attribute(self, info, mb, ids, pos, per, m):
        """Send each graded token's loss back to the shard and document it came from."""
        scores = {d["sample_id"]: d["opus_score"] for d in info["decisions"]}
        for b, p in enumerate(mb["packs"]):
            for mem in p.members:
                lo, hi = mem["placed_at"]
                a, z = lo, min(hi - 1, per.shape[1])
                if z <= a:
                    continue
                sel = m[b, a:z] > 0
                if not bool(sel.any()):
                    continue
                idx = torch.nonzero(sel).flatten() + a
                self.trace.add(
                    step=info["global_step"], stage=info["curriculum_stage"],
                    shard_id=mem["shard_id"], doc_id=mem["doc_id"], lane=mem["lane"],
                    sample_id=mem["sample_id"],
                    repeated_pass=mem.get("repeated_pass", 1),
                    token_ids=ids[b, idx + 1].tolist(),
                    losses=per[b, idx].tolist(),
                    positions=pos[b, idx + 1].tolist(),
                    checkpoint_id=self.last_checkpoint,
                    opus_score=scores.get(mem["sample_id"]),
                    tokens_seen=self.stream.tokens_seen)

    # -------------------------------------------------------------- runs
    def run(self, n_steps, crash_at=None, crash_after_microbatch=1, checkpoint_first=True):
        """Train ``n_steps`` steps, optionally crashing at a given global step."""
        if checkpoint_first and not self.checkpoints:
            self.save(self.stream.step)
        for _ in range(n_steps):
            step = self.stream.step
            crash = crash_after_microbatch if crash_at is not None and step == crash_at else None
            info, _ = self.train_step(crash_after_microbatch=crash)
            if (info["global_step"] + 1) % self.ckpt_interval == 0:
                self.save(self.stream.step)
        return self.stream.step


def stream_signature(records):
    """The identity of a stretch of the data stream, for comparison."""
    return [{"global_step": r["global_step"], "microbatch_id": r["microbatch_id"],
             "rank": r["rank"], "batch_hash": r["batch_hash"],
             "token_span_ids": r["token_span_ids"],
             "packed_sample_ids": r["packed_sample_ids"]} for r in records]


def compare_signatures(a, b):
    """Field by field comparison of two stretches of stream.  Returns a verdict."""
    diffs = []
    if len(a) != len(b):
        diffs.append(f"length {len(a)} against {len(b)}")
    for x, y in zip(a, b):
        for k in ("global_step", "microbatch_id", "rank", "batch_hash",
                  "token_span_ids", "packed_sample_ids"):
            if x[k] != y[k]:
                diffs.append(f"step {x['global_step']} {x['microbatch_id']}: {k} differs")
    return {"matched": not diffs, "compared_microbatches": min(len(a), len(b)),
            "differences": diffs[:20]}


# ------------------------------------------------------------------ audit
def audit_token_range(records, lo_tokens, hi_tokens, seq_len_key="sequence_length"):
    """Which shards, lanes and selector decisions trained a token interval."""
    seen, shards, lanes, samples = 0, {}, {}, 0
    steps = []
    for r in records:
        n = r["n_sequences"] * r[seq_len_key]
        if seen + n > lo_tokens and seen < hi_tokens:
            steps.append(r["global_step"])
            samples += len(r["packed_sample_ids"])
            for s in r["shard_ids"]:
                shards[s] = shards.get(s, 0) + 1
            for lane, t in r["tokens_by_lane"].items():
                lanes[lane] = lanes.get(lane, 0) + t
        seen += n
    return {
        "token_range": [lo_tokens, hi_tokens],
        "global_steps": [min(steps), max(steps)] if steps else [],
        "n_microbatches": len(steps),
        "n_packed_samples": samples,
        "shards_involved": dict(sorted(shards.items(), key=lambda kv: -kv[1])),
        "tokens_by_lane": dict(sorted(lanes.items(), key=lambda kv: -kv[1])),
    }
