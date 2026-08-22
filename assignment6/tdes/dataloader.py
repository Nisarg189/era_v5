#!/usr/bin/env python3
"""The data stream: mixture quota, selector, packing, microbatches, ledger record.

This is the object that has to be exactly reproducible.  Everything it does at
step N depends on what it did at every step before N, which is what makes a naive
restart produce a different stream and what makes ``state`` worth checkpointing:

    lane cursors      where in each lane's shuffled order the stream had reached
    epoch counters    how many passes over a lane have completed, which is the
                      repeated pass number the learning ledger reports
    quota carry       the fractional sequences owed to each lane, from mixture
    tokens consumed   which curriculum stage and which blend applies next

Restore those four and the stream continues.  Fail to restore any one of them and
the run silently repeats or skips data while the loss curve looks perfectly normal.
"""
import random

import numpy as np

from . import packing
from .ledger import batch_fingerprint
from .mixture import LANES


class DataStream:
    def __init__(self, registry, schedule, opus, pad_id, eos_id, *,
                 ranks=2, microbatch=4, accumulation=2, seed=1234,
                 candidate_multiplier=1.8, branch_id="main"):
        self.registry, self.schedule, self.opus = registry, schedule, opus
        self.pad_id, self.eos_id = pad_id, eos_id
        self.ranks, self.microbatch, self.accumulation = ranks, microbatch, accumulation
        self.seqs_per_step = ranks * microbatch * accumulation
        self.seed, self.branch_id = seed, branch_id
        self.candidate_multiplier = candidate_multiplier

        # one deterministic item pool per lane, built from admitted train shards only
        self.pool = {l: [] for l in LANES}
        for m in sorted(registry.trainable(), key=lambda x: x["shard_id"]):
            tokens = registry_tokens(registry, m)
            for it in packing.make_items(m, tokens):
                self.pool[m["capability_lane"]].append(it)

        self.cursors = {l: 0 for l in LANES}
        self.epochs = {l: 0 for l in LANES}
        self.orders = {l: self._order(l, 0) for l in LANES}
        self.tokens_seen = 0
        self.step = 0

    def _order(self, lane, epoch):
        idx = list(range(len(self.pool[lane])))
        random.Random(f"{self.seed}:{lane}:{epoch}").shuffle(idx)
        return idx

    # ------------------------------------------------------------------ state
    def state(self):
        return {"cursors": dict(self.cursors), "epochs": dict(self.epochs),
                "tokens_seen": self.tokens_seen, "step": self.step,
                "seed": self.seed, "branch_id": self.branch_id,
                "allocator": self.allocator.state()}

    def load_state(self, st):
        self.cursors = dict(st["cursors"])
        self.epochs = dict(st["epochs"])
        self.tokens_seen = st["tokens_seen"]
        self.step = st["step"]
        self.seed = st["seed"]
        self.orders = {l: self._order(l, self.epochs[l]) for l in LANES}
        self.allocator.load_state(st["allocator"])

    def attach_allocator(self, allocator):
        self.allocator = allocator
        return self

    # ------------------------------------------------------------------ drawing
    def draw(self, lane, token_budget):
        """Take items from a lane until the token budget is met, wrapping on epoch end."""
        out, got, guard = [], 0, 0
        if not self.pool[lane]:
            return out
        while got < token_budget and guard < 4 * len(self.pool[lane]) + 8:
            guard += 1
            if self.cursors[lane] >= len(self.orders[lane]):
                self.epochs[lane] += 1
                self.orders[lane] = self._order(lane, self.epochs[lane])
                self.cursors[lane] = 0
            it = dict(self.pool[lane][self.orders[lane][self.cursors[lane]]])
            it["repeated_pass"] = self.epochs[lane] + 1
            self.cursors[lane] += 1
            out.append(it)
            got += it["n"]
        return out

    # ------------------------------------------------------------------ one step
    def build_step(self, scoring_checkpoint):
        """Produce the microbatches for the next global step, and their records."""
        mixture, stage, warmup_pos = self.schedule.mixture_at(self.tokens_seen)
        seq_len = stage["sequence_length"]
        counts, floor_seqs = self.allocator.allocate(mixture, self.seqs_per_step)

        # Each lane must deliver exactly the number of sequences its quota names, so
        # the lane is refilled until it does.  A lane that runs short of candidates
        # after a few attempts is left short and the shortfall shows up as mixture
        # error rather than being silently covered by another lane.
        packs, pack_lane, pack_policy, decisions = [], [], [], []
        for lane in LANES:
            target = counts[lane]
            if target <= 0:
                continue
            lane_packs, attempts, policy = [], 0, "none"
            while len(lane_packs) < target and attempts < 6:
                attempts += 1
                short = target - len(lane_packs)
                budget = short * seq_len
                cands = self.draw(lane, int(budget * self.candidate_multiplier))
                if not cands:
                    break
                floor_budget = min(budget, floor_seqs.get(lane, 0.0) * seq_len)
                chosen, decs = self.opus.select(
                    cands, {lane: budget}, stage["stage"], self.step,
                    scoring_checkpoint, {lane: floor_budget})
                decisions.extend(decs)
                items = [i for i, _ in chosen.get(lane, [])]
                if not items:
                    continue
                policy = packing.POLICY_FOR_KIND.get(items[0]["kind"], "best_fit")
                if lane == "web" and seq_len >= self.schedule.max_seq_len:
                    policy = "long_context"      # Session 5 puts long documents in S4
                lane_packs.extend(packing.pack(packing.chunk_items(items, seq_len),
                                               seq_len, self.pad_id, policy))
            for p in lane_packs[:target]:
                packs.append(p)
                pack_lane.append(lane)
                pack_policy.append(policy if lane_packs else "none")

        # a deterministic shuffle so one microbatch is not a single lane
        order = list(range(len(packs)))
        random.Random(f"{self.seed}:step:{self.step}").shuffle(order)
        packs = [packs[i] for i in order]
        pack_lane = [pack_lane[i] for i in order]
        pack_policy = [pack_policy[i] for i in order]

        microbatches = []
        n_mb = self.ranks * self.accumulation
        for m in range(n_mb):
            sl = slice(m * self.microbatch, (m + 1) * self.microbatch)
            mb_packs = packs[sl]
            if not mb_packs:
                continue
            rank = m % self.ranks
            accum = m // self.ranks
            microbatches.append({
                "rank": rank, "accumulation_index": accum,
                "microbatch_id": f"s{self.step:05d}-r{rank}-a{accum}",
                "packs": mb_packs,
                "lanes": pack_lane[sl], "policies": pack_policy[sl],
            })

        consumed = sum(p.seq_len for p in packs)
        self.tokens_seen += consumed
        self.step += 1
        return {
            "global_step": self.step - 1,
            "curriculum_stage": stage["stage"],
            "sequence_length": seq_len,
            "mixture": {l: round(mixture.get(l, 0.0), 6) for l in LANES},
            "warmup_position": warmup_pos,
            "planned_sequences": counts,
            "floor_sequences": floor_seqs,
            "microbatches": microbatches,
            "decisions": decisions,
            "tokens_consumed": consumed,
        }


def registry_tokens(registry, manifest):
    from .shards import load_tokens
    key = manifest["shard_id"]
    cache = getattr(registry, "_token_cache", None)
    if cache is None:
        cache = registry._token_cache = {}
    if key not in cache:
        cache[key] = load_tokens(registry.shard_dir, manifest)
    return cache[key]


def microbatch_tensors(mb, torch):
    """Stack a microbatch into the four arrays the model consumes."""
    packs = mb["packs"]
    return (torch.tensor(np.stack([p.input_ids for p in packs])),
            torch.tensor(np.stack([p.position_ids for p in packs])),
            torch.tensor(np.stack([p.segment_ids for p in packs])),
            torch.tensor(np.stack([p.loss_mask for p in packs])))


def microbatch_record(step_info, mb, tokenizer_hash, dataloader_version, run_id,
                      branch_id, checkpoint_id):
    """The consumption ledger record for one microbatch."""
    packs = mb["packs"]
    members, loss_by_lane, tok_by_lane, shard_ids, spans = [], {}, {}, set(), []
    for p, lane in zip(packs, mb["lanes"]):
        st = p.stats()
        loss_by_lane[lane] = loss_by_lane.get(lane, 0) + st["loss_tokens"]
        tok_by_lane[lane] = tok_by_lane.get(lane, 0) + st["sequence_length"]
        for m in p.members:
            shard_ids.add(m["shard_id"])
            spans.append(f"{m['shard_id']}:{m['sample_id']}:{m['token_span'][0]}-{m['token_span'][1]}")
            members.append({"sample_id": m["sample_id"], "doc_id": m["doc_id"],
                            "shard_id": m["shard_id"], "lane": m["lane"],
                            "token_span": m["token_span"], "truncated": m["truncated"],
                            "repeated_pass": m["repeated_pass"]})
    return {
        "run_id": run_id,
        "branch_id": branch_id,
        "global_step": step_info["global_step"],
        "checkpoint_id": checkpoint_id,
        "rank": mb["rank"],
        "accumulation_index": mb["accumulation_index"],
        "microbatch_id": mb["microbatch_id"],
        "batch_hash": batch_fingerprint(packs),
        "loss_mask_hash": batch_fingerprint(packs)[:32],
        "packed_sample_ids": [m["sample_id"] for m in members],
        "shard_ids": sorted(shard_ids),
        "token_span_ids": spans,
        "attention_policy": "block_diagonal_causal",
        "position_policy": "per_sample_reset",
        "packing_policies": sorted(set(mb["policies"])),
        "mixture_lane": sorted(set(mb["lanes"])),
        "curriculum_stage": step_info["curriculum_stage"],
        "sequence_length": step_info["sequence_length"],
        "tokenizer_version": tokenizer_hash[:16],
        "dataloader_version": dataloader_version,
        "loss_tokens_by_lane": loss_by_lane,
        "tokens_by_lane": tok_by_lane,
        "n_sequences": len(packs),
        "members": members,
    }
