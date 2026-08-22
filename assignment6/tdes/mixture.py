#!/usr/bin/env python3
"""Compile the Session 5 curriculum into per-step lane quotas.

Session 5 states the plan in human terms: six stages, each with its own mixture,
a fifteen percent protected floor, and blended transitions.  A dataloader cannot
act on that.  It needs to know, at step 41, how many sequences of each lane to
serve.  This module performs that conversion.

The stage table below is the Session 5 plan verbatim, including the sequence
lengths, scaled from a nine trillion token run down to the demonstration budget.
Only the absolute token counts change; every share, floor and boundary is the
plan's own number.

Three things make this harder than multiplying a percentage by a batch size.

Whole sequences.  A lane owed 3.84 sequences cannot be served 3.84 sequences, so
the fractional part is carried forward and the average holds over many steps.  The
carried remainder is loader state, which means it has to survive a crash.

Warmup.  A stage boundary does not switch the mixture in one step.  The two
mixtures blend across a warmup band, because a hard distribution change moves the
gradients and Session 5 records V4 losing runs to exactly that.

Floors.  A protected lane keeps its share whether or not the selector wants it.
The floor is applied after the mixture and before the selector, so the selector
never sees the floor portion as something it is allowed to reject.
"""
import json

LANES = ["web", "code", "indic", "stem", "reasoning", "agentic"]

# Session 5, section 05.  Shares are percentages of that stage's own tokens.
# seq_scale is the stage's sequence length as a fraction of the run's maximum,
# preserving the plan's 4K / 8K / 32K progression at demonstration scale.
STAGES = [
    {"stage": "S0-seed",         "span": (0.00, 0.02), "seq_scale": 0.25,
     "mixture": {"web": .60, "code": .14, "indic": .14, "stem": .09, "reasoning": .02, "agentic": .01}},
    {"stage": "S1-general",      "span": (0.02, 0.34), "seq_scale": 0.25,
     "mixture": {"web": .56, "code": .22, "indic": .13, "stem": .07, "reasoning": .01, "agentic": .01}},
    {"stage": "S2-capability",   "span": (0.34, 0.66), "seq_scale": 0.50,
     "mixture": {"web": .36, "code": .33, "indic": .12, "stem": .15, "reasoning": .03, "agentic": .01}},
    {"stage": "S3-reasoning",    "span": (0.66, 0.84), "seq_scale": 0.50,
     "mixture": {"web": .26, "code": .31, "indic": .13, "stem": .12, "reasoning": .17, "agentic": .01}},
    {"stage": "S4-long-context", "span": (0.84, 0.97), "seq_scale": 1.00,
     "mixture": {"web": .31, "code": .35, "indic": .14, "stem": .09, "reasoning": .10, "agentic": .01}},
    {"stage": "S5-anneal",       "span": (0.97, 1.00), "seq_scale": 0.50,
     "mixture": {"web": .10, "code": .20, "indic": .28, "stem": .12, "reasoning": .22, "agentic": .08}},
]

# Session 5, section 11.  Fifteen percent of every batch sits outside the selector.
PROTECTED_FLOORS = {"indic": 0.10, "reasoning": 0.04, "agentic": 0.01}

# Session 5, section 12.  Held back and spent once, in the anneal stage.
ANNEAL_RESERVE_FRACTION = 0.03

WARMUP_FRACTION = 0.30      # of the shorter adjacent stage, the plan's blended band


def _blend(a, b, t):
    return {l: a.get(l, 0.0) * (1 - t) + b.get(l, 0.0) * t for l in LANES}


class Schedule:
    """The compiled timeline.  Answers 'what mixture applies at token N'."""

    def __init__(self, total_tokens, max_seq_len):
        self.total_tokens = total_tokens
        self.max_seq_len = max_seq_len
        self.stages = []
        for i, s in enumerate(STAGES):
            lo, hi = s["span"]
            start, end = int(lo * total_tokens), int(hi * total_tokens)
            seq_len = max(32, int(round(max_seq_len * s["seq_scale"] / 32)) * 32)
            span_tokens = max(1, end - start)
            self.stages.append({
                "stage": s["stage"], "token_start": start, "token_end": end,
                "tokens": span_tokens, "sequence_length": seq_len,
                "mixture": dict(s["mixture"]), "protected_floors": dict(PROTECTED_FLOORS),
                "warmup_tokens": int(span_tokens * WARMUP_FRACTION),
                "index": i,
            })

    def stage_at(self, tokens_seen):
        for s in self.stages:
            if tokens_seen < s["token_end"]:
                return s
        return self.stages[-1]

    def mixture_at(self, tokens_seen):
        """The effective mixture, with the stage boundary blended.

        Returns (mixture, stage_record, warmup_position) where warmup_position is
        0.0 outside a transition and rises to 1.0 across the band.
        """
        s = self.stage_at(tokens_seen)
        if s["index"] == 0:
            return dict(s["mixture"]), s, 0.0
        prev = self.stages[s["index"] - 1]
        band = min(s["warmup_tokens"], prev["warmup_tokens"]) or s["warmup_tokens"]
        into = tokens_seen - s["token_start"]
        if band <= 0 or into >= band:
            return dict(s["mixture"]), s, 0.0
        t = max(0.0, into / band)
        return _blend(prev["mixture"], s["mixture"], t), s, round(1.0 - t, 4)

    def anneal_reserve_tokens(self):
        return int(self.total_tokens * ANNEAL_RESERVE_FRACTION)

    def to_json(self):
        return {"total_tokens": self.total_tokens, "max_sequence_length": self.max_seq_len,
                "warmup_fraction": WARMUP_FRACTION,
                "anneal_reserve_fraction": ANNEAL_RESERVE_FRACTION,
                "anneal_reserve_tokens": self.anneal_reserve_tokens(),
                "protected_floors": PROTECTED_FLOORS, "stages": self.stages}


class QuotaAllocator:
    """Turns a mixture into whole sequence counts, carrying the remainders.

    The carried remainder is the reason this class exists and the reason a naive
    resume produces a different stream: the allocation at step N depends on every
    step before it.  ``state`` and ``load_state`` put that dependency into the
    checkpoint where it belongs.
    """

    def __init__(self, floors=None):
        self.carry = {l: 0.0 for l in LANES}
        self.floors = dict(PROTECTED_FLOORS if floors is None else floors)
        self.planned = {l: 0 for l in LANES}       # sequences owed, cumulative
        self.floor_forced = {l: 0 for l in LANES}  # sequences the floor rescued

    def allocate(self, mixture, n_sequences):
        """Return (counts, floor_sequences).  counts sums to exactly n_sequences.

        Floors are applied by raising the lane's effective share before allocation,
        and the extra is taken proportionally from the lanes that have no floor.
        A floor smaller than one whole sequence still has to appear sometimes, so
        it is carried as a fraction rather than rounded away to nothing.
        """
        eff = {l: max(mixture.get(l, 0.0), self.floors.get(l, 0.0)) for l in LANES}
        over = sum(eff.values()) - 1.0
        if over > 1e-9:
            free = {l: mixture.get(l, 0.0) for l in LANES if l not in self.floors}
            pool = sum(free.values())
            for l, v in free.items():
                eff[l] = max(0.0, v - over * (v / pool)) if pool > 0 else v
        total = sum(eff.values()) or 1.0
        eff = {l: v / total for l, v in eff.items()}

        floor_seqs = {l: self.floors.get(l, 0.0) * n_sequences for l in LANES}
        want = {l: eff[l] * n_sequences + self.carry[l] for l in LANES}
        counts = {l: int(want[l]) for l in LANES}
        rem = n_sequences - sum(counts.values())
        if rem > 0:
            order = sorted(LANES, key=lambda l: -(want[l] - counts[l]))
            for l in order[:rem]:
                counts[l] += 1
        elif rem < 0:
            order = sorted([l for l in LANES if counts[l] > 0],
                           key=lambda l: (want[l] - counts[l]))
            for l in order[:-rem]:
                counts[l] -= 1
        for l in LANES:
            self.carry[l] = want[l] - counts[l]
            self.planned[l] += counts[l]
            if counts[l] and eff[l] > mixture.get(l, 0.0) + 1e-9:
                self.floor_forced[l] += 1
        return counts, floor_seqs

    def state(self):
        return {"carry": dict(self.carry), "planned": dict(self.planned),
                "floor_forced": dict(self.floor_forced)}

    def load_state(self, st):
        self.carry = dict(st["carry"])
        self.planned = dict(st["planned"])
        self.floor_forced = dict(st.get("floor_forced", {l: 0 for l in LANES}))


def compliance(planned_tokens, actual_tokens):
    """Planned share against actual share, per lane.  The mixture compliance check."""
    tp, ta = max(1, sum(planned_tokens.values())), max(1, sum(actual_tokens.values()))
    rows = {}
    for l in LANES:
        p, a = planned_tokens.get(l, 0) / tp, actual_tokens.get(l, 0) / ta
        rows[l] = {"planned_share": round(p, 5), "actual_share": round(a, 5),
                   "planned_tokens": planned_tokens.get(l, 0),
                   "actual_tokens": actual_tokens.get(l, 0),
                   "abs_error": round(abs(p - a), 5)}
    return rows
