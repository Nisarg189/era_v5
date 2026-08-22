#!/usr/bin/env python3
"""Throughput accounting, measured three times for the same run.

A loader that reports a large tokens per second number can still deliver very
little learning.  Session 6 section 15 names the three figures that have to be
separated, and this module measures all three from real timings rather than
estimating any of them:

    raw tokens/sec        every position moved, padding included
    useful tokens/sec     only positions where loss is applied
    accepted tokens/sec   what survived the selector

The gap between the first and the second is padding and context.  The gap between
the second and the third is the price of selection.  Loader wait time is measured
as the share of wall clock spent building batches rather than computing on them,
which is the practical definition of compute sitting idle.
"""
import json, time


class Stopwatch:
    def __init__(self):
        self.totals = {}
        self._t0 = None
        self._key = None

    def start(self, key):
        self._key, self._t0 = key, time.perf_counter()
        return self

    def stop(self):
        if self._t0 is not None:
            self.totals[self._key] = self.totals.get(self._key, 0.0) + (time.perf_counter() - self._t0)
            self._t0 = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()

    def add(self, key, seconds):
        self.totals[key] = self.totals.get(key, 0.0) + seconds


class PerfMeter:
    def __init__(self):
        self.watch = Stopwatch()
        self.raw_tokens = 0
        self.loss_tokens = 0
        self.pad_tokens = 0
        self.context_tokens = 0
        self.accepted_tokens = 0
        self.offered_tokens = 0
        self.sequences = 0
        self.steps = 0
        self.events = {}

    def add_batch(self, packs):
        for p in packs:
            st = p.stats()
            self.raw_tokens += st["sequence_length"]
            self.loss_tokens += st["loss_tokens"]
            self.pad_tokens += st["pad_tokens"]
            self.context_tokens += st["context_only_tokens"]
            self.sequences += 1

    def mark(self, name, seconds):
        self.events[name] = round(seconds, 4)

    def report(self, opus_audit=None, wall=None):
        t = self.watch.totals
        wall = wall if wall is not None else sum(t.values())
        loader = t.get("loader", 0.0)
        compute = t.get("compute", 0.0)
        scoring = t.get("scoring", 0.0)
        denom = max(1e-9, wall)
        rep = {
            "wall_seconds": round(wall, 3),
            "loader_seconds": round(loader, 3),
            "compute_seconds": round(compute, 3),
            "selector_scoring_seconds": round(scoring, 3),
            "steps": self.steps,
            "sequences": self.sequences,
            "raw_token_positions": self.raw_tokens,
            "pad_positions": self.pad_tokens,
            "context_only_positions": self.context_tokens,
            "loss_bearing_tokens": self.loss_tokens,
            "accepted_tokens_after_opus": self.accepted_tokens,
            "offered_tokens_to_opus": self.offered_tokens,
            "raw_tokens_per_second": round(self.raw_tokens / denom, 1),
            "useful_loss_bearing_tokens_per_second": round(self.loss_tokens / denom, 1),
            "accepted_tokens_per_second": round(self.accepted_tokens / denom, 1),
            "packing_occupancy": round(1 - self.pad_tokens / max(1, self.raw_tokens), 6),
            "loss_coverage_of_positions": round(self.loss_tokens / max(1, self.raw_tokens), 6),
            "compute_idle_fraction": round(1 - compute / denom, 6),
            "loader_wait_fraction": round(loader / denom, 6),
            "selector_overhead_fraction": round(scoring / denom, 6),
            "seconds_per_step": round(wall / max(1, self.steps), 4),
            "phase_seconds": self.events,
        }
        if opus_audit:
            rep["opus_rejection_rate_by_lane"] = opus_audit.get("lane_rejection_rate", {})
            rep["opus_totals"] = opus_audit.get("totals", {})
            rep["opus_acceptance_rate"] = round(
                self.accepted_tokens / max(1, self.offered_tokens), 6)
        return rep

    def dump(self, path, opus_audit=None, wall=None):
        rep = self.report(opus_audit, wall)
        with open(path, "w") as f:
            json.dump(rep, f, indent=1)
        return rep


def reconstruct_check(report, ledger_records):
    """Recompute the headline throughput numbers from the ledger alone.

    Session 6 says that a throughput claim which cannot be reconstructed receives
    no credit, so the report is checked against an independent count taken from the
    consumption ledger rather than from the counters that produced it.
    """
    raw = sum(r["n_sequences"] * r["sequence_length"] for r in ledger_records)
    loss = sum(sum(r["loss_tokens_by_lane"].values()) for r in ledger_records)
    ok_raw = raw == report["raw_token_positions"]
    ok_loss = loss == report["loss_bearing_tokens"]
    return {
        "ledger_raw_token_positions": raw,
        "report_raw_token_positions": report["raw_token_positions"],
        "ledger_loss_bearing_tokens": loss,
        "report_loss_bearing_tokens": report["loss_bearing_tokens"],
        "raw_matches": bool(ok_raw),
        "loss_matches": bool(ok_loss),
        "ok": bool(ok_raw and ok_loss),
    }
