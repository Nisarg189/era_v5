#!/usr/bin/env python3
"""The OPUS proxy: a reference model, and the excess loss score built from it.

The selector needs an opinion about whether a candidate is worth training on.  A
plain "the model finds this hard" signal rewards noise, because the hardest text in
any corpus is the most corrupted.  Excess loss against a reference model separates
the two: text that is hard for the current model but easy for the reference is
learnable, and text that is hard for both is usually just difficult or foreign.

    score = current model loss - reference model loss

The reference here is trained on English web text alone.  That choice is the point.
Session 5 measured an English-proxy selector rejecting 98.6 percent of the Hindi it
was offered, and this reproduces the effect from a real model instead of asserting
it: Devanagari is enormously expensive for a reference that has never seen the
script, so the excess loss goes deeply negative and the lane is rejected on merit
it does not lack.  The protected floor is the answer, and section 11 of the Session
5 report is the reason it exists.

Both models are frozen while scoring, so a score is cacheable by sample.  The
current model is not the live one but the snapshot taken at the last checkpoint,
which is what makes the same score reproducible after a resume.
"""
import time

import torch

SCORE_MAX_TOKENS = 192          # candidates are scored on their opening tokens


class ExcessLossScorer:
    def __init__(self, model_factory, reference_state, snapshot_state, proxy_version):
        self.reference = model_factory()
        self.reference.load_state_dict(reference_state)
        self.reference.eval()
        self.current = model_factory()
        self.current.load_state_dict(snapshot_state)
        self.current.eval()
        self.proxy_version = proxy_version
        self.snapshot_id = "init"
        self._ref_cache = {}
        self._cur_cache = {}
        self.n_scored = 0
        self.seconds = 0.0

    def set_snapshot(self, state, snapshot_id):
        """Point the scorer at a new checkpoint.  Clears only the current-model cache."""
        if snapshot_id == self.snapshot_id:
            return
        self.current.load_state_dict(state)
        self.current.eval()
        self.snapshot_id = snapshot_id
        self._cur_cache = {}

    @staticmethod
    def _prepare(item):
        from .packing import best_window
        off = best_window(item, SCORE_MAX_TOKENS)
        ids = item["ids"][off:off + SCORE_MAX_TOKENS]
        n = len(ids)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        pos = torch.arange(n).unsqueeze(0)
        seg = torch.zeros(1, n, dtype=torch.long)
        from .packing import _loss_from_roles
        lm = torch.tensor(_loss_from_roles(item["n"], item["roles"])[off:off + n]).unsqueeze(0)
        if lm.sum() == 0:
            lm = torch.ones(1, n, dtype=torch.long)
        return x, pos, seg, lm

    def _loss(self, model, cache, item):
        key = item["sample_id"]
        if key in cache:
            return cache[key]
        x, pos, seg, lm = self._prepare(item)
        if x.shape[1] < 2:
            cache[key] = 0.0
            return 0.0
        with torch.no_grad():
            _, _, mean = model.token_losses(x, pos, seg, lm)
        cache[key] = float(mean)
        return cache[key]

    def __call__(self, item):
        t0 = time.perf_counter()
        ref = self._loss(self.reference, self._ref_cache, item)
        cur = self._loss(self.current, self._cur_cache, item)
        self.n_scored += 1
        self.seconds += time.perf_counter() - t0
        return cur - ref, cur, ref
