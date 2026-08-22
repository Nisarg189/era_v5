#!/usr/bin/env python3
"""OPUS: the batch selector, and the audit trail its decisions leave behind.

OPUS is the Session 5 selector, not the Anthropic model.  It scores candidate
batches before they are served and keeps the useful fraction.  Because it sits
inside the data path, its decisions are training events and belong in the ledger
alongside the batches themselves.

The score is excess loss, computed from two real models rather than asserted:

    score = mean cross entropy of the current model on the candidate's graded tokens
          - mean cross entropy of a reference model on the same tokens

The reference model is trained on English web text only.  This is deliberate.  It
reproduces the proxy bias Session 5 measured rather than describing it: text the
English reference already finds easy scores high when the current model still finds
it hard, and text the English reference finds impossible, such as Devanagari, scores
far below the accept threshold no matter how much the model has left to learn from
it.  Section 15 of the Session 5 report measured an English-only selector rejecting
98.6 percent of the Hindi it was offered.  The same thing happens here, for the same
reason, and the protected floor is what stops it.

Four outcomes are recorded, and rejected candidates are kept rather than discarded:

    accepted                 served, gradient bearing
    rejected                 not served now, and the reason is recorded
    deferred                 explicitly held for a later phase
    protected_floor_override rejected on score, served because a floor required it
"""
import json

ACCEPT_THRESHOLD = 0.0      # excess loss at or above this is worth training on
DEFER_BAND = 1.00           # how far below the threshold still counts as "later"


class Opus:
    def __init__(self, scorer, floors, proxy_version, accept_threshold=ACCEPT_THRESHOLD,
                 defer_band=DEFER_BAND):
        """``scorer(item) -> (score, current_loss, reference_loss)``."""
        self.scorer = scorer
        self.floors = dict(floors)
        self.proxy_version = proxy_version
        self.accept_threshold = accept_threshold
        self.defer_band = defer_band
        self.decisions = []
        self.counts = {"accepted": 0, "rejected": 0, "deferred": 0, "protected_floor_override": 0}
        self.by_lane = {}
        self._seq = 0

    def _record(self, item, status, score, cur, ref, reason, stage, step,
                scoring_checkpoint, floor_override=False, quota_topup=False):
        self._seq += 1
        d = {
            "candidate_id": f"cand-{self._seq:06d}",
            "shard_ids": [item["shard_id"]],
            "sample_id": item["sample_id"],
            "capability_lane": item["lane"],
            "curriculum_stage": stage,
            "global_step": step,
            "scoring_checkpoint": scoring_checkpoint,
            "proxy_version": self.proxy_version,
            "opus_score": round(float(score), 6),
            "current_model_loss": round(float(cur), 6),
            "reference_model_loss": round(float(ref), 6),
            "status": status,
            "rejection_reason": reason,
            "protected_floor_override": bool(floor_override),
            "quota_topup": bool(quota_topup),
            "effective_token_estimate": int(max(0.0, min(1.0, (score + 1.0) / 2.0)) * item["n"]),
        }
        self.decisions.append(d)
        self.counts[status if not floor_override else "protected_floor_override"] += 1
        if quota_topup:
            self.counts["quota_topup"] = self.counts.get("quota_topup", 0) + 1
        lane = self.by_lane.setdefault(item["lane"], {"offered": 0, "accepted": 0, "rejected": 0,
                                                      "deferred": 0, "floor_rescued": 0,
                                                      "quota_topup": 0})
        lane["offered"] += 1
        if quota_topup:
            lane["quota_topup"] += 1
            lane["accepted"] += 1
        elif floor_override:
            lane["floor_rescued"] += 1
            lane["accepted"] += 1
        elif status == "accepted":
            lane["accepted"] += 1
        elif status == "deferred":
            lane["deferred"] += 1
        else:
            lane["rejected"] += 1
        return d

    def select(self, candidates, need_tokens, stage, step, scoring_checkpoint,
               floor_tokens):
        """Choose items per lane until each lane's token budget is met.

        Budgets are in tokens rather than items because the mixture is a promise
        about tokens.  The selector works inside each lane's budget and never
        across lanes, which is the structural rule Session 5's experiments forced
        into the plan.  The floor portion of a lane is filled before the selector
        is consulted, so the floor is not something the selector can reject.
        """
        chosen, decisions = {}, []
        for lane, budget in need_tokens.items():
            if budget <= 0:
                continue
            pool = [c for c in candidates if c["lane"] == lane]
            if not pool:
                continue
            scored = []
            for item in pool:
                score, cur, ref = self.scorer(item)
                scored.append((score, cur, ref, item))
            scored.sort(key=lambda t: -t[0])

            taken, taken_tokens, used = [], 0, set()
            floor_budget = min(float(floor_tokens.get(lane, 0)), budget)

            # 1. the protected floor, filled from the top of the lane regardless of score
            for score, cur, ref, item in scored:
                if taken_tokens >= floor_budget:
                    break
                override = score < self.accept_threshold
                d = self._record(item, "accepted", score, cur, ref,
                                 "below threshold, retained by protected floor" if override else None,
                                 stage, step, scoring_checkpoint, floor_override=override)
                decisions.append(d)
                taken.append((item, d))
                taken_tokens += item["n"]
                used.add(id(item))

            # 2. the remainder of the lane budget, decided on score
            for score, cur, ref, item in scored:
                if id(item) in used:
                    continue
                if taken_tokens >= budget:
                    d = self._record(item, "rejected", score, cur, ref,
                                     "quota_pressure", stage, step, scoring_checkpoint)
                    decisions.append(d)
                    continue
                if score >= self.accept_threshold:
                    d = self._record(item, "accepted", score, cur, ref, None,
                                     stage, step, scoring_checkpoint)
                    taken.append((item, d))
                    taken_tokens += item["n"]
                    used.add(id(item))
                elif score >= self.accept_threshold - self.defer_band:
                    d = self._record(item, "deferred", score, cur, ref,
                                     "low_proxy_utility_for_this_phase", stage, step,
                                     scoring_checkpoint)
                else:
                    d = self._record(item, "rejected", score, cur, ref,
                                     "low_proxy_utility", stage, step, scoring_checkpoint)
                decisions.append(d)

            # 3. a lane short of its budget is topped up rather than left underfilled:
            #    the mixture is a promise about how many tokens the lane receives, and
            #    Session 5 measured what happens when the selector is allowed to break it
            if taken_tokens < budget:
                for score, cur, ref, item in scored:
                    if taken_tokens >= budget:
                        break
                    if id(item) in used:
                        continue
                    d = self._record(item, "accepted", score, cur, ref,
                                     "lane_underfilled_after_selection", stage, step,
                                     scoring_checkpoint, quota_topup=True)
                    decisions.append(d)
                    taken.append((item, d))
                    taken_tokens += item["n"]
                    used.add(id(item))
            chosen[lane] = taken
        return chosen, decisions

    def audit(self):
        """The four ledgers the session asks for, plus the lane view."""
        buckets = {"accepted": [], "rejected": [], "deferred": [], "protected": [],
                   "quota_topup": []}
        for d in self.decisions:
            if d.get("quota_topup"):
                buckets["quota_topup"].append(d)
            elif d["protected_floor_override"]:
                buckets["protected"].append(d)
            elif d["status"] == "accepted":
                buckets["accepted"].append(d)
            elif d["status"] == "deferred":
                buckets["deferred"].append(d)
            else:
                buckets["rejected"].append(d)
        reasons = {}
        for d in self.decisions:
            if d["rejection_reason"]:
                reasons[d["rejection_reason"]] = reasons.get(d["rejection_reason"], 0) + 1
        return {
            "proxy_version": self.proxy_version,
            "accept_threshold": self.accept_threshold,
            "defer_band": self.defer_band,
            "totals": {k: len(v) for k, v in buckets.items()},
            "by_lane": self.by_lane,
            "rejection_reasons": reasons,
            "lane_rejection_rate": {
                lane: round(v["rejected"] / max(1, v["offered"]), 4)
                for lane, v in self.by_lane.items()
            },
        }

    def dump(self, path):
        with open(path, "w") as f:
            json.dump({"audit": self.audit(), "decisions": self.decisions}, f, indent=1)
        return path
