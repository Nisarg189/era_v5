#!/usr/bin/env python3
"""The learning ledger: what the model actually got out of each shard.

The consumption ledger records what was served.  This records what came back.  It
is built from per-token cross entropy captured during the run, attributed to the
shard, document, lane and token id that produced it, so nothing here is an average
standing in for a measurement.

Three signals are kept, at the three levels of detail Session 6 describes.

Full token traces for shards marked for tracing, which is how a difficulty pattern
inside a shard becomes visible rather than being hidden by the shard mean.

Aggregates for every shard: mean token loss, first and last exposure, the change
between them, the gradient norm of the steps the shard appeared in, and the effect
of a repeated pass.

Token clusters: the token ids that stay expensive.  For the Indic lane this is
where conjuncts and joiner sequences show up, and that pattern is the instruction
for what the next version of the corpus should contain.
"""
import json
import math


class LossTrace:
    """Accumulates per-token loss and attributes it back to its source."""

    def __init__(self, trace_shards=(), max_trace_rows=4000):
        self.trace_shards = set(trace_shards)
        self.max_trace_rows = max_trace_rows
        self.rows = []                     # full token trace, tiered
        self.by_shard = {}
        self.by_lane = {}
        self.by_token = {}
        self.by_shard_pass = {}
        self.step_grad_norm = {}
        self.shard_steps = {}

    def record_grad_norm(self, step, norm):
        self.step_grad_norm[step] = float(norm)

    def add(self, *, step, stage, shard_id, doc_id, lane, sample_id, repeated_pass,
            token_ids, losses, positions, checkpoint_id, opus_score, tokens_seen):
        """One member's graded tokens.  ``losses`` is aligned to ``token_ids``."""
        n = len(losses)
        if not n:
            return
        total = float(sum(losses))
        s = self.by_shard.setdefault(shard_id, {
            "shard_id": shard_id, "capability_lane": lane, "loss_sum": 0.0, "tokens": 0,
            "first_step": step, "last_step": step, "first_loss_sum": 0.0, "first_tokens": 0,
            "last_loss_sum": 0.0, "last_tokens": 0, "passes": {},
            "first_seen_at_tokens": tokens_seen, "checkpoints": set(),
        })
        s["loss_sum"] += total
        s["tokens"] += n
        s["checkpoints"].add(checkpoint_id)
        if step == s["first_step"]:
            s["first_loss_sum"] += total
            s["first_tokens"] += n
        if step > s["last_step"]:
            s["last_step"] = step
            s["last_loss_sum"], s["last_tokens"] = 0.0, 0
        if step == s["last_step"]:
            s["last_loss_sum"] += total
            s["last_tokens"] += n
        p = s["passes"].setdefault(str(repeated_pass), {"loss_sum": 0.0, "tokens": 0})
        p["loss_sum"] += total
        p["tokens"] += n
        self.shard_steps.setdefault(shard_id, set()).add(step)

        L = self.by_lane.setdefault(lane, {"loss_sum": 0.0, "tokens": 0})
        L["loss_sum"] += total
        L["tokens"] += n

        for tid, ls in zip(token_ids, losses):
            t = self.by_token.setdefault(int(tid), {"loss_sum": 0.0, "count": 0, "lanes": {}})
            t["loss_sum"] += float(ls)
            t["count"] += 1
            # per lane as well as overall, so "hardest tokens in the Indic lane" ranks
            # by what those tokens cost inside that lane rather than everywhere
            lt = t["lanes"].setdefault(lane, {"count": 0, "loss_sum": 0.0})
            lt["count"] += 1
            lt["loss_sum"] += float(ls)

        if shard_id in self.trace_shards and len(self.rows) < self.max_trace_rows:
            for tid, ls, pos in zip(token_ids, losses, positions):
                if len(self.rows) >= self.max_trace_rows:
                    break
                self.rows.append({
                    "global_step": step, "curriculum_stage": stage,
                    "shard_id": shard_id, "doc_id": doc_id, "sample_id": sample_id,
                    "capability_lane": lane, "token_id": int(tid),
                    "position_in_sequence": int(pos),
                    "cross_entropy": round(float(ls), 6),
                    "token_perplexity": round(math.exp(min(20.0, float(ls))), 4),
                    "model_age_tokens": tokens_seen, "repeated_pass": repeated_pass,
                    "checkpoint_before": checkpoint_id, "opus_score": opus_score,
                })

    # ------------------------------------------------------------------ reports
    def token_clusters(self, tok, lane=None, top=12, min_count=8, lane_min_count=15):
        """The token ids that stay expensive, decoded where the tokenizer allows.

        With ``lane`` set, both the filter and the ranking use that lane's own loss.
        Ranking a lane view by the overall mean would put tokens at the top that are
        expensive somewhere else and cheap here.
        """
        rows = []
        for tid, t in self.by_token.items():
            if lane:
                lt = t["lanes"].get(lane)
                if not lt or lt["count"] < lane_min_count:
                    continue          # a token seen three times is noise, not a pattern
                count, loss_sum = lt["count"], lt["loss_sum"]
            else:
                if t["count"] < min_count:
                    continue
                count, loss_sum = t["count"], t["loss_sum"]
            mean = loss_sum / count
            rows.append({
                "token_id": tid,
                "decoded": tok.decode([tid]) if tok else None,
                "scope": lane or "all lanes",
                "count": count,
                "mean_loss": round(mean, 5),
                "mean_perplexity": round(math.exp(min(20.0, mean)), 3),
                "dominant_lane": max(t["lanes"], key=lambda k: t["lanes"][k]["count"]),
            })
        rows.sort(key=lambda r: -r["mean_loss"])
        return rows[:top]

    def shard_report_cards(self, opus_scores=None, phase_of=None):
        """One row per shard: what it cost, what it changed, and a classification."""
        opus_scores = opus_scores or {}
        cards = []
        for shard_id, s in sorted(self.by_shard.items()):
            mean = s["loss_sum"] / max(1, s["tokens"])
            first = s["first_loss_sum"] / max(1, s["first_tokens"])
            last = s["last_loss_sum"] / max(1, s["last_tokens"])
            delta = last - first if s["last_step"] > s["first_step"] else 0.0
            norms = [self.step_grad_norm[st] for st in self.shard_steps.get(shard_id, ())
                     if st in self.step_grad_norm]
            grad = sum(norms) / len(norms) if norms else None
            passes = {k: round(v["loss_sum"] / max(1, v["tokens"]), 5)
                      for k, v in sorted(s["passes"].items())}
            repeat_effect = None
            keys = sorted(passes, key=int)
            if len(keys) > 1:
                repeat_effect = round(passes[keys[-1]] - passes[keys[0]], 5)
            if s["last_step"] == s["first_step"]:
                cls = "insufficient_exposure"
            elif delta < -0.02:
                cls = "useful"
            elif delta > 0.05:
                cls = "harmful"
            else:
                cls = "neutral"
            cards.append({
                "shard_id": shard_id,
                "capability_lane": s["capability_lane"],
                "phase": phase_of(s["first_step"]) if phase_of else None,
                "loss_bearing_tokens_seen": s["tokens"],
                "avg_token_loss": round(mean, 5),
                "avg_token_perplexity": round(math.exp(min(20.0, mean)), 3),
                "loss_at_first_exposure": round(first, 5),
                "loss_at_last_exposure": round(last, 5),
                "loss_delta_after_exposure": round(delta, 5),
                "mean_grad_norm_of_steps_seen": round(grad, 5) if grad is not None else None,
                "opus_mean_score": opus_scores.get(shard_id),
                "loss_by_repeated_pass": passes,
                "repeated_pass_effect": repeat_effect,
                "first_seen_at_tokens": s["first_seen_at_tokens"],
                "steps_seen": len(self.shard_steps.get(shard_id, ())),
                "classification": cls,
            })
        return cards

    def lane_summary(self):
        return {l: {"avg_token_loss": round(v["loss_sum"] / max(1, v["tokens"]), 5),
                    "avg_token_perplexity": round(math.exp(min(20.0, v["loss_sum"] / max(1, v["tokens"]))), 3),
                    "loss_bearing_tokens": v["tokens"]}
                for l, v in sorted(self.by_lane.items())}

    def dump(self, path, tok=None, opus_scores=None, phase_of=None):
        payload = {
            "storage_tiers": {
                "full_token_trace": sorted(self.trace_shards),
                "aggregates": "every shard",
                "note": "full traces are kept only for the shards named above, "
                        "which is the tiering Session 6 section 11 describes",
            },
            "lane_summary": self.lane_summary(),
            "shard_report_cards": self.shard_report_cards(opus_scores, phase_of),
            "hardest_token_clusters": self.token_clusters(tok),
            "hardest_token_clusters_indic": self.token_clusters(tok, lane="indic"),
            "token_trace_rows": len(self.rows),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        return payload


def write_token_trace(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
