#!/usr/bin/env python3
"""The shard registry, the admission gate and the evaluation firewall.

One registry holds every shard the system knows about, at three permission levels:

    train        may enter batches and carry gradient
    validation   may be read to measure, never carries gradient
    test         never_train, blocked at the gate and never read during training

Evaluation data is registered precisely so that it can be recognised and refused.
A shard that is unknown to the registry cannot be checked against anything.

The gate runs two independent checks.  The first is a document check: does this
shard carry a valid licence, a matching tokenizer hash, a known cleaning lineage
and a completed dedup pass.  The second is the firewall: does any sample in it
overlap the held-out sets, by exact content hash or by token n-gram fingerprint.
Either failure blocks the shard and writes a rejection event.
"""
import json, os

from .shards import sample_fingerprint, verify_shard

ALLOWED_LICENCE_TIERS = {"A", "B"}
CONTAMINATION_JACCARD = 0.35     # deliberately lower than the dedup threshold
CONTAMINATION_MIN_SHARED = 4     # a handful of shared n-grams is not an overlap


class Registry:
    def __init__(self, shard_dir, tokenizer_hash):
        self.shard_dir = shard_dir
        self.tokenizer_hash = tokenizer_hash
        self.shards = {}          # shard_id -> manifest
        self.events = []          # every admission decision, in order
        self._eval_exact = {}     # sha256 of a held-out sample -> its shard and id
        self._eval_prints = []    # (shard_id, sample_id, fingerprint set)
        self._never_train_docs = set()

    # -------------------------------------------------------------- registration
    def register_heldout(self, manifest, tokens):
        """Record a validation or test shard, and index it for the firewall."""
        import hashlib
        self.shards[manifest["shard_id"]] = manifest
        for s in manifest["samples"]:
            span = tokens[s["start"]:s["end"]].tobytes()
            self._eval_exact[hashlib.sha256(span).hexdigest()] = (manifest["shard_id"], s["sample_id"])
            self._eval_prints.append((manifest["shard_id"], s["sample_id"],
                                      sample_fingerprint(tokens, s)))
            if manifest.get("never_train"):
                self._never_train_docs.add(s["doc_id"])
        self._event("registered", manifest["shard_id"], True,
                    f"{manifest['split']} shard indexed for the firewall, "
                    f"never_train={bool(manifest.get('never_train'))}")
        return manifest

    def never_train_docs(self):
        return set(self._never_train_docs)

    # -------------------------------------------------------------- the gate
    def _event(self, kind, shard_id, ok, detail, extra=None):
        e = {"event": kind, "shard_id": shard_id, "result": "PASS" if ok else "BLOCK",
             "detail": detail}
        if extra:
            e.update(extra)
        self.events.append(e)
        return e

    def document_checks(self, manifest):
        """Licence, tokenizer, lineage, dedup and content hash.  Returns list of failures."""
        fails = []
        if manifest.get("license_tier") not in ALLOWED_LICENCE_TIERS:
            fails.append(f"licence tier {manifest.get('license_tier')!r} is not admissible")
        if manifest.get("tokenizer_hash") != self.tokenizer_hash:
            fails.append("tokenizer hash does not match the frozen tokenizer")
        if not manifest.get("cleaning_pipeline_hash"):
            fails.append("cleaning lineage is unknown")
        if manifest.get("dedup_status") != "pass":
            fails.append(f"dedup status is {manifest.get('dedup_status')!r}")
        if not manifest.get("capability_lane"):
            fails.append("no capability lane, the scheduler could not place this shard")
        ok, detail = verify_shard(self.shard_dir, manifest)
        if not ok:
            fails.append(f"content hash check failed: {detail}")
        return fails

    def firewall_check(self, manifest, tokens):
        """Overlap against every registered held-out sample.  Returns list of hits."""
        import hashlib
        hits = []
        for s in manifest["samples"]:
            span = tokens[s["start"]:s["end"]].tobytes()
            exact = self._eval_exact.get(hashlib.sha256(span).hexdigest())
            if exact:
                hits.append({"sample_id": s["sample_id"], "match": "exact_content_hash",
                             "held_out_shard": exact[0], "held_out_sample": exact[1]})
                continue
            fp = sample_fingerprint(tokens, s)
            if not fp:
                continue
            for shard_id, sample_id, prev in self._eval_prints:
                shared = len(fp & prev)
                if shared < CONTAMINATION_MIN_SHARED:
                    continue
                if shared / len(fp | prev) >= CONTAMINATION_JACCARD:
                    hits.append({"sample_id": s["sample_id"], "match": "ngram_fingerprint",
                                 "shared_ngrams": shared,
                                 "jaccard": round(shared / len(fp | prev), 4),
                                 "held_out_shard": shard_id, "held_out_sample": sample_id})
                    break
        return hits

    def screen_documents(self, tok, docs, eos_id):
        """Check documents against the held-out index before any shard is built.

        Screening belongs here rather than after sharding.  A shard built from
        unscreened documents has to be thrown away and rebuilt when a hit is found,
        and because deduplication decides which near duplicates survive, removing
        one contaminated document changes which documents remain and can expose a
        second one.  Screening the full document set once ends that cascade.

        Returns (clean_docs, rejected) where each rejection names what it matched.
        """
        from .shards import tokenize_documents
        tokens, samples = tokenize_documents(tok, docs, eos_id)
        probe = {"shard_id": "__screen__", "samples": samples}
        hits = self.firewall_check(probe, tokens)
        bad = {h["sample_id"]: h for h in hits}
        clean = [d for d in docs if d["doc_id"] not in bad]
        for h in hits:
            self._event("document_screened_out", "__screen__", False,
                        f"{h['sample_id']} matched {h['held_out_sample']} by {h['match']}",
                        {"hit": h})
        return clean, list(bad.values())

    def admit(self, manifest, tokens):
        """Run both checks.  Admitted shards enter the registry, blocked ones do not."""
        fails = self.document_checks(manifest)
        hits = self.firewall_check(manifest, tokens)
        if hits:
            fails.append(f"evaluation overlap on {len(hits)} sample(s)")
        if fails:
            self._event("admission_blocked", manifest["shard_id"], False, "; ".join(fails),
                        {"firewall_hits": hits})
            return False, fails, hits
        manifest["admitted"] = True
        self.shards[manifest["shard_id"]] = manifest
        self._event("admitted", manifest["shard_id"], True,
                    f"{manifest['n_tokens']} tokens, lane {manifest['capability_lane']}, "
                    f"content hash verified")
        return True, [], []

    # -------------------------------------------------------------- views
    def trainable(self):
        """Shards the stream may draw from.  Probe shards are registered and audited
        but never trained on, so an admission test cannot alter the run it tests."""
        return [m for m in self.shards.values()
                if m.get("admitted") and m.get("split") == "train"
                and not m.get("never_train") and not m.get("probe")]

    def by_split(self, split):
        return [m for m in self.shards.values() if m.get("split") == split]

    def by_lane(self, lane):
        return [m for m in self.trainable() if m["capability_lane"] == lane]

    def dump(self, path):
        with open(path, "w") as f:
            json.dump({"tokenizer_hash": self.tokenizer_hash,
                       "shards": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                                  for k, v in self.shards.items()},
                       "events": self.events}, f, indent=1)
        return path
