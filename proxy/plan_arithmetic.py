#!/usr/bin/env python3
"""Check the plan's mixture against the real supply in the Session 5 inventory.

Nothing here is typed by hand except the plan's own shares, the epoch caps and
the acquisition programme the plan commits to. Inventory supply is summed from
inventory.json, extracted from the Session 5 dataset-inventory widget.

The point of this file is that no share in the plan can be asserted without the
supply arithmetic behind it being printed at the same time.

    python3 plan_arithmetic.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
# --- the plan ----------------------------------------------------------------
# The curriculum is the plan. Six stages, each with its own mixture, each spanning
# a fraction of the 15T budget. The headline pretraining mixture is not a separate
# decision: it is what these stages average to, which is why it is computed here
# rather than asserted.
STAGES = {
    "S0 seed":       (0.02, {"web": 52, "code": 14, "indic": 15, "stem": 10, "reason": 3,  "longctx": 2,  "agentic": 4}),
    "S1 general":    (0.32, {"web": 46, "code": 20, "indic": 14, "stem": 11, "reason": 4,  "longctx": 2,  "agentic": 3}),
    "S2 capability": (0.32, {"web": 30, "code": 31, "indic": 12, "stem": 12, "reason": 8,  "longctx": 4,  "agentic": 3}),
    "S3 reasoning":  (0.18, {"web": 22, "code": 26, "indic": 12, "stem": 9,  "reason": 17, "longctx": 11, "agentic": 3}),
    "S4 long-ctx":   (0.13, {"web": 18, "code": 24, "indic": 12, "stem": 6,  "reason": 11, "longctx": 26, "agentic": 3}),
    "S5 anneal":     (0.03, {"web": 8,  "code": 16, "indic": 26, "stem": 10, "reason": 18, "longctx": 10, "agentic": 12}),
}
LANES = ["web", "code", "indic", "stem", "reason", "longctx", "agentic"]

# The anneal is the last stage, held back and spent at a decayed learning rate.
# Session 5 puts this phase at about 2% of tokens, in a 1 to 5% band. The earlier
# draft of this plan used 8%, which is outside that band, and the proxy found no
# benefit from the reserve at all. It is set at the low end of the course's range.
ANNEAL_RESERVE_SHARE = 3

FLOORS = {"indic": 10, "reason": 5, "agentic": 2}

# Natural text holds up to about four epochs before returns decay (Muennighoff et
# al., Scaling Data-Constrained Language Models, NeurIPS 2023). Model-generated
# text is capped at two because its errors compound under repetition.
EPOCH_CAP = {"natural": 4, "synthetic": 2}
CAP = 4
BUDGET_T = 15.0

# Tokens the plan commits to adding, in trillions of UNIQUE tokens, with the
# mechanism. A lane whose demand exceeds inventory supply must either name one of
# these or have its share cut. There is no third option.
ADDED = {
    "indic": [(0.20, "harvest", "re-mine the full Common Crawl archive for Indian-language "
                                "pages, plus openly licensed public-sector text: court "
                                "judgments, Parliament and assembly proceedings, gazettes, "
                                "NCERT and state textbooks, NPTEL and SWAYAM transcripts, "
                                "Indian-language Wikipedia and Wikisource, National Digital "
                                "Library public-domain scans -> tier A, native text"),
              (0.185, "generate", "translation of curated English with IndicTrans2 into tier C, "
                                  "templated and self-instruct text into tier D, both at 2 epochs")],
    "stem": [(0.25, "mine", "classifier-selected STEM already inside DCLM and FineWeb-Edu, "
                            "relabelled rather than re-sourced"),
             (0.10, "harvest", "arXiv bulk access, PubMed Central open-access subset, "
                               "OpenStax and other open textbooks")],
    "longctx": [(0.60, "pack", "documents and repositories of 8K tokens or more already "
                               "inside the web and code lanes")],
    "reason": [(0.28, "generate", "teacher distillation filtered by verifiable rewards")],
    "agentic": [(0.10, "mine", "issue to pull-request to diff chains, CI logs, notebooks and "
                               "shell sessions inside the code lane"),
                (0.03, "generate", "execution-verified trajectories in tool sandboxes")],
}


def implied_mixture():
    """The headline mixture, averaged out of the stage schedule."""
    raw = {l: sum(sp * m[l] for sp, m in STAGES.values()) for l in LANES}
    # largest-remainder rounding so the printed shares still total 100
    out = {l: int(raw[l]) for l in LANES}
    rem = sorted(LANES, key=lambda l: raw[l] - int(raw[l]), reverse=True)
    for i in range(100 - sum(out.values())):
        out[rem[i]] += 1
    return raw, out


STEM_ROWS = {"proof-pile-2", "peS2o", "D4 STEM (V4)"}


def load_supply():
    rows = json.load(open(os.path.join(HERE, "inventory.json")))
    sup = {k: 0.0 for k in LANES}
    for r in rows:
        g, n, t = r["g"], r["name"], r.get("tokens", 0) / 1e12
        if g == "general":
            sup["stem" if n in STEM_ROWS else "web"] += t
        elif g == "long":
            sup["longctx"] += t
        elif g in ("code", "indic", "agentic", "reason"):
            sup[g] += t
    return sup, rows


def indic_tiers(rows):
    by = {r["name"]: r.get("tokens", 0) / 1e9 for r in rows if r["g"] == "indic"}
    return {
        "A verified native": (by["Sangraha (verified)"], "natural"),
        "B unverified crawl": (by["Sangraha (unverified)"] + by["IndicCorpV2"], "natural"),
        "C translated": (by["BPCC (parallel)"] + by["Samanantar"], "natural"),
        "D synthetic": (by["Sangraha (synthetic)"], "synthetic"),
    }


# Epoch AI's estimate of the effective stock of public human text, cited in
# Session 3. The inventory is a curated shortlist, not the world's supply, and the
# difference between the two is the difference between "hard to collect" and
# "does not exist".
WORLD_HUMAN_TEXT_T = 300.0


def rule(w=78):
    print("-" * w)


def main():
    sup, rows = load_supply()
    raw, MAIN = implied_mixture()
    ANNEAL = STAGES["S5 anneal"][1]
    assert sum(MAIN.values()) == 100
    for k, (sp, m) in STAGES.items():
        assert sum(m.values()) == 100, (k, sum(m.values()))
    assert abs(sum(sp for sp, _ in STAGES.values()) - 1.0) < 1e-9

    print("CURRICULUM — six stages, each with its own mixture (percent of that stage)\n")
    print(f"{'stage':<15}{'span':>7}{'tokens':>9}" + "".join(f"{l[:7]:>9}" for l in LANES))
    rule(87)
    for name, (span, m) in STAGES.items():
        print(f"{name:<15}{span:>6.0%}{span*BUDGET_T:>8.2f}T" + "".join(f"{m[l]:>9}" for l in LANES))
    rule(87)
    print(f"{'implied average':<15}{1:>6.0%}{BUDGET_T:>8.2f}T" + "".join(f"{raw[l]:>9.1f}" for l in LANES))
    print(f"{'headline mixture':<15}{'':>6}{'':>9}" + "".join(f"{MAIN[l]:>9}" for l in LANES))
    print("\n  The headline mixture is not chosen separately. It is what the stages average to.")

    print(f"\n\nMAIN PRETRAINING MIXTURE vs SUPPLY — {BUDGET_T}T tokens, "
          f"{CAP}-epoch repetition cap\n")
    print(f"{'lane':<9}{'share':>6}{'demand':>9}{'invent':>9}{'added':>8}{'effective':>11}"
          f"{'epochs':>8}  verdict")
    rule()
    tot_add = 0.0
    for lane, share in sorted(MAIN.items(), key=lambda x: -x[1]):
        d = share / 100 * BUDGET_T
        add = sum(t for t, _, _ in ADDED.get(lane, []))
        tot_add += add
        eff = sup[lane] + add
        ep = d / eff if eff else float("inf")
        v = "covered" if ep <= 1 else (f"repeat {ep:.1f}x" if ep <= CAP else "OVER CAP")
        print(f"{lane:<9}{share:>5}%{d:>8.2f}T{sup[lane]:>8.2f}T{add:>7.2f}T{eff:>10.2f}T"
              f"{ep:>8.1f}  {v}")
    rule()
    inv = sum(sup.values())
    print(f"{'total':<9}{100:>5}%{BUDGET_T:>8.2f}T{inv:>8.2f}T{tot_add:>7.2f}T"
          f"{inv+tot_add:>10.2f}T{BUDGET_T/(inv+tot_add):>8.1f}")
    print(f"\n  The {inv:.2f}T column is the Session 5 inventory: {len(rows)} named, filtered,")
    print(f"  licensed corpora. It is not the world's supply. Session 3 cites Epoch AI at")
    print(f"  ~{WORLD_HUMAN_TEXT_T:.0f}T tokens of public human text, so the inventory is "
          f"{inv/WORLD_HUMAN_TEXT_T:.1%} of it.")
    print(f"  That distinction decides which shortages are real:")
    print(f"    web      abundant in the world, the inventory row is a quality-filtered slice")
    print(f"    stem     abundant, mostly a labelling and extraction problem")
    print(f"    longctx  not a shortage at all, it is a packing decision over other lanes")
    print(f"    indic    genuinely scarce in the world, not merely uncollected")
    print(f"    reason   genuinely scarce, and the most legitimately generatable")
    print(f"    agentic  barely exists anywhere, must be mined and built")

    print(f"\n\nWHERE THE {tot_add:.2f}T OF NEW UNIQUE TOKENS COMES FROM\n")
    by_kind = {}
    for lane, items in ADDED.items():
        for t, kind, why in items:
            by_kind[kind] = by_kind.get(kind, 0) + t
            print(f"  {lane:<9}{t:>6.2f}T  {kind:<9}{why}")
    rule()
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {'':<9}{v:>6.2f}T  {k}")
    print(f"\n  Only {by_kind['generate']:.2f}T of the {tot_add:.2f}T is model-generated. The rest is\n"
          f"  harvested, mined or repacked. Every source is open, public-domain or already\n"
          f"  held: no purchase, licence negotiation or outside vendor is required.")

    print(f"\n\nINDIC SLOT — {MAIN['indic']}% = {MAIN['indic']/100*BUDGET_T:.2f}T, "
          f"by provenance tier\n")
    demand_b = MAIN["indic"] / 100 * BUDGET_T * 1000
    tiers = indic_tiers(rows)
    alloc = {
        "A verified native":  (tiers["A verified native"][0],  200.0, 4),
        "B unverified crawl": (tiers["B unverified crawl"][0],   0.0, 4),
        "C translated":       (tiers["C translated"][0],        60.0, 2),
        "D synthetic":        (tiers["D synthetic"][0],        125.0, 2),
    }
    print(f"{'tier':<20}{'existing':>10}{'new':>8}{'epochs':>8}{'tokens':>10}{'% of slot':>11}")
    rule(67)
    total = 0.0
    for name, (ex, new, ep) in alloc.items():
        t = (ex + new) * ep
        total += t
        print(f"{name:<20}{ex:>9.1f}B{new:>7.0f}B{ep:>7}x{t:>9.0f}B{t/demand_b:>10.1%}")
    rule(67)
    print(f"{'total':<20}{sum(a[0] for a in alloc.values()):>9.1f}B"
          f"{sum(a[1] for a in alloc.values()):>7.0f}B{'':>8}{total:>9.0f}B{total/demand_b:>10.1%}")
    nat = sum((ex + new) * ep for k, (ex, new, ep) in alloc.items() if k[0] in "AB")
    print(f"\n  Native text (tiers A and B):  {nat/demand_b:.0%} of the slot")
    print(f"  Model-generated (tier D):     "
          f"{alloc['D synthetic'][2]*sum(alloc['D synthetic'][:2])/demand_b:.0%} of the slot")
    print(f"\n  The lever is acquisition, not generation. 200B newly digitised native tokens")
    print(f"  move tier A from {tiers['A verified native'][0]*4/demand_b:.0%} to "
          f"{alloc['A verified native'][2]*sum(alloc['A verified native'][:2])/demand_b:.0%} "
          f"of the slot. No amount of synthetic text does that.")

    print(f"\n\nAGENTIC SLOT — {MAIN['agentic']}% = {MAIN['agentic']/100*BUDGET_T*1000:.0f}B "
          f"against {sup['agentic']*1000:.2f}B natural trajectories")
    print(f"  Natural supply covers "
          f"{sup['agentic']*1000*4/(MAIN['agentic']/100*BUDGET_T*1000):.1%} of the slot even at "
          f"4 epochs.")
    print(f"  A trajectory runs 15-30K tokens, so the "
          f"{ADDED['agentic'][1][0]*1000:.0f}B generated portion is roughly "
          f"{ADDED['agentic'][1][0]*1e12/22000/1e6:.1f}M validated trajectories.")

    print(f"\n\nPROTECTED FLOORS — the selector may not go below these in any batch\n")
    for k, v in FLOORS.items():
        print(f"  {k:<9}{v:>3}%   (headline share {MAIN[k]}%)")
    print(f"  {'total':<9}{sum(FLOORS.values()):>3}% of every batch is outside the selector's control")

    print(f"\n\nANNEAL — {ANNEAL_RESERVE_SHARE}% of the budget "
          f"= {ANNEAL_RESERVE_SHARE/100*BUDGET_T:.2f}T held back for the cooldown\n")
    print(f"{'lane':<10}{'anneal':>8}{'headline':>10}{'change':>9}")
    rule(37)
    for lane, share in sorted(ANNEAL.items(), key=lambda x: -x[1]):
        print(f"{lane:<10}{share:>7}%{MAIN[lane]:>9}%{share-MAIN[lane]:>+8}pp")


if __name__ == "__main__":
    main()
