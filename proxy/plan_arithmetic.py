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
BUDGET_T = 15.0                       # total pretraining tokens, in trillions

# --- the plan --------------------------------------------------------------
MAIN = {"web": 34, "code": 24, "indic": 13, "stem": 10, "longctx": 8, "reason": 8, "agentic": 3}
ANNEAL = {"indic": 26, "reason": 18, "code": 16, "agentic": 12, "stem": 10, "longctx": 10, "web": 8}
FLOORS = {"indic": 10, "agentic": 2, "reason": 5}
ANNEAL_RESERVE_SHARE = 8              # percent of the budget held back for the cooldown

# Natural text holds up to about four epochs before returns decay (Muennighoff et
# al., Scaling Data-Constrained Language Models, NeurIPS 2023). Model-generated
# text is capped at two because its errors compound under repetition.
EPOCH_CAP = {"natural": 4, "synthetic": 2}
CAP = 4

# Tokens the plan commits to adding, in trillions of UNIQUE tokens, with the
# mechanism. A lane whose demand exceeds inventory supply must either name one of
# these or have its share cut. There is no third option.
ADDED = {
    "indic": [(0.20, "acquire", "licensed news archives, book and textbook digitisation, "
                                "broadcast ASR, government and court records (Session 3 sourcing) "
                                "-> lands in tier A, native text"),
              (0.185, "generate", "verified machine translation into tier C and templated "
                                  "or self-instruct text into tier D, both capped at 2 epochs")],
    "stem": [(0.25, "mine", "classifier-selected STEM already inside DCLM and FineWeb-Edu, "
                            "relabelled rather than re-sourced"),
             (0.10, "acquire", "arXiv full text, PubMed, open textbooks")],
    "longctx": [(0.60, "pack", "documents and repositories of 8K tokens or more already "
                               "inside the web and code lanes")],
    "reason": [(0.25, "generate", "teacher distillation filtered by verifiable rewards")],
    "agentic": [(0.10, "mine", "issue to pull-request to diff chains, CI logs, notebooks and "
                               "shell sessions inside the code lane"),
                (0.03, "generate", "execution-verified trajectories in tool sandboxes")],
}

STEM_ROWS = {"proof-pile-2", "peS2o", "D4 STEM (V4)"}


def load_supply():
    rows = json.load(open(os.path.join(HERE, "inventory.json")))
    sup = {k: 0.0 for k in MAIN}
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


def rule(w=78):
    print("-" * w)


def main():
    sup, rows = load_supply()
    assert sum(MAIN.values()) == 100
    assert sum(ANNEAL.values()) == 100

    print(f"MAIN PRETRAINING MIXTURE — {BUDGET_T}T tokens, {CAP}-epoch repetition cap\n")
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
    print(f"{'total':<9}{100:>5}%{BUDGET_T:>8.2f}T{sum(sup.values()):>8.2f}T{tot_add:>7.2f}T"
          f"{sum(sup.values())+tot_add:>10.2f}T{BUDGET_T/(sum(sup.values())+tot_add):>8.1f}")

    print(f"\n\nWHERE THE {tot_add:.2f}T OF NEW UNIQUE TOKENS COMES FROM\n")
    by_kind = {}
    for lane, items in ADDED.items():
        for t, kind, why in items:
            by_kind[kind] = by_kind.get(kind, 0) + t
            print(f"  {lane:<9}{t:>6.2f}T  {kind:<9}{why}")
    rule()
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {'':<9}{v:>6.2f}T  {k}")
    print(f"\n  Only {by_kind['generate']:.2f}T of the {tot_add:.2f}T is model-generated. The rest is "
          f"acquired,\n  mined or repacked from data the programme already holds.")

    print(f"\n\nINDIC SLOT — {MAIN['indic']}% = {MAIN['indic']/100*BUDGET_T:.2f}T, "
          f"by provenance tier\n")
    demand_b = MAIN["indic"] / 100 * BUDGET_T * 1000
    tiers = indic_tiers(rows)
    # planned allocation: (existing unique, newly acquired or generated unique, epochs)
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
    assert abs(total - demand_b) < 30, (total, demand_b)
    nat = sum((ex + new) * ep for k, (ex, new, ep) in alloc.items() if k[0] in "AB")
    print(f"\n  Native text (tiers A and B):  {nat/demand_b:.0%} of the slot")
    print(f"  Translated (tier C):          {alloc['C translated'][2]*sum(alloc['C translated'][:2])/demand_b:.0%}")
    print(f"  Model-generated (tier D):     {alloc['D synthetic'][2]*sum(alloc['D synthetic'][:2])/demand_b:.0%}")
    print(f"\n  The lever is acquisition, not generation. 200B newly digitised native tokens")
    print(f"  move tier A from {tiers['A verified native'][0]*4/demand_b:.0%} to "
          f"{alloc['A verified native'][2]*sum(alloc['A verified native'][:2])/demand_b:.0%} of the slot. "
          f"No amount of synthetic text does that.")

    print(f"\n\nAGENTIC SLOT — {MAIN['agentic']}% = {MAIN['agentic']/100*BUDGET_T*1000:.0f}B "
          f"against {sup['agentic']*1000:.2f}B natural trajectories")
    print(f"  Natural supply covers {sup['agentic']*1000*4/(MAIN['agentic']/100*BUDGET_T*1000):.1%} "
          f"of the slot even at 4 epochs.")
    print(f"  The lane is therefore mined and built, not collected. A trajectory runs 15-30K")
    print(f"  tokens, so the {ADDED['agentic'][1][0]*1000:.0f}B generated portion is roughly "
          f"{ADDED['agentic'][1][0]*1e12/22000/1e6:.1f}M validated trajectories.")

    print(f"\n\nPROTECTED FLOORS — the selector may not go below these in any batch\n")
    for k, v in FLOORS.items():
        print(f"  {k:<9}{v:>3}%   (main-run share {MAIN[k]}%)")
    print(f"  {'total':<9}{sum(FLOORS.values()):>3}% of every batch is outside the selector's control")

    print(f"\n\nANNEAL — {ANNEAL_RESERVE_SHARE}% of the budget "
          f"= {ANNEAL_RESERVE_SHARE/100*BUDGET_T:.2f}T held back for the cooldown\n")
    print(f"{'lane':<10}{'anneal':>8}{'main':>7}{'change':>9}")
    rule(34)
    for lane, share in sorted(ANNEAL.items(), key=lambda x: -x[1]):
        print(f"{lane:<10}{share:>7}%{MAIN[lane]:>6}%{share-MAIN[lane]:>+8}pp")


if __name__ == "__main__":
    main()
