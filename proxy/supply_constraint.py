#!/usr/bin/env python3
"""What the mixture can be when no new data may be collected.

The programme has no budget for crawling or for optical character recognition, so
every token must come from a dataset that has already been published and can simply
be downloaded. This file works out what that constraint costs.

It computes, for each lane, the ceiling on tokens available at the repetition caps,
then asks what token budget that ceiling can actually support without the mixture
collapsing into general web.

    python3 supply_constraint.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_NATURAL, CAP_SYNTHETIC = 4, 2

# Datasets already published and downloadable, beyond the Session 5 inventory.
# These are ESTIMATES of unique tokens surviving deduplication against the
# inventory, not measured counts. Section 17 of the README carries the rule for
# what happens if a measurement comes in below them.
EXTRA = {
    "indic": [("FineWeb-2, Indian-language subsets", 60),
              ("CulturaX, Indian-language subsets", 40),
              ("HPLT v2, Indian-language subsets", 12),
              ("Varta Indian news corpus", 5),
              ("Indian-language Wikipedia dumps", 3)],
    "stem":  [("FineMath", 34),
              ("RedPajama arXiv", 28),
              ("PubMed Central open access", 26),
              ("StackExchange", 20),
              ("OpenWebMath", 15)],
    "reason": [("OpenR1-Math-220k and open R1 distillations", 30),
               ("Nemotron and AM reasoning post-training sets", 20)],
    "agentic": [("the-stack-github-issues", 15)],
    "code": [], "web": [],
}

# Long context is deliberately absent. It is not a lane with its own supply: a
# long-context corpus is built by choosing long documents out of the web and code
# lanes and packing them. Counting it separately double-counts those tokens. It is
# handled in the plan as a sequence-length policy applied during one stage.
LANES = ["web", "code", "indic", "stem", "reason", "agentic"]
STEM_ROWS = {"proof-pile-2", "peS2o", "D4 STEM (V4)"}


def inventory():
    rows = json.load(open(os.path.join(HERE, "inventory.json")))
    nat = {k: 0.0 for k in LANES}
    syn = {k: 0.0 for k in LANES}
    for r in rows:
        g, n, t = r["g"], r["name"], r.get("tokens", 0) / 1e9
        if g == "general":
            nat["stem" if n in STEM_ROWS else "web"] += t
        elif g == "long":
            nat["web"] += t                      # packed web and code, folded back
        elif g in ("code", "stem", "reason", "agentic"):
            nat[g] += t
        elif g == "indic":
            (syn if "synthetic" in n else nat)["indic"] += t
    return nat, syn


def rule(w=76):
    print("-" * w)


def main():
    nat, syn = inventory()
    extra = {k: sum(t for _, t in v) for k, v in EXTRA.items()}

    print("SUPPLY UNDER THE CONSTRAINT: published datasets only, no crawling, no OCR\n")
    print(f"{'lane':<9}{'inventory':>11}{'+published':>12}{'unique':>10}{'cap':>6}{'ceiling':>10}")
    rule(58)
    ceiling = {}
    for l in LANES:
        uniq = nat[l] + extra[l]
        c = uniq * CAP_NATURAL + syn[l] * CAP_SYNTHETIC
        ceiling[l] = c
        capnote = f"{CAP_NATURAL}x" + ("/2x" if syn[l] else "")
        print(f"{l:<9}{nat[l]+syn[l]:>10.1f}B{extra[l]:>11.0f}B{uniq:>9.1f}B{capnote:>6}{c:>9.0f}B")
    rule(58)
    print(f"{'total':<9}{sum(nat.values())+sum(syn.values()):>10.1f}B{sum(extra.values()):>11.0f}B"
          f"{'':>10}{'':>6}{sum(ceiling.values()):>9.0f}B")

    print("\n  Additional published sources counted above (estimated unique after dedup):")
    for l, items in EXTRA.items():
        for name, t in items:
            print(f"    {l:<8}{t:>5}B  {name}")

    print("\n\nWHAT BUDGET THIS SUPPORTS\n")
    print("  The ceiling for each scarce lane, expressed as the largest share of a run")
    print("  of a given size that the lane can actually fill:\n")
    print(f"{'budget':>8}" + "".join(f"{l:>10}" for l in LANES))
    rule(68)
    for B in [15.0, 12.0, 9.0, 8.0]:
        row = "".join(f"{ceiling[l]/1000/B:>9.1%}" for l in LANES)
        print(f"{B:>7.0f}T{row}")
    rule(68)
    print("\n  Read the Indic column. At a 15T budget the Indic lane cannot exceed 8.4%")
    print("  however it is arranged, because that is all the Indic text there is to")
    print("  download. Everything the mixture cannot fill with a scarce lane it must")
    print("  fill with general web, which is the one lane with room to spare. A 15T")
    print("  budget therefore forces the web-heavy mixture this session warns against,")
    print("  not by choice but by arithmetic.")
    print("\n  Shrinking the budget raises every scarce lane's reachable share, because")
    print("  the ceiling is fixed in tokens while the denominator falls.")

    print("\n\nRECOMMENDED PLAN — 9T budget\n")
    B = 9.0
    PLAN = {"web": 41, "code": 28, "indic": 13, "stem": 11, "reason": 6, "agentic": 1}
    assert sum(PLAN.values()) == 100
    print(f"{'lane':<9}{'share':>7}{'demand':>10}{'ceiling':>10}{'headroom':>10}{'epochs':>8}  verdict")
    rule()
    for l in LANES:
        d = PLAN[l] / 100 * B * 1000
        c = ceiling[l]
        uniq = nat[l] + extra[l] + syn[l]
        ep = d / uniq
        v = "fits" if d <= c else "OVER"
        print(f"{l:<9}{PLAN[l]:>6}%{d:>9.0f}B{c:>9.0f}B{(c-d)/c:>9.1%}{ep:>8.1f}  {v}")
    rule()
    over = [l for l in LANES if PLAN[l] / 100 * B * 1000 > ceiling[l]]
    print(f"\n  Lanes over their ceiling: {over if over else 'none'}")
    print(f"  Indic reaches {PLAN['indic']}% with no crawling, no OCR and no new generation.")
    print(f"  At 15T the same ceiling would cap it at {ceiling['indic']/1000/15:.1%}.")
    print(f"\n  9T is {9e12/40e9:.0f} tokens per parameter for a 40B model, against Chinchilla's")
    print(f"  compute-optimal 20 and Llama-3-70B's ~214. It stays inside the 10 to 30T")
    print(f"  range Session 3 set for V5 only at its lower edge, and below it by 1T.")


if __name__ == "__main__":
    main()
