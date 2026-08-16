#!/usr/bin/env python3
"""What the mixture can be when no new data may be collected.

The programme has no budget for crawling or for optical character recognition, so
every token must come from a dataset that has already been published and can simply
be downloaded. This file works out what that constraint costs.

It computes, for each lane, the ceiling on tokens available at the repetition caps.
plan_arithmetic.py then checks the plan's shares against those ceilings.

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

    print("\n\nWHAT THAT SUPPLY SUPPORTS AT A 9T BUDGET\n")
    B = 9.0
    print(f"{'lane':<9}{'ceiling':>10}{'as share of 9T':>17}   comment")
    rule(66)
    for l in LANES:
        share = ceiling[l] / 1000 / B
        if share >= 1:
            note = f"{share:.1f}x the whole run"
            shown = "unconstrained"
        else:
            note = "this is the hard limit on the lane"
            shown = f"{share:.1%}"
        print(f"{l:<9}{ceiling[l]:>9.0f}B{shown:>17}   {note}")
    rule(66)
    print("\n  Web and code can supply more than an entire 9T run on their own, so they")
    print("  never constrain the mixture. The four scarce lanes each have a real ceiling,")
    print("  and those ceilings are what the mixture has to be built around.")
    print("\n  The ceilings are fixed in tokens, so a larger budget would not raise them.")
    print("  It would only lower each scarce lane's reachable share and push the freed")
    print("  space into general web, which is the one lane with room to spare.")

    print("\n  The plan checked against these ceilings is in plan_arithmetic.py.")


if __name__ == "__main__":
    main()
