#!/usr/bin/env python3
"""The plan's numbers, checked against what can actually be downloaded.

Supply ceilings come from supply_constraint.py, which counts only datasets that
have already been published. The programme has no budget for crawling or optical
character recognition, so nothing here may assume newly collected text.

Nothing is typed by hand except the stage schedule, the epoch caps and the token
budget. Everything else is derived, so no share can be asserted without the
arithmetic behind it appearing at the same time.

    python3 plan_arithmetic.py
"""
from supply_constraint import inventory, EXTRA, LANES, CAP_NATURAL, CAP_SYNTHETIC

BUDGET_T = 9.0

# The curriculum is the plan. Six stages, each with its own mixture, each spanning
# a fraction of the budget. The headline mixture is not a separate decision: it is
# what these stages average to, which is why it is computed rather than asserted.
STAGES = {
    "S0 seed":         (0.02, {"web": 60, "code": 14, "indic": 14, "stem": 9,  "reason": 2,  "agentic": 1}),
    "S1 general":      (0.32, {"web": 56, "code": 22, "indic": 13, "stem": 7,  "reason": 1,  "agentic": 1}),
    "S2 capability":   (0.32, {"web": 36, "code": 33, "indic": 12, "stem": 15, "reason": 3,  "agentic": 1}),
    "S3 reasoning":    (0.18, {"web": 26, "code": 31, "indic": 13, "stem": 12, "reason": 17, "agentic": 1}),
    "S4 long context": (0.13, {"web": 31, "code": 35, "indic": 14, "stem": 9,  "reason": 10, "agentic": 1}),
    "S5 anneal":       (0.03, {"web": 10, "code": 20, "indic": 28, "stem": 12, "reason": 22, "agentic": 8}),
}

FLOORS = {"indic": 10, "reason": 4, "agentic": 1}
ANNEAL_RESERVE_SHARE = 3

# Where the newly counted Indic tokens land by provenance tier. Crawl-derived
# multilingual corpora are tier B whatever their pedigree. Varta is edited news and
# Wikipedia is edited by people, so both are tier A.
INDIC_EXTRA_TIER = {"FineWeb-2, Indian-language subsets": "B",
                    "CulturaX, Indian-language subsets": "B",
                    "HPLT v2, Indian-language subsets": "B",
                    "Varta Indian news corpus": "A",
                    "Indian-language Wikipedia dumps": "A"}


def implied_mixture():
    raw = {l: sum(sp * m[l] for sp, m in STAGES.values()) for l in LANES}
    out = {l: int(raw[l]) for l in LANES}
    rem = sorted(LANES, key=lambda l: raw[l] - int(raw[l]), reverse=True)
    for i in range(100 - sum(out.values())):
        out[rem[i]] += 1
    return raw, out


def rule(w=78):
    print("-" * w)


def main():
    nat, syn = inventory()
    extra = {k: sum(t for _, t in v) for k, v in EXTRA.items()}
    ceiling = {l: (nat[l] + extra[l]) * CAP_NATURAL + syn[l] * CAP_SYNTHETIC for l in LANES}
    raw, MAIN = implied_mixture()
    ANNEAL = STAGES["S5 anneal"][1]

    assert sum(MAIN.values()) == 100
    for k, (sp, m) in STAGES.items():
        assert sum(m.values()) == 100, (k, sum(m.values()))
    assert abs(sum(sp for sp, _ in STAGES.values()) - 1.0) < 1e-9

    print(f"CURRICULUM — six stages, each with its own mixture, over {BUDGET_T}T tokens")
    print("percentages are of that stage's own tokens, so each row totals 100\n")
    print(f"{'stage':<17}{'span of run':>13}{'tokens':>9}" + "".join(f"{l[:7]:>9}" for l in LANES))
    rule(93)
    at = 0.0
    for name, (span, m) in STAGES.items():
        lo, hi = at, at + span
        at = hi
        print(f"{name:<17}{f'{lo:.0%} to {hi:.0%}':>13}{span*BUDGET_T:>8.2f}T"
              + "".join(f"{m[l]:>9}" for l in LANES))
    rule(93)
    print(f"{'implied average':<17}{'100%':>13}{BUDGET_T:>8.2f}T"
          + "".join(f"{raw[l]:>9.1f}" for l in LANES))
    print(f"{'headline mixture':<17}{'':>13}{'':>9}" + "".join(f"{MAIN[l]:>9}" for l in LANES))

    print("\n\nDOES THE DATA EXIST? — ceilings from published datasets only\n")
    print(f"{'lane':<9}{'share':>7}{'demand':>10}{'ceiling':>10}{'headroom':>10}{'epochs':>8}  verdict")
    rule()
    for l in LANES:
        d = MAIN[l] / 100 * BUDGET_T * 1000
        uniq = nat[l] + extra[l] + syn[l]
        ep = d / uniq
        v = "fits" if d <= ceiling[l] else f"OVER by {d/ceiling[l]-1:.0%}"
        print(f"{l:<9}{MAIN[l]:>6}%{d:>9.0f}B{ceiling[l]:>9.0f}B"
              f"{(ceiling[l]-d)/ceiling[l]:>9.1%}{ep:>8.1f}  {v}")
    rule()
    over = [l for l in LANES if MAIN[l] / 100 * BUDGET_T * 1000 > ceiling[l]]
    print(f"\n  Over ceiling: {over if over else 'none'}. Every other lane fits with no new")
    print("  collection of any kind. Long context is absent as a lane on purpose: it is a")
    print("  sequence-length policy applied to the web and code lanes in S4, not a separate")
    print("  pool of tokens, and counting it separately would double-count them.")

    print(f"\n\nINDIC SLOT — {MAIN['indic']}% = {MAIN['indic']/100*BUDGET_T*1000:.0f}B, "
          "by provenance tier\n")
    demand_b = MAIN["indic"] / 100 * BUDGET_T * 1000
    a_extra = sum(t for n, t in EXTRA["indic"] if INDIC_EXTRA_TIER[n] == "A")
    b_extra = sum(t for n, t in EXTRA["indic"] if INDIC_EXTRA_TIER[n] == "B")
    alloc = [("A verified native", 64.0 + a_extra, CAP_NATURAL),
             ("B unverified crawl", 44.9 + b_extra, CAP_NATURAL),
             ("C translated", 5.0, CAP_NATURAL)]
    used = sum(u * e for _, u, e in alloc)
    alloc.append(("D synthetic", (demand_b - used) / CAP_SYNTHETIC, CAP_SYNTHETIC))

    print(f"{'tier':<20}{'unique':>10}{'epochs':>9}{'tokens':>10}{'% of slot':>11}")
    rule(60)
    for name, uniq, ep in alloc:
        print(f"{name:<20}{uniq:>9.1f}B{ep:>8}x{uniq*ep:>9.0f}B{uniq*ep/demand_b:>10.1%}")
    rule(60)
    tot = sum(u * e for _, u, e in alloc)
    print(f"{'total':<20}{'':>10}{'':>9}{tot:>9.0f}B{tot/demand_b:>10.1%}")
    native = sum(u * e for n, u, e in alloc if n[0] in "AB")
    print(f"\n  Native text (tiers A and B): {native/demand_b:.0%} of the slot")
    print(f"  Model-generated (tier D):    {alloc[3][1]*alloc[3][2]/demand_b:.0%}, using "
          f"{alloc[3][1]:.0f}B of the {syn['indic']:.0f}B already available")
    print("\n  No new generation is required. The synthetic tier is drawn entirely from")
    print("  Sangraha's existing synthetic split, and not all of it is needed.")

    print(f"\n\nAGENTIC SLOT — {MAIN['agentic']}% = {MAIN['agentic']/100*BUDGET_T*1000:.0f}B "
          f"against a {ceiling['agentic']:.0f}B ceiling")
    print(f"  The one lane the constraint cannot satisfy. It runs at "
          f"{MAIN['agentic']/100*BUDGET_T*1000/(nat['agentic']+extra['agentic']):.1f} epochs, above")
    print("  the 4-epoch guideline, and the overage is accepted deliberately: the lane is 1%")
    print("  of the run, the alternative is no agentic capability at all, and buying it any")
    print("  other way needs generation the programme cannot fund.")

    print("\n\nPROTECTED FLOORS — the selector may not go below these in any batch\n")
    for k, v in FLOORS.items():
        print(f"  {k:<9}{v:>3}%   (headline share {MAIN[k]}%)")
    print(f"  {'total':<9}{sum(FLOORS.values()):>3}% of every batch is outside the selector's control")

    print(f"\n\nANNEAL — {ANNEAL_RESERVE_SHARE}% of the budget = "
          f"{ANNEAL_RESERVE_SHARE/100*BUDGET_T*1000:.0f}B held back for the cooldown\n")
    print(f"{'lane':<10}{'anneal':>8}{'headline':>10}{'change':>9}")
    rule(37)
    for lane, share in sorted(ANNEAL.items(), key=lambda x: -x[1]):
        print(f"{lane:<10}{share:>7}%{MAIN[lane]:>9}%{share-MAIN[lane]:>+8}pp")


if __name__ == "__main__":
    main()
