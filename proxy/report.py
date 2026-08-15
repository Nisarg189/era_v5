#!/usr/bin/env python3
"""Turn runs/*.json into the result tables quoted in the README.

    python3 report.py            # prints all three experiments
"""
import json, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
LANES = ["web", "code", "indic", "math"]


def load(name):
    p = os.path.join(RUNS, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def bpb(r, lane):
    return r["final_eval"][lane]["bpb"]


def rule(w=72):
    print("-" * w)


def e1():
    arms = [("e1_indic0", "0%"), ("e1_indic3", "3%"), ("e1_indic8", "8%"),
            ("e1_indicplan", "13% (plan)"), ("e1_indic30", "30%")]
    rs = [(lbl, load(n)) for n, lbl in arms]
    if not all(r for _, r in rs):
        print("E1: incomplete\n")
        return
    print("E1  INDIC SHARE SWEEP — held-out bits per byte, 30M tokens, identical everything else\n")
    print(f"{'indic share':<14}" + "".join(f"{l:>10}" for l in LANES) + f"{'non-indic avg':>15}")
    rule(74)
    base = None
    for lbl, r in rs:
        non = sum(bpb(r, l) for l in LANES if l != "indic") / 3
        if base is None:
            base = (bpb(r, "indic"), non)
        print(f"{lbl:<14}" + "".join(f"{bpb(r,l):>10.4f}" for l in LANES) + f"{non:>15.4f}")
    rule(74)
    print(f"\n{'indic share':<14}{'indic bpb':>12}{'gain vs 0%':>13}{'cost to others':>16}")
    for lbl, r in rs:
        non = sum(bpb(r, l) for l in LANES if l != "indic") / 3
        print(f"{lbl:<14}{bpb(r,'indic'):>12.4f}{(base[0]-bpb(r,'indic'))/base[0]:>12.1%}"
              f"{(non-base[1])/base[1]:>15.2%}")
    print()


def e2():
    print("E2  ANNEAL ORDERING — identical tokens per pool, identical LR schedule, "
          "only the timing differs\n")
    seeds = [0, 1]
    rows = []
    for s in seeds:
        a, b = load(f"e2_spread_s{s}"), load(f"e2_anneal_s{s}")
        if not (a and b):
            print("E2: incomplete\n")
            return
        rows.append((s, a, b))
    print(f"{'seed':<6}{'arm':<26}" + "".join(f"{l:>10}" for l in LANES))
    rule(76)
    for s, a, b in rows:
        print(f"{s:<6}{'reserve spread evenly':<26}" + "".join(f"{bpb(a,l):>10.4f}" for l in LANES))
        print(f"{'':<6}{'reserve held for anneal':<26}" + "".join(f"{bpb(b,l):>10.4f}" for l in LANES))
    rule(76)
    print(f"\n{'seed':<6}{'indic bpb spread':>18}{'held':>10}{'improvement':>14}")
    imps = []
    for s, a, b in rows:
        imp = (bpb(a, "indic") - bpb(b, "indic")) / bpb(a, "indic")
        imps.append(imp)
        print(f"{s:<6}{bpb(a,'indic'):>18.4f}{bpb(b,'indic'):>10.4f}{imp:>13.2%}")
    print(f"{'mean':<6}{'':>18}{'':>10}{sum(imps)/len(imps):>13.2%}")
    # verify the arms really were token-matched
    a = rows[0][1]
    b = rows[0][2]
    print("\ntokens per pool (seed 0), confirming the arms are matched:")
    for k in sorted(set(a["tokens_seen_per_pool"]) | set(b["tokens_seen_per_pool"])):
        va, vb = a["tokens_seen_per_pool"].get(k, 0), b["tokens_seen_per_pool"].get(k, 0)
        print(f"  {k:<15}{va:>12,}{vb:>12,}   delta {(vb-va)/max(va,1):>+7.2%}")
    print()


def e3():
    nf, fl = load("e3_sel_nofloor"), load("e3_sel_floor")
    if not (nf and fl):
        print("E3: incomplete\n")
        return
    print("E3  SELECTOR AND THE PROTECTED FLOOR — English-only proxy, keeps the top 40%\n")
    print(f"{'pool':<8}{'offered':>12}{'kept (no floor)':>18}{'rate':>8}"
          f"{'kept (floor)':>15}{'rate':>8}")
    rule(72)
    for k in LANES:
        o1, k1 = nf["tokens_offered_per_pool"].get(k, 0), nf["tokens_seen_per_pool"].get(k, 0)
        o2, k2 = fl["tokens_offered_per_pool"].get(k, 0), fl["tokens_seen_per_pool"].get(k, 0)
        print(f"{k:<8}{o1:>12,}{k1:>18,}{k1/max(o1,1):>8.1%}{k2:>15,}{k2/max(o2,1):>8.1%}")
    rule(72)
    print(f"\n{'lane':<8}{'bpb no floor':>15}{'bpb with floor':>17}{'change':>12}")
    for k in LANES:
        d = (bpb(fl, k) - bpb(nf, k)) / bpb(nf, k)
        print(f"{k:<8}{bpb(nf,k):>15.4f}{bpb(fl,k):>17.4f}{d:>11.2%}")
    ref = load("e3_ref_en")
    if ref:
        print(f"\nproxy scorer: web-only model, {ref['tokens_trained']/1e6:.0f}M tokens, "
              f"indic bpb {bpb(ref,'indic'):.4f} vs web bpb {bpb(ref,'web'):.4f}")
    print()


def main():
    done = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(RUNS, "*.json")))
    print(f"runs present: {len(done)}\n{done}\n")
    e1()
    e2()
    e3()


if __name__ == "__main__":
    main()
