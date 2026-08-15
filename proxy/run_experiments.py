#!/usr/bin/env python3
"""Run every proxy experiment in the plan, one after another.

Three experiments, each testing one number in the mixture-and-curriculum plan:

  E1  Indic-share sweep      where does the Indic lane stop paying, and where does
                             it start costing the other lanes? Sets the floor.
  E2  Anneal ordering        same tokens, same budget, same learning-rate schedule:
                             is holding the reserve back for the cooldown better
                             than spending it evenly?
  E3  Selector and the floor an English-only proxy scores candidate batches. Does
                             the Indic lane collapse without an always-on floor,
                             and does the floor recover it?

    python3 run_experiments.py [--only E1] [--tokens 20e6] [--dry-run]
"""
import argparse, itertools, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUNS = os.path.join(HERE, "runs")

# The plan's main pretraining mixture, projected onto the four lanes the proxy
# can actually source. The plan's seven lanes collapse as:
#   web 34 -> web | code 24 -> code | indic 13 -> indic | stem 11 + reasoning 6 -> math
# Long-context (8) and agentic (4) have no proxy lane and are excluded here.
PLAN = {"web": 34, "code": 24, "indic": 13, "math": 17}
PLAN_N = {k: v / sum(PLAN.values()) for k, v in PLAN.items()}
OTHERS = ["web", "code", "math"]
OTHERS_W = {k: PLAN_N[k] / sum(PLAN_N[o] for o in OTHERS) for k in OTHERS}


def mix_with_indic(share):
    """The plan mixture with the Indic lane forced to `share`, others in plan ratio."""
    m = {k: OTHERS_W[k] * (1 - share) for k in OTHERS}
    m["indic"] = share
    return m


def fmt(mix):
    return ",".join(f"{k}={v:.6f}" for k, v in mix.items())


def run(name, args, tokens, extra=None, force=False):
    out = os.path.join(RUNS, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"== {name}: exists, skipping")
        return
    cmd = [PY, os.path.join(HERE, "train.py"), "--name", name, "--tokens", str(tokens)] + args
    if extra:
        cmd += extra
    print("==", name, "\n  ", " ".join(cmd[2:]), flush=True)
    if DRY:
        return
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"run {name} failed")
    print(f"== {name} done in {(time.time()-t)/60:.1f} min\n", flush=True)


def e1(tokens):
    """Indic share sweep at a fixed budget."""
    for share in [0.0, 0.03, 0.08, PLAN_N["indic"], 0.30]:
        tag = f"{share*100:.0f}" if share != PLAN_N["indic"] else "plan"
        run(f"e1_indic{tag}", ["--mix", fmt(mix_with_indic(share))], tokens)


def e2(tokens, seeds=(0, 1)):
    """Anneal ordering: spend the reserve evenly, or hold it for the cooldown.

    Both arms train on the same budget with the same learning-rate schedule and
    give the reserve pool the same number of tokens. Only the timing differs.
    """
    reserve_frac = 0.15           # of the Indic pool
    indic = PLAN_N["indic"]
    anneal = 0.10                 # last tenth of the run is the cooldown
    # tokens the reserve pool receives in either arm
    res_tokens = tokens * indic * reserve_frac
    # arm B: all of that spent inside the anneal window
    res_share_anneal = res_tokens / (tokens * anneal)
    # arm B main phase: the non-reserve Indic share, over the first 90%
    main_indic = (tokens * indic * (1 - reserve_frac)) / (tokens * (1 - anneal))

    for seed in seeds:
        # arm A: reserve is just part of the Indic lane the whole way through
        mix_a = dict(mix_with_indic(indic))
        mix_a["indic"] = indic * (1 - reserve_frac)
        mix_a["indic_reserve"] = indic * reserve_frac
        run(f"e2_spread_s{seed}",
            ["--mix", fmt(mix_a), "--reserve-frac", str(reserve_frac), "--seed", str(seed)],
            tokens)

        # arm B: reserve untouched until the cooldown
        main = dict(mix_with_indic(indic))
        main["indic"] = main_indic
        rest = 1 - main_indic
        for k in OTHERS:
            main[k] = OTHERS_W[k] * rest
        # during the cooldown the Indic budget is the reserve and nothing else, so
        # both arms hand each pool the same number of tokens over the whole run
        ann = {k: OTHERS_W[k] * (1 - res_share_anneal) for k in OTHERS}
        ann["indic_reserve"] = res_share_anneal
        sched = json.dumps([{"until": 1 - anneal, "mix": fmt(main)},
                            {"until": 1.0, "mix": fmt(ann)}])
        run(f"e2_anneal_s{seed}",
            ["--schedule", sched, "--reserve-frac", str(reserve_frac),
             "--blend", "0.02", "--seed", str(seed)],
            tokens)


def e3(tokens):
    """An English-only selector, with and without the always-on floor."""
    ref = os.path.join(RUNS, "e3_ref_en.pt")
    if not os.path.exists(ref) and not DRY:
        run("e3_ref_en", ["--mix", "web=1.0", "--save-ckpt"], tokens * 0.4)
    plan = fmt(mix_with_indic(PLAN_N["indic"]))
    run("e3_sel_nofloor", ["--mix", plan, "--select-ref", ref, "--select-keep", "0.4",
                           "--floor-share", "0.0"], tokens)
    run("e3_sel_floor", ["--mix", plan, "--select-ref", ref, "--select-keep", "0.4",
                         "--floor-share", str(PLAN_N["indic"])], tokens)


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="E1, E2 or E3")
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    DRY = a.dry_run
    os.makedirs(RUNS, exist_ok=True)
    print("plan mixture (4-lane projection):", {k: round(v, 4) for k, v in PLAN_N.items()})
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3)]:
        if a.only and a.only.upper() != name:
            continue
        print(f"\n########## {name} ##########")
        fn(a.tokens)


DRY = False
if __name__ == "__main__":
    main()
