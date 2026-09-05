"""
Experiment B: does removing the output head cost accuracy?

Three transformers are trained from scratch on the same V5-mixture corpus, with the
same initialisation seed, the same batches in the same order, and the same optimizer.
They differ only in how tokens enter and leave the model:

  1. Kronecker input  + reversible (tied-Kronecker) head   <- the proposal
  2. Kronecker input  + dense V x D output head
  3. dense input table + dense V x D output head            <- the classic baseline

The reversible head introduces zero parameters that scale with the vocabulary. The
question is whether it matches the dense head's language-modelling quality. A second
table extrapolates the head's parameter and memory cost to the V5 width and to a
1,000,000-token vocabulary, which is the concrete "1M vocab for free" claim.
"""

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rkron.data import make_streams
from rkron.model import TinyLM
from rkron.train import train_model

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
os.makedirs(OUT, exist_ok=True)

CFG = dict(d_model=96, block_size=48, n_layer=2, steps=600, lr=1e-3, batch=24, seed=0)


def head_cost_table(code_dim=8192):
    rows = []
    for D in [CFG["d_model"], 8096]:
        for V in [1200, 32768, 131072, 262144, 1_000_000]:
            dense = V * D
            tied = 0  # reuses the shared projection code_dim x D
            shared_proj = code_dim * D
            rows.append({
                "width_D": D, "vocab_V": V,
                "dense_head_params": dense,
                "tied_head_params": tied,
                "shared_projection_params": shared_proj,
                "dense_train_GB": round(dense * 16 / 1e9, 3),
                "dense_infer_GB": round(dense * 2 / 1e9, 3),
                "tied_train_GB": 0.0,
            })
    return rows


def main():
    t0 = time.time()
    log = open(os.path.join(OUT, "heads.log"), "w")

    def say(*a):
        m = " ".join(str(x) for x in a)
        print(m); log.write(m + "\n"); log.flush()

    say("loading corpus...")
    data = make_streams(target_vocab=1200, max_chars_per_lane=300000, seed=0)
    say(f"vocab {len(data['vocab'])}  train {len(data['train'])}  val {len(data['val'])}")

    configs = [
        ("kron", "tied_kron", "Kronecker input + reversible head"),
        ("kron", "dense", "Kronecker input + dense head"),
        ("dense", "dense", "dense input + dense head (classic)"),
    ]
    results = {"config": CFG, "vocab_size": len(data["vocab"]),
               "lane_stats": data["lane_stats"], "runs": {}}

    for input_path, head, label in configs:
        say(f"\n=== {label} ===")
        m = TinyLM(data["vocab"], d_model=CFG["d_model"], block_size=CFG["block_size"],
                   n_layer=CFG["n_layer"], input_path=input_path, head=head, seed=CFG["seed"])
        say(f"    total params {m.total_param_count():,}  "
            f"V-dependent head params {m.head_param_count():,}")
        r = train_model(m, data, steps=CFG["steps"], lr=CFG["lr"], batch=CFG["batch"],
                        block=CFG["block_size"], eval_every=100, seed=0, log=say)
        r["total_params"] = m.total_param_count()
        r["head_params_Vdep"] = m.head_param_count()
        r["label"] = label
        results["runs"][f"{input_path}+{head}"] = r
        say(f"    FINAL val loss {r['final']['loss']:.4f}  ppl {r['final']['ppl']:.2f}  "
            f"acc {r['final']['acc']:.4f}")
        say(f"    per-lane val loss: " +
            "  ".join(f"{ln} {v['loss']:.3f}" for ln, v in r["per_lane"].items()))

    results["head_cost_extrapolation"] = head_cost_table()
    results["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "heads.json"), "w") as f:
        json.dump(results, f, indent=2)
    say(f"\nDONE in {results['wall_seconds']}s -> artifacts/heads.json")


if __name__ == "__main__":
    main()
