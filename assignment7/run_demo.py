"""
One command that regenerates every number in the README.

    python3 run_demo.py

It runs the gradient checks, the invariant tests, then the three experiments:
  A  codec invertibility and the per-script collision audit   (artifacts/invertibility.json)
  B  head-free vs dense-head language models, trained          (artifacts/heads.json)
  C+D reversal, decoding without a head, and the 1M-vocab claim (artifacts/reversal.json)

Everything runs on CPU with numpy only. Expect roughly 15 to 20 minutes end to end;
the language-model training dominates. Nothing here reaches the network.
"""
import subprocess, sys, os, json, time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED: {label}")
        sys.exit(r.returncode)
    print(f"[{label} finished in {time.time()-t0:.0f}s]")


def main():
    run(["-m", "tests.test_autograd"], "gradient checks on the autograd engine")
    run(["-m", "tests.test_reversible"], "reversible-Kronecker invariant tests")
    run(["experiments/exp_invertibility.py"], "A: codec invertibility + collision audit")
    run(["experiments/exp_heads.py"], "B: head-free vs dense-head training")
    run(["experiments/exp_reversal.py"], "C+D: reversal, decoding, 1M-vocab")

    print(f"\n{'#'*70}\nSUMMARY\n{'#'*70}")
    art = os.path.join(ROOT, "artifacts")
    h = json.load(open(os.path.join(art, "heads.json")))
    for key, r in h["runs"].items():
        f = r["final"]
        print(f"  {r['label']:42s} val_ppl {f['ppl']:6.2f}  acc {f['acc']:.3f}  "
              f"V-dep head params {r['head_params_Vdep']:,}")
    inv = json.load(open(os.path.join(art, "invertibility.json")))
    w32 = inv["windows"]["32"]
    print(f"  codec collisions on {inv['n_tokens']} real tokens: "
          f"{w32['total_collisions']} at pos_dim=32, "
          f"{inv['windows']['48']['total_collisions']} at 48")
    rev = json.load(open(os.path.join(art, "reversal.json")))
    print(f"  head == codec adjoint, max abs diff: {rev['adjoint_identity_max_abs_diff']:.1e}")
    sc = rev["candidate_scaling"]
    print(f"  val acc at {sc[0]['candidates']} candidates: {sc[0]['val_top1_acc']:.3f}  ->  "
          f"at {sc[-1]['candidates']}: {sc[-1]['val_top1_acc']:.3f}  "
          f"(head params added: 0)")
    print("\nAll artifacts written to artifacts/. See README.md for the tables.")


if __name__ == "__main__":
    main()
