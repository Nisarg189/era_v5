#!/usr/bin/env python3
"""A small decoder-only LM trained on a configurable data mixture.

This is the measuring instrument for every mixture hypothesis in the plan. One
run = one mixture (or one curriculum of mixtures) trained from scratch on a fixed
token budget, scored by held-out bits per byte on each lane separately.

Bits per byte, not loss, because the lanes use different scripts. Devanagari
costs about three UTF-8 bytes per character, so per-token loss is not comparable
across lanes while bits per byte is.

Everything that is not the mixture is held fixed across runs: architecture, token
budget, learning-rate schedule, batch size, seed and tokenizer. The mixture is
the only independent variable.

    python3 train.py --name my_run --mix web=0.6,code=0.2,indic=0.15,math=0.05
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(HERE, "runs")
LANES = ["web", "code", "indic", "math"]


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=True)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=256, n_layer=6, n_head=4, seq=512, drop=0.0):
        super().__init__()
        self.seq = seq
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, n_head, drop) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight            # tied
        self.apply(self._init)
        mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok.weight.numel() + self.pos.weight.numel()
        return n


# ----------------------------------------------------------------- data
class Pools:
    """Named token pools. 'indic' is split so a reserve can be held back."""

    def __init__(self, reserve_frac=0.0, seed=0):
        self.train, self.val, self.meta = {}, {}, json.load(open(os.path.join(DATA, "token_meta.json")))
        for lane in LANES:
            self.train[lane] = np.load(os.path.join(DATA, f"{lane}.train.npy"), mmap_mode="r")
            self.val[lane] = np.load(os.path.join(DATA, f"{lane}.val.npy"))
        if reserve_frac > 0:
            a = self.train["indic"]
            cut = int(len(a) * (1 - reserve_frac))
            self.train["indic"], self.train["indic_reserve"] = a[:cut], a[cut:]
        self.rng = np.random.default_rng(seed)

    def batch(self, mix, B, T, device):
        """One batch, lanes drawn according to `mix` (a dict of pool -> weight)."""
        names = list(mix)
        p = np.array([mix[n] for n in names], dtype=np.float64)
        p = p / p.sum()
        picks = self.rng.choice(len(names), size=B, p=p)
        xs = np.empty((B, T + 1), dtype=np.int64)
        for i, pi in enumerate(picks):
            a = self.train[names[pi]]
            j = self.rng.integers(0, len(a) - T - 1)
            xs[i] = a[j:j + T + 1]
        t = torch.from_numpy(xs).to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:], picks, names


# ----------------------------------------------------------------- selection
@torch.no_grad()
def per_seq_loss(model, x, y):
    """Mean cross-entropy per sequence, without reducing across the batch."""
    logits, _ = model(x)
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return l.view(y.shape).mean(dim=1)


def select_batch(model, ref, x, y, picks, names, keep, floor_share, protected, rng):
    """Keep the `keep` highest-scoring sequences of a candidate batch.

    Score is excess loss, current model minus reference model. The reference here
    is deliberately English-only, which is the failure mode Session 5 describes:
    a proxy that does not speak Hindi scores every Hindi sequence as worthless.

    If floor_share > 0, that fraction of the kept batch is reserved for sequences
    from the protected lanes regardless of score.
    """
    cur = per_seq_loss(model, x, y)
    r = per_seq_loss(ref, x, y)
    score = (cur - r).cpu().numpy()
    order = list(np.argsort(-score))

    chosen = []
    if floor_share > 0:
        n_floor = int(round(keep * floor_share))
        prot_idx = [i for i in order if names[picks[i]] in protected]
        chosen = prot_idx[:n_floor]
    rest = [i for i in order if i not in set(chosen)]
    chosen = chosen + rest[: keep - len(chosen)]
    idx = torch.tensor(sorted(chosen), device=x.device)
    return x[idx], y[idx], [picks[i] for i in sorted(chosen)], score


# ----------------------------------------------------------------- schedule
def parse_mix(s):
    d = {}
    for part in s.split(","):
        k, v = part.split("=")
        d[k.strip()] = float(v)
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}


def mix_at(schedule, frac, blend):
    """Mixture at progress `frac`, linearly blended across phase boundaries."""
    phases = schedule
    for i, ph in enumerate(phases):
        if frac <= ph["until"] or i == len(phases) - 1:
            cur = ph["mix"]
            if i == 0 or blend <= 0:
                return cur
            start = phases[i - 1]["until"]
            if frac >= start + blend:
                return cur
            prev, w = phases[i - 1]["mix"], (frac - start) / blend
            keys = set(prev) | set(cur)
            return {k: (1 - w) * prev.get(k, 0.0) + w * cur.get(k, 0.0) for k in keys}
    return phases[-1]["mix"]


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, pools, T, device, batches=24, B=16):
    model.eval()
    out = {}
    for lane in LANES:
        a = pools.val[lane]
        n = min(batches, max(1, (len(a) - 1) // (B * T)))
        tot_loss, tot_tok = 0.0, 0
        for k in range(n):
            seqs = []
            for b in range(B):
                j = (k * B + b) * T
                if j + T + 1 > len(a):
                    break
                seqs.append(a[j:j + T + 1])
            if not seqs:
                break
            t = torch.from_numpy(np.array(seqs, dtype=np.int64)).to(device)
            _, loss = model(t[:, :-1], t[:, 1:])
            ntok = t.shape[0] * T
            tot_loss += loss.item() * ntok
            tot_tok += ntok
        nats = tot_loss / max(1, tot_tok)
        bpt = pools.meta["lanes"][lane]["bytes_per_val_token"]
        out[lane] = {"loss_nats": round(nats, 5),
                     "ppl": round(math.exp(min(20, nats)), 3),
                     "bpb": round(nats / math.log(2) / bpt, 5)}
    model.train()
    return out


# ----------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mix", default="web=0.34,code=0.24,indic=0.16,math=0.26")
    ap.add_argument("--schedule", default="", help="JSON list of {until, mix} phases")
    ap.add_argument("--blend", type=float, default=0.0, help="phase blend width, fraction of run")
    ap.add_argument("--reserve-frac", type=float, default=0.0)
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--select-ref", default="", help="checkpoint of the English-only proxy model")
    ap.add_argument("--select-keep", type=float, default=0.4, help="fraction of candidates kept")
    ap.add_argument("--floor-share", type=float, default=0.0, help="always-on share of each kept batch")
    ap.add_argument("--protected", default="indic", help="comma-separated protected lanes")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(RUNS, exist_ok=True)
    device = torch.device(a.device)

    schedule = ([{"until": p["until"], "mix": parse_mix(p["mix"]) if isinstance(p["mix"], str) else p["mix"]}
                 for p in json.loads(a.schedule)] if a.schedule
                else [{"until": 1.0, "mix": parse_mix(a.mix)}])

    pools = Pools(reserve_frac=a.reserve_frac, seed=a.seed)
    model = GPT(pools.meta["vocab_size"], a.d_model, a.n_layer, a.n_head, a.seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    tok_per_step = a.batch * a.seq
    steps = int(a.tokens // tok_per_step)
    warm = max(1, int(steps * a.warmup))

    def lr_at(s):
        if s < warm:
            return a.lr * s / warm
        p = (s - warm) / max(1, steps - warm)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    print(f"run={a.name} device={a.device} steps={steps} tokens={steps*tok_per_step/1e6:.1f}M "
          f"params_total={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params_non_embed={model.n_params()/1e6:.2f}M", flush=True)

    ref = None
    if a.select_ref:
        ref = GPT(pools.meta["vocab_size"], a.d_model, a.n_layer, a.n_head, a.seq).to(device)
        ref.load_state_dict(torch.load(a.select_ref, map_location=device))
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        cand_B = int(round(a.batch / a.select_keep))
        print(f"  selection on: {cand_B} candidates -> keep {a.batch}, "
              f"floor_share={a.floor_share}, protected={a.protected}", flush=True)
    protected = set(a.protected.split(","))

    hist, seen, offered = [], {}, {}
    t0 = time.time()
    for s in range(steps):
        frac = s / steps
        mix = mix_at(schedule, frac, a.blend)
        if ref is None:
            x, y, picks, names = pools.batch(mix, a.batch, a.seq, device)
        else:
            cx, cy, cpicks, names = pools.batch(mix, cand_B, a.seq, device)
            for pi in cpicks:
                offered[names[pi]] = offered.get(names[pi], 0) + a.seq
            x, y, picks, _score = select_batch(model, ref, cx, cy, cpicks, names,
                                               a.batch, a.floor_share, protected, pools.rng)
        for pi in picks:
            seen[names[pi]] = seen.get(names[pi], 0) + a.seq

        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()

        if s % 50 == 0 or s == steps - 1:
            hist.append({"step": s, "frac": round(frac, 4), "loss": round(loss.item(), 4),
                         "grad_norm": round(gnorm, 4), "lr": round(lr_at(s), 7),
                         "mix": {k: round(v, 4) for k, v in mix.items()}})
        if s % 500 == 0:
            el = time.time() - t0
            print(f"  step {s}/{steps} loss {loss.item():.3f} gnorm {gnorm:.2f} "
                  f"{(s+1)*tok_per_step/max(el,1e-9)/1e3:.1f}k tok/s eta {el/(s+1)*(steps-s-1)/60:.0f}m",
                  flush=True)

    final = evaluate(model, pools, a.seq, device)
    result = {
        "name": a.name, "args": vars(a), "schedule": schedule,
        "steps": steps, "tokens_trained": steps * tok_per_step,
        "params_non_embed": model.n_params(), "params_total": sum(p.numel() for p in model.parameters()),
        "tokens_seen_per_pool": seen, "tokens_offered_per_pool": offered,
        "wall_seconds": round(time.time() - t0, 1),
        "final_eval": final, "history": hist,
    }
    with open(os.path.join(RUNS, f"{a.name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if a.save_ckpt:
        torch.save(model.state_dict(), os.path.join(RUNS, f"{a.name}.pt"))
    print(json.dumps(final, indent=1))
    print(f"wrote runs/{a.name}.json in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
