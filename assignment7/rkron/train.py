"""Training and evaluation helpers shared by the experiments."""

import numpy as np
from .data import get_batch
from .model import Adam


def _ce_and_acc(logits, y):
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    N = y.size
    flatp = p.reshape(N, -1)
    ce = -np.log(flatp[np.arange(N), y.reshape(-1)] + 1e-12).mean()
    acc = (flatp.argmax(-1) == y.reshape(-1)).mean()
    return float(ce), float(acc)


def evaluate(model, ids, block, n_batches=20, batch=24, seed=1234):
    if len(ids) < block + 2:
        return {"loss": float("nan"), "ppl": float("nan"), "acc": float("nan")}
    rng = np.random.default_rng(seed)
    ces, accs = [], []
    for _ in range(n_batches):
        x, y = get_batch(ids, batch, block, rng)
        logits, _ = model.forward(x)
        ce, acc = _ce_and_acc(logits.data, y)
        ces.append(ce); accs.append(acc)
    loss = float(np.mean(ces))
    return {"loss": loss, "ppl": float(np.exp(loss)), "acc": float(np.mean(accs))}


def train_model(model, data, steps=600, lr=1e-3, clip=1.0, batch=24, block=48,
                eval_every=100, seed=0, log=print):
    opt = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed)          # identical batch order across configs
    history = []
    for step in range(1, steps + 1):
        x, y = get_batch(data["train"], batch, block, rng)
        loss = model.loss(x, y)
        opt.zero_grad()
        loss.backward()
        gnorm = opt.clip_global_norm(clip)
        opt.step()
        if step % eval_every == 0 or step == 1:
            ev = evaluate(model, data["val"], block, batch=batch)
            history.append({"step": step, "train_loss": float(loss.data),
                            "val_loss": ev["loss"], "val_acc": ev["acc"],
                            "grad_norm": float(gnorm)})
            if log:
                log(f"    step {step:4d}  train {float(loss.data):.3f}  "
                    f"val {ev['loss']:.3f}  val_acc {ev['acc']:.3f}")
    final = evaluate(model, data["val"], block, n_batches=40, batch=batch)
    per_lane = {ln: evaluate(model, ids, block, n_batches=10, batch=batch)
                for ln, ids in data.get("val_by_lane", {}).items()
                if len(ids) > block + 2}
    return {"history": history, "final": final, "per_lane": per_lane}
