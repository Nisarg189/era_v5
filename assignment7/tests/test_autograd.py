"""Numerical gradient checks for the autograd engine.

Each op's analytic gradient is compared against a finite-difference estimate on a
scalar loss. Run: python3 -m tests.test_autograd
"""
import numpy as np
from rkron.autograd import Tensor, cross_entropy

rng = np.random.default_rng(0)


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; a = f()
        x[i] = old - eps; b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def check(name, params, forward):
    # analytic
    for p in params:
        p.grad = None
    loss = forward()
    loss.backward()
    ok = True
    for k, p in enumerate(params):
        ana = p.grad.copy()

        def f(pp=p):
            return forward().data

        num = numgrad(f, p.data)
        err = np.abs(ana - num).max() / (np.abs(num).max() + 1e-9)
        status = "ok" if err < 1e-4 else "FAIL"
        if err >= 1e-4:
            ok = False
        print(f"  {name} param[{k}] rel-err {err:.2e} {status}")
    return ok


def main():
    allok = True

    A = Tensor(rng.standard_normal((4, 5)), requires_grad=True)
    B = Tensor(rng.standard_normal((5, 3)), requires_grad=True)
    allok &= check("matmul+sum", [A, B], lambda: (A @ B).sum())

    x = Tensor(rng.standard_normal((3, 6)), requires_grad=True)
    allok &= check("relu", [x], lambda: x.relu().sum())

    a = Tensor(rng.standard_normal((3, 4)), requires_grad=True)
    b = Tensor(rng.standard_normal((4,)), requires_grad=True)
    allok &= check("add-broadcast+mul", [a, b], lambda: (a + b).__mul__(a).sum())

    s = Tensor(rng.standard_normal((3, 5)), requires_grad=True)
    csoft = Tensor(rng.standard_normal((3, 5)))  # fixed coefficients
    allok &= check("softmax", [s], lambda: (s.softmax_lastdim() * csoft).sum())

    h = Tensor(rng.standard_normal((3, 7)), requires_grad=True)
    g = Tensor(rng.standard_normal((7,)), requires_grad=True)
    be = Tensor(rng.standard_normal((7,)), requires_grad=True)
    cln = Tensor(rng.standard_normal((3, 7)))  # fixed coefficients
    allok &= check("layernorm", [h, g, be], lambda: (h.layernorm(g, be) * cln).sum())

    logits = Tensor(rng.standard_normal((4, 6)), requires_grad=True)
    tgt = rng.integers(0, 6, size=4)
    allok &= check("cross_entropy", [logits], lambda: cross_entropy(logits, tgt))

    print("ALL AUTOGRAD CHECKS", "PASS" if allok else "FAIL")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
