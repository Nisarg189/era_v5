"""
A small reverse-mode autograd engine over numpy arrays.

Only the operations the proxy transformer needs are implemented, each with an
explicit backward. Nothing here is a wrapper around a deep-learning framework:
every gradient is written out and gradient-checked in tests/test_autograd.py, so
every number the experiments report comes from arithmetic in this file.
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---- helpers ---------------------------------------------------------

    def _accum(self, g):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    @staticmethod
    def _unbroadcast(g, shape):
        """Sum a gradient back to `shape` after numpy broadcasting."""
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    # ---- ops -------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._accum(self._unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(self._unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._accum(self._unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accum(self._unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def matmul(self, other):
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self._accum(self._unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2),
                                              self.data.shape))
            if other.requires_grad:
                other._accum(self._unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad,
                                               other.data.shape))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                     (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._accum(np.broadcast_to(g, self.data.shape).copy())
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad.reshape(self.data.shape))
        out._backward = _backward
        return out

    def transpose_last(self):
        out = Tensor(np.swapaxes(self.data, -1, -2), self.requires_grad, (self,), "T")

        def _backward():
            if self.requires_grad:
                self._accum(np.swapaxes(out.grad, -1, -2))
        out._backward = _backward
        return out

    def softmax_lastdim(self):
        x = self.data
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                # Jacobian-vector product for softmax along the last axis
                dot = (out.grad * s).sum(axis=-1, keepdims=True)
                self._accum(s * (out.grad - dot))
        out._backward = _backward
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mu) * inv
        out_data = xhat * gamma.data + beta.data
        out = Tensor(out_data, True, (self, gamma, beta), "layernorm")
        D = x.shape[-1]

        def _backward():
            g = out.grad
            if gamma.requires_grad:
                gamma._accum(gamma._unbroadcast(g * xhat, gamma.data.shape))
            if beta.requires_grad:
                beta._accum(beta._unbroadcast(g, beta.data.shape))
            if self.requires_grad:
                dxhat = g * gamma.data
                dvar_term = (dxhat * xhat).sum(axis=-1, keepdims=True)
                dmean_term = dxhat.sum(axis=-1, keepdims=True)
                dx = inv * (dxhat - dmean_term / D - xhat * dvar_term / D)
                self._accum(dx)
        out._backward = _backward
        return out

    # ---- graph -----------------------------------------------------------

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def gather_rows(table: Tensor, idx: np.ndarray) -> Tensor:
    """Look up rows of `table` [V, D] at integer indices `idx` (any shape).
    Returns a Tensor of shape idx.shape + (D,). Backward is a scatter-add, exactly
    the embedding-table update the Session 7 lesson describes."""
    idx = np.asarray(idx)
    out = Tensor(table.data[idx.reshape(-1)].reshape(idx.shape + (table.data.shape[1],)),
                 table.requires_grad, (table,), "gather")

    def _backward():
        if table.requires_grad:
            g = np.zeros_like(table.data)
            flat_idx = idx.reshape(-1)
            flat_grad = out.grad.reshape(-1, table.data.shape[1])
            np.add.at(g, flat_idx, flat_grad)
            table._accum(g)
    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy. logits [N, V], targets [N] int. Fused for a
    stable, exact gradient (softmax(logits) - onehot(targets)) / N."""
    x = logits.data
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    N = x.shape[0]
    loss_val = -np.log(p[np.arange(N), targets] + 1e-12).mean()
    out = Tensor(loss_val, True, (logits,), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            g = p.copy()
            g[np.arange(N), targets] -= 1.0
            g /= N
            logits._accum(g * out.grad)
    out._backward = _backward
    return out
