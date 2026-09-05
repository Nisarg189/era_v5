"""
A small pre-LayerNorm transformer language model, built on the numpy autograd.

Two axes are configurable, and only these two, so that a controlled comparison
isolates one decision at a time:

  input_path : "kron"  -> token vector is codes @ W_in (the Kronecker byte codec
                          plus one shared projection; no per-token rows)
               "dense" -> a learned V x D embedding table

  head       : "dense"      -> a learned V x D output projection (the classic head)
               "tied_kron"  -> logits are the dot product of the hidden state with
                               the SAME codes @ W_in used at the input. This is the
                               reversible-Kronecker head: it introduces no parameter
                               that depends on the vocabulary size, and prediction
                               reduces to nearest-code, which the codec inverts back
                               to a token string.

The headline experiment fixes input_path="kron" and varies only head, so the two
models differ by exactly the output head under test.
"""

import numpy as np
from .autograd import Tensor, gather_rows, cross_entropy
from .codec import KroneckerCodec


def _p(shape, scale, rng):
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=True)


class TinyLM:
    def __init__(self, tokens, d_model=128, block_size=64, n_layer=2,
                 input_path="kron", head="tied_kron", codec=None, seed=0):
        self.tokens = list(tokens)
        self.V = len(self.tokens)
        self.D = d_model
        self.T = block_size
        self.n_layer = n_layer
        self.input_path = input_path
        self.head = head
        self.codec = codec or KroneckerCodec()
        rng = np.random.default_rng(seed)

        # fixed Kronecker codes for the vocabulary (a constant, never trained)
        self.C = Tensor(self.codec.encode_batch(self.tokens))  # [V, 8192]

        self.params = []

        def track(t):
            self.params.append(t)
            return t

        s = 0.02
        if input_path == "kron" or head == "tied_kron":
            # code rows are z-normalised (||code|| = sqrt(code_dim)), so scale the
            # projection init by 1/sqrt(code_dim) to give E = codes @ W_in entries of
            # ~0.02, matching a normal embedding table. Identical for both head types,
            # so the A/B comparison keeps the input path fixed.
            win_s = s / np.sqrt(self.codec.code_dim)
            self.W_in = track(_p((self.codec.code_dim, self.D), win_s, rng))  # [8192, D]
        if input_path == "dense":
            self.Emb = track(_p((self.V, self.D), s, rng))
        self.pos = track(_p((self.T, self.D), s, rng))

        self.layers = []
        for _ in range(n_layer):
            L = {
                "ln1_g": track(Tensor(np.ones(self.D), requires_grad=True)),
                "ln1_b": track(Tensor(np.zeros(self.D), requires_grad=True)),
                "Wq": track(_p((self.D, self.D), s, rng)),
                "Wk": track(_p((self.D, self.D), s, rng)),
                "Wv": track(_p((self.D, self.D), s, rng)),
                "Wo": track(_p((self.D, self.D), s, rng)),
                "ln2_g": track(Tensor(np.ones(self.D), requires_grad=True)),
                "ln2_b": track(Tensor(np.zeros(self.D), requires_grad=True)),
                "W1": track(_p((self.D, 4 * self.D), s, rng)),
                "b1": track(Tensor(np.zeros(4 * self.D), requires_grad=True)),
                "W2": track(_p((4 * self.D, self.D), s, rng)),
                "b2": track(Tensor(np.zeros(self.D), requires_grad=True)),
            }
            self.layers.append(L)

        self.lnf_g = track(Tensor(np.ones(self.D), requires_grad=True))
        self.lnf_b = track(Tensor(np.zeros(self.D), requires_grad=True))

        if head == "dense":
            self.W_out = track(_p((self.V, self.D), s, rng))  # [V, D]
        if head == "tied_kron":
            self.logit_scale = track(Tensor(np.array(1.0), requires_grad=True))

        # causal mask, constant
        mask = np.triu(np.ones((self.T, self.T)) * -1e9, k=1)
        self.mask = Tensor(mask)

    # ---- parameter accounting -------------------------------------------

    def head_param_count(self):
        """Parameters whose count grows with vocabulary size V, on the OUTPUT side."""
        if self.head == "dense":
            return self.V * self.D
        return 0  # tied_kron reuses W_in; nothing scales with V

    def total_param_count(self):
        return int(sum(p.data.size for p in self.params))

    # ---- forward ---------------------------------------------------------

    def _vocab_embeddings(self):
        """E[v] = codes[v] @ W_in, shape [V, D]. Used by the Kronecker input path
        and reused verbatim by the tied head."""
        return self.C @ self.W_in

    def forward(self, idx):
        B, T = idx.shape
        if self.input_path == "kron":
            E = self._vocab_embeddings()
            x = gather_rows(E, idx)
        else:
            x = gather_rows(self.Emb, idx)
        x = x + self.pos.reshape(1, self.T, self.D)

        for L in self.layers:
            h = x.layernorm(L["ln1_g"], L["ln1_b"])
            q = h @ L["Wq"]; k = h @ L["Wk"]; v = h @ L["Wv"]
            scores = (q @ k.transpose_last()) * (1.0 / np.sqrt(self.D))
            scores = scores + self.mask.reshape(1, self.T, self.T)
            a = scores.softmax_lastdim()
            o = a @ v
            x = x + (o @ L["Wo"])
            h2 = x.layernorm(L["ln2_g"], L["ln2_b"])
            ff = ((h2 @ L["W1"]) + L["b1"]).relu()
            ff = (ff @ L["W2"]) + L["b2"]
            x = x + ff

        x = x.layernorm(self.lnf_g, self.lnf_b)

        if self.head == "dense":
            logits = x @ self.W_out.transpose_last()  # [B,T,V]
        else:
            E = self._vocab_embeddings()              # reuse the input codec + W_in
            logits = (x @ E.transpose_last()) * self.logit_scale
        return logits, x

    def loss(self, idx, targets):
        logits, _ = self.forward(idx)
        B, T, V = logits.data.shape
        return cross_entropy(logits.reshape(B * T, V), targets.reshape(-1))


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def clip_global_norm(self, max_norm):
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += float((p.grad ** 2).sum())
        norm = np.sqrt(total)
        if norm > max_norm and norm > 0:
            scale = max_norm / (norm + 1e-12)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= scale
        return norm

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            if self.wd:
                g = g + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
