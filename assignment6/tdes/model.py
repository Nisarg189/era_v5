#!/usr/bin/env python3
"""A small decoder-only transformer, and the per-token loss the ledgers need.

The model is deliberately small.  Session 6 is about the data system, and the model
is here to be a real consumer of it: real gradients, real cross entropy per token,
real checkpoints.  Nothing downstream would be meaningful if the losses in the
learning ledger were invented.

Two things distinguish this from a textbook implementation, and both come from
packing.  Attention is masked by segment as well as causally, so two samples packed
into one window cannot see each other.  Loss is returned per token rather than
averaged, so a loss value can be attributed back to the shard, the document and the
token id that produced it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, attn_bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.mlp(self.ln2(x)))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=192, n_layer=4, n_head=4, max_seq=512, drop=0.0):
        super().__init__()
        self.max_seq = max_seq
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.blocks = nn.ModuleList([Block(d, n_head, drop) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, position_ids, segment_ids):
        """``segment_ids`` blocks attention between packed samples."""
        B, T = input_ids.shape
        x = self.tok(input_ids) + self.pos(position_ids.clamp(max=self.max_seq - 1))
        idx = torch.arange(T, device=input_ids.device)
        causal = idx[None, :, None] >= idx[None, None, :]
        same = segment_ids[:, :, None] == segment_ids[:, None, :]
        allowed = (causal & same).unsqueeze(1)                       # B,1,T,T
        # a row with nothing visible would produce NaN, so each token always sees itself
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=input_ids.device)[None, None]
        for b in self.blocks:
            x = b(x, allowed)
        return self.head(self.ln_f(x))

    def token_losses(self, input_ids, position_ids, segment_ids, loss_mask):
        """Cross entropy for every graded position, and the mean over them.

        Returns (per_token_loss, mask, mean_loss).  ``per_token_loss`` is aligned to
        the target position, so entry t is the loss paid predicting token t+1.
        """
        logits = self(input_ids, position_ids, segment_ids)
        tgt = input_ids[:, 1:]
        lp = logits[:, :-1]
        m = loss_mask[:, 1:].float()
        # a target must belong to the same packed sample as the token predicting it,
        # otherwise the last token of one sample is graded on the first of the next
        m = m * (segment_ids[:, 1:] == segment_ids[:, :-1]).float()
        per = F.cross_entropy(lp.reshape(-1, lp.size(-1)), tgt.reshape(-1),
                              reduction="none").view(tgt.shape)
        denom = m.sum().clamp(min=1.0)
        return per, m, (per * m).sum() / denom


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))
