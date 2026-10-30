"""Project 3: the transformer's parts, from tensors up.

Allowed: tensor operations and autograd (torch.*, tensor methods, nn.Module, nn.Parameter,
nn.ModuleList, nn.init). Not allowed: PyTorch's ready-made layers and functions
(torch.nn.Linear, torch.nn.functional, torch.softmax, torch.logsumexp, ...).
tests/test_rules.py checks this.
"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    """y = x W^T, without a bias. W has shape (d_out, d_in), drawn from a normal distribution
    with standard deviation std, cut off at 3 standard deviations (nn.init.trunc_normal_)."""

    def __init__(self, d_in, d_out, std=0.02):
        super().__init__()
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, x):
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


class Embedding(nn.Module):
    """A table of n vectors of size d (the parameter `weight`, shape (n, d), initialised
    like Linear's). forward(ids) looks up one row per id: shape (*ids.shape, d)."""

    def __init__(self, n, d, std=0.02):
        super().__init__()
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, ids):
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight, over the last dimension; weight (size d) starts at
    one. Compute in float32 and return the input's dtype: in 16-bit, squaring overflows."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, x):
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


def softmax(x, dim=-1):
    """exp(x) / sum(exp(x)) along dim. Subtract the maximum first, or exp overflows for large
    scores; the result is the same."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def cross_entropy(logits, targets):
    """The mean of -log softmax(logits)[target] over the rows. logits: (N, V); targets: (N,)
    of ids. Compute in float32 as logsumexp(row) - row[target], where
    logsumexp(row) = m + log(sum(exp(row - m))) and m is the row's maximum."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def dropout(x, p, training):
    """In training, zero each entry with probability p and divide the rest by (1 - p), so the
    expected value doesn't change. Outside training, or when p is 0, return x unchanged."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def rope_frequencies(head_dim, context, theta=10000.0):
    """The rotary embedding tables: (cos, sin), each of shape (context, head_dim // 2), in
    float32, with angle[t, i] = t * theta ** (-2i / head_dim) for position t and pair i."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def apply_rope(x, cos, sin):
    """Rotate each pair of neighbours (x[..., 2i], x[..., 2i+1]) of the vector at position t by
    angle[t, i]:  (a, b) -> (a cos - b sin, a sin + b cos).  x: (..., T, head_dim);
    cos, sin: (T, head_dim // 2), already cut to x's positions. Returns x's shape and dtype.

    Rotating queries and keys this way makes their dot product depend only on how far apart
    two positions are, not where they are.
    """
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def causal_attention(q, k, v):
    """softmax(q k^T / sqrt(head_dim)) v, where each query sees only keys at its own position
    or earlier. q: (B, H, Tq, D); k, v: (B, H, Tk, D), with Tq <= Tk: the queries are the last
    Tq positions (Tq < Tk when a KV cache holds the earlier keys, in project 5). Compute the
    softmax in float32 and return v's dtype."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


class Attention(nn.Module):
    """Multi-head causal self-attention with rotary positions.

    One Linear makes queries, keys and values together (d_model -> 3 * d_model); they are
    cut into n_head heads of head_dim each; queries and keys are rotated with apply_rope;
    causal_attention mixes the values; the heads are joined back and an output Linear
    (d_model -> d_model) follows, then dropout.

    The output Linear starts smaller, with std = init_std / sqrt(2 * n_layer): each block adds
    its output to the residual stream, and this keeps the sum from growing with depth.
    """

    def __init__(self, cfg, layer):
        super().__init__()
        self.layer = layer                    # which block this is, for the KV cache
        self.n_head = cfg.n_head
        self.p = cfg.dropout
        self.fused = cfg.fused_attention
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, x, cos, sin, cache=None):
        """x: (B, T, d_model). cos, sin: the full rope tables. cache (project 5): a KVCache, or
        None. With a cache, x holds only the new positions: they start at
        cache.length(self.layer), and cache.update(self.layer, k, v) returns all keys and
        values so far."""
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


class FeedForward(nn.Module):
    """SwiGLU: w2( silu(w1 x) * w3 x ), then dropout. silu(z) = z * sigmoid(z).
    w1 and w3: d_model -> cfg.ff_dim; w2: cfg.ff_dim -> d_model, with the smaller start
    (init_std / sqrt(2 * n_layer)) for the same reason as Attention's output."""

    def __init__(self, cfg):
        super().__init__()
        self.p = cfg.dropout
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, x):
        raise NotImplementedError("write this in project 3: see projects/03_model.md")
