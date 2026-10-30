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
        # BEGIN SOLUTION 3
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)
        # END SOLUTION

    def forward(self, x):
        # BEGIN SOLUTION 3
        return x @ self.weight.T
        # END SOLUTION


class Embedding(nn.Module):
    """A table of n vectors of size d (the parameter `weight`, shape (n, d), initialised
    like Linear's). forward(ids) looks up one row per id: shape (*ids.shape, d)."""

    def __init__(self, n, d, std=0.02):
        super().__init__()
        # BEGIN SOLUTION 3
        self.weight = nn.Parameter(torch.empty(n, d))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)
        # END SOLUTION

    def forward(self, ids):
        # BEGIN SOLUTION 3
        return self.weight[ids]
        # END SOLUTION


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight, over the last dimension; weight (size d) starts at
    one. Compute in float32 and return the input's dtype: in 16-bit, squaring overflows."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        # BEGIN SOLUTION 3
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        # END SOLUTION

    def forward(self, x):
        # BEGIN SOLUTION 3
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)
        # END SOLUTION


def softmax(x, dim=-1):
    """exp(x) / sum(exp(x)) along dim. Subtract the maximum first, or exp overflows for large
    scores; the result is the same."""
    # BEGIN SOLUTION 3
    shifted = x - x.amax(dim=dim, keepdim=True)
    e = shifted.exp()
    return e / e.sum(dim=dim, keepdim=True)
    # END SOLUTION


def cross_entropy(logits, targets):
    """The mean of -log softmax(logits)[target] over the rows. logits: (N, V); targets: (N,)
    of ids. Compute in float32 as logsumexp(row) - row[target], where
    logsumexp(row) = m + log(sum(exp(row - m))) and m is the row's maximum."""
    # BEGIN SOLUTION 3
    logits = logits.float()
    m = logits.amax(dim=-1, keepdim=True)
    logsumexp = m.squeeze(-1) + (logits - m).exp().sum(dim=-1).log()
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - picked).mean()
    # END SOLUTION


def dropout(x, p, training):
    """In training, zero each entry with probability p and divide the rest by (1 - p), so the
    expected value doesn't change. Outside training, or when p is 0, return x unchanged."""
    # BEGIN SOLUTION 3
    if not training or p == 0:
        return x
    keep = torch.rand_like(x) >= p
    return x * keep / (1 - p)
    # END SOLUTION


def rope_frequencies(head_dim, context, theta=10000.0):
    """The rotary embedding tables: (cos, sin), each of shape (context, head_dim // 2), in
    float32, with angle[t, i] = t * theta ** (-2i / head_dim) for position t and pair i."""
    # BEGIN SOLUTION 3
    pair = torch.arange(head_dim // 2, dtype=torch.float32)
    speed = theta ** (-2 * pair / head_dim)
    angle = torch.outer(torch.arange(context, dtype=torch.float32), speed)
    return angle.cos(), angle.sin()
    # END SOLUTION


def apply_rope(x, cos, sin):
    """Rotate each pair of neighbours (x[..., 2i], x[..., 2i+1]) of the vector at position t by
    angle[t, i]:  (a, b) -> (a cos - b sin, a sin + b cos).  x: (..., T, head_dim);
    cos, sin: (T, head_dim // 2), already cut to x's positions. Returns x's shape and dtype.

    Rotating queries and keys this way makes their dot product depend only on how far apart
    two positions are, not where they are.
    """
    # BEGIN SOLUTION 3
    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)
    # END SOLUTION


def causal_attention(q, k, v):
    """softmax(q k^T / sqrt(head_dim)) v, where each query sees only keys at its own position
    or earlier. q: (B, H, Tq, D); k, v: (B, H, Tk, D), with Tq <= Tk: the queries are the last
    Tq positions (Tq < Tk when a KV cache holds the earlier keys, in project 5). Compute the
    softmax in float32 and return v's dtype."""
    # BEGIN SOLUTION 3
    t_q, t_k = q.shape[-2], k.shape[-2]
    scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(q.shape[-1])
    future = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).triu(t_k - t_q + 1)
    weights = softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    return weights.to(v.dtype) @ v
    # END SOLUTION


class Attention(nn.Module):
    """Multi-head causal self-attention with rotary positions.

    One Linear, self.qkv, makes queries, keys and values together (d_model -> 3 * d_model,
    in that order); they are cut into n_head heads of head_dim each; queries and keys are
    rotated with apply_rope; causal_attention mixes the values; the heads are joined back
    and an output Linear, self.out (d_model -> d_model), follows, then dropout.
    With cfg.fused_attention (project 5), call perf.fused_causal_attention instead of
    causal_attention: same inputs, same meaning, PyTorch's fast kernel.

    The output Linear starts smaller, with std = init_std / sqrt(2 * n_layer): each block adds
    its output to the residual stream, and this keeps the sum from growing with depth.
    """

    def __init__(self, cfg, layer):
        super().__init__()
        self.layer = layer                    # which block this is, for the KV cache
        self.n_head = cfg.n_head
        self.p = cfg.dropout
        self.fused = cfg.fused_attention
        # BEGIN SOLUTION 3
        self.qkv = Linear(cfg.d_model, 3 * cfg.d_model, cfg.init_std)
        self.out = Linear(cfg.d_model, cfg.d_model, cfg.init_std / math.sqrt(2 * cfg.n_layer))
        # END SOLUTION

    def forward(self, x, cos, sin, cache=None):
        """x: (B, T, d_model). cos, sin: the full rope tables. cache (project 5): a KVCache, or
        None. With a cache, x holds only the new positions: they start at
        cache.length(self.layer), and cache.update(self.layer, k, v) returns all keys and
        values so far."""
        # BEGIN SOLUTION 3
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        start = 0 if cache is None else cache.length(self.layer)
        q = apply_rope(q, cos[start : start + T], sin[start : start + T])
        k = apply_rope(k, cos[start : start + T], sin[start : start + T])
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        if self.fused:
            from scratchgpt.perf import fused_causal_attention
            y = fused_causal_attention(q, k, v)
        else:
            y = causal_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, T, C)
        return dropout(self.out(y), self.p, self.training)
        # END SOLUTION


class FeedForward(nn.Module):
    """SwiGLU: w2( silu(w1 x) * w3 x ), then dropout. silu(z) = z * sigmoid(z).
    w1 and w3: d_model -> cfg.ff_dim; w2: cfg.ff_dim -> d_model, with the smaller start
    (init_std / sqrt(2 * n_layer)) for the same reason as Attention's output."""

    def __init__(self, cfg):
        super().__init__()
        self.p = cfg.dropout
        # BEGIN SOLUTION 3
        self.w1 = Linear(cfg.d_model, cfg.ff_dim, cfg.init_std)
        self.w3 = Linear(cfg.d_model, cfg.ff_dim, cfg.init_std)
        self.w2 = Linear(cfg.ff_dim, cfg.d_model, cfg.init_std / math.sqrt(2 * cfg.n_layer))
        # END SOLUTION

    def forward(self, x):
        # BEGIN SOLUTION 3
        h = self.w1(x)
        return dropout(self.w2(h * torch.sigmoid(h) * self.w3(x)), self.p, self.training)
        # END SOLUTION
