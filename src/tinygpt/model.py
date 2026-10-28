"""The models.

BigramLM is the counting bigram again, but learned by gradient descent.
GPT is the transformer. Read it from the bottom of the file up: GPT stacks
Blocks, a Block is attention plus a small feed-forward network, and attention
is several Heads working side by side.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def count_params(model):
    """How many numbers the model learns."""
    return sum(p.numel() for p in model.parameters())


def next_char_loss(logits, targets):
    """Cross-entropy: the average of -log(probability given to the right next character)."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


class BigramLM(nn.Module):
    """One table of scores: row a holds a score for every character that might follow a."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)
        nn.init.zeros_(self.table.weight)       # all scores equal: every next character equally likely

    def forward(self, idx, targets=None):
        logits = self.table(idx)                # (B, T, vocab_size): look up the row for each character
        loss = None if targets is None else next_char_loss(logits, targets)
        return logits, loss


class Head(nn.Module):
    """One head of causal self-attention.

    Every position asks a question (its query) and advertises what it holds
    (its key). Each position then takes a weighted average of the values of
    earlier positions, weighting most heavily the ones whose keys best match
    its query. It may only look backwards.
    """

    def __init__(self, cfg, head_size):
        super().__init__()
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.causal = cfg.causal

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)          # each (B, T, head_size)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])    # (B, T, T): how well query t matches key s
        if self.causal:
            past = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~past, float("-inf"))        # position t may not look at any s > t
        weights = F.softmax(scores, dim=-1)                          # each row now sums to 1
        weights = self.dropout(weights)
        return weights @ v                                           # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several heads side by side, each free to look for something different."""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)    # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small two-layer network that works on each position separately."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Attention (gather information from earlier positions), then feed-forward
    (process it). Each result is added back onto x rather than replacing it."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Characters in, a score for every possible next character out."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)      # what each character is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_emb else None  # where it is
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Small random starting weights make every score close to zero, so the
        # untrained model finds all characters equally likely: its loss starts
        # at about ln(vocab_size) = 4.17.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"got {T} characters, but block_size is {self.cfg.block_size}")
        x = self.tok_emb(idx)                                        # (B, T, n_embd)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, vocab_size)
        loss = None if targets is None else next_char_loss(logits, targets)
        return logits, loss
