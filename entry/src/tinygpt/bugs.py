"""Deliberate bugs for chapter 7. Each one is a mistake people really make.

    with tinygpt.bugs.planted("no_shift"):
        ...                          # tinyGPT has the bug in here, and only in here

tinygpt.run_tests(bug="no_shift") runs the test suite with the bug planted.
"""

import contextlib
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

BUGS = {
    "no_mask": "attention can see the future: the causal mask is gone",
    "softmax_wrong_way": "attention's softmax runs down each column instead of along each row",
    "no_shift": "the targets are the inputs themselves, not the inputs moved on by one character",
    "big_init": "the starting weights are 50 times too big (spread 1.0 instead of 0.02)",
    "no_position": "the model never adds its position table, so it can't tell where a character is",
}


def _attention_weights(mask=True, dim=-1):
    """Head.attention_weights, with one of its steps left out or changed."""
    def attention_weights(self, x):
        B, T, C = x.shape
        q, k = self.query(x), self.key(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        if mask and self.causal:
            past = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~past, float("-inf"))
        return F.softmax(scores, dim=dim)
    return attention_weights


def _get_batch_without_shift(ids, block_size, batch_size, device="cpu"):
    starts = torch.randint(len(ids) - block_size, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts])
    return x.to(device), x.clone().to(device)


def _forward_without_positions(self, idx, targets=None):
    """GPT.forward, but the position table is never added."""
    from tinygpt.model import next_char_loss

    B, T = idx.shape
    if T > self.cfg.block_size:
        raise ValueError(f"got {T} characters, but block_size is {self.cfg.block_size}")
    x = self.drop(self.tok_emb(idx))
    x = self.ln_f(self.blocks(x))
    logits = self.lm_head(x)
    loss = None if targets is None else next_char_loss(logits, targets)
    return logits, loss


def _big_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def _patches(name):
    from tinygpt import data
    from tinygpt.model import GPT, Head

    if name == "no_mask":
        return [(Head, "attention_weights", _attention_weights(mask=False))]
    if name == "softmax_wrong_way":
        return [(Head, "attention_weights", _attention_weights(dim=-2))]
    if name == "no_position":
        return [(GPT, "forward", _forward_without_positions)]
    if name == "no_shift":
        # swap the code inside get_batch itself, so every name that points at it gets the bug
        return [(data.get_batch, "__code__", _get_batch_without_shift.__code__)]
    if name == "big_init":
        return [(GPT, "_init_weights", staticmethod(_big_init))]
    raise ValueError(f"no bug called {name!r}; choose from: {', '.join(BUGS)}")


def plant(name):
    """Put the bug into tinyGPT. Returns a function that takes it out again."""
    patches = _patches(name)
    originals = [(owner, attr, vars(owner)[attr] if attr in vars(owner) else getattr(owner, attr))
                 for owner, attr, _ in patches]
    for owner, attr, replacement in patches:
        setattr(owner, attr, replacement)

    def remove():
        for owner, attr, original in originals:
            setattr(owner, attr, original)
    return remove


@contextlib.contextmanager
def planted(name):
    """Have the bug inside a `with` block, and only there."""
    remove = plant(name)
    try:
        yield
    finally:
        remove()
