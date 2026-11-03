# Chapter 4 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Your own attention

```python
import math
import torch
from torch.nn import functional as F

def my_attention(q, k, v):
    T = q.shape[1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])   # (B, T, T)
    past = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~past, float("-inf"))           # hide the future
    weights = F.softmax(scores, dim=-1)                         # each row sums to 1
    return weights @ v                                          # (B, T, head_size)

tinygpt.check.attention(my_attention)
```

## Step by step

- `k.transpose(-2, -1)` swaps the last two dimensions of `k`, so `q @ k.transpose(...)`
  multiplies every query with every key: a T × T table of scores for each sequence.
- Dividing by `math.sqrt(k.shape[-1])`, the square root of the head size, keeps the
  scores from growing as heads get wider.
- `torch.tril(...)` is True on and below the diagonal. `~past` is True above it, the
  future, and `masked_fill` puts `-inf` there. The softmax turns `-inf` into exactly 0,
  so no weight at all goes to future positions.
- The softmax over the last dimension makes each row sum to 1.
- `weights @ v` gives each position a weighted average of the values it is allowed to see.

Compare with `Head` in `tinygpt/model.py`: the same four steps, split into
`attention_weights` and `forward`, with the queries, keys and values computed from the
characters by three `Linear` layers.
