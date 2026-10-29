# Chapter 6 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Your own writing loop

```python
import torch

def my_greedy(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        context = idx[:, -model.cfg.block_size:]               # only what the model can read
        logits, _ = model(context)
        scores = logits[:, -1, :]                              # the next character's scores
        next_id = torch.argmax(scores, dim=-1, keepdim=True)   # the likeliest, as a (B, 1) tensor
        idx = torch.cat([idx, next_id], dim=1)
    return idx

tinygpt.check.greedy(my_greedy)
```

## Step by step

1. The model was trained on windows of `block_size` characters and its position table has
   no more rows, so it can't read further back than that. The slice keeps the last
   `block_size` characters of every row.
2. The model returns scores for every position. Only the last one predicts a character
   that hasn't been written yet.
3. `torch.argmax(..., keepdim=True)` picks the highest score in each row and keeps the
   result two-dimensional, so it can be glued onto `idx`.
4. `torch.cat(..., dim=1)` adds one column: one new character for each row.

## Going further: temperature

Replace the `argmax` line with a dice roll:

```python
from torch.nn import functional as F

def my_generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

That is tinyGPT's `generate`, without `top_k`. Compare the two with `tinygpt.show(tinygpt.generate)`.
