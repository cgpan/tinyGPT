# Chapter 5 · Solutions to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Part 1: counting the parameters

```python
def param_count(vocab_size, block_size, n_layer, n_embd):
    d = n_embd
    embeddings = vocab_size * d + block_size * d           # tok_emb, pos_emb
    attention = 3 * d * d + (d * d + d)                    # query, key, value of all heads; proj
    feed_forward = (d * 4 * d + 4 * d) + (4 * d * d + d)   # Linear(d, 4d), Linear(4d, d)
    layer_norms = 2 * (2 * d)                              # ln1, ln2
    per_block = attention + feed_forward + layer_norms
    final = 2 * d + (d * vocab_size + vocab_size)          # ln_f, lm_head
    return embeddings + n_layer * per_block + final

tinygpt.check.param_count(param_count)
```

Adding up `per_block` gives **12d² + 10d**: 4d² in attention, 8d² in the feed-forward
network, and a few small terms. For tinyGPT (d = 384, six blocks) that is 1,773,312 per
block and **10,739,777** in all. Two-thirds of the model is feed-forward, not attention.

**Why `n_head` doesn't appear.** Each head gets `head_size = n_embd / n_head` columns of
the query, key and value layers. More heads means narrower heads, so between them they
always have `n_embd` columns: 3d² numbers, however the work is split.

## Part 2: the three-minute race

There is no single answer: it depends on how many steps your machine fits into three
minutes. Some ideas worth trying, one at a time:

- **Wider:** `n_embd=192` or `256` (keep it divisible by `n_head`).
- **Longer memory:** `block_size=128`. Chapter 4 showed what memory is worth.
- **A little dropout,** such as `0.1`, once the model is big enough to start memorising.
- **Not too big:** a model that is too large gets too few steps in three minutes and
  finishes worse than a smaller one.

Keep `learning_rate` near `0.001`, and change one thing at a time so you know what helped.
