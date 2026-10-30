# Project 5 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/perf.py`](scratchgpt/perf.py), and the cache in
[`scratchgpt/sampling.py`](scratchgpt/sampling.py) and `Attention.forward` in
[`scratchgpt/layers.py`](scratchgpt/layers.py).

## Predict 1 and 2, worked out

- **Compute for `base`.** 29,884,416 parameters in matrix multiplications, so 6 × that is
  179.3M FLOPs per token, plus 25.2M for attention's own products (12 × 8 layers × 512 ×
  512): 204.5M FLOPs per token. 5,000 steps of 32,768 tokens are 163.8M tokens, so
  3.35 × 10¹⁶ FLOPs. At 20 TFLOPS that is 28 minutes.
- **Memory for `base` with a batch of 16** (the project's estimate): weights 0.12 GB,
  gradients 0.12 GB, AdamW's moments 0.24 GB, activations 4.16 GB. The activations win by
  far, and they are what a bigger batch or a longer context costs.
- **Context 512 to 1,024.** Each sequence's score matrices grow 4 times (context²); per
  token, twice. With half the batch (8), activations go from 4.16 to 5.77 GB in the estimate.

## Predict 4: the cache

Without a cache, writing 200 tokens after a 1-token prompt processes the first token 200
times, and the whole sequence's cost grows with the square of its length. With one, every
token is processed once, and each step costs one token's worth plus reading the cache.

## What's easy to get wrong

- **Positions with a cache.** The new token's position is the cache's length, not 0. Rotate
  it with `cos[start:start + T]`, or the model sees every new token at position 0.
- **The mask with a cache.** With one query, it sees every key: there is nothing in the
  future. PyTorch's `is_causal=True` aligns the mask to the top left, which would let the
  query see only the first key; `fused_causal_attention` handles that case separately.
- **Counting the embedding as compute.** Looking up a row multiplies nothing. The output
  layer does multiply, even when it shares the table.
- **MFU against the wrong peak.** Use the 16-bit dense number, without "sparsity". A GPU's
  marketing peak is often twice that.
