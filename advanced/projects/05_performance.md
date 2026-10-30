# Project 5 · Performance

How much compute does a run need, how much memory, and how much of the GPU are you
actually using? Teams plan runs with these numbers before they spend a GPU-hour. You'll
also build the KV cache, the trick that makes writing text fast.

**You write**, in `src/scratchgpt/perf.py`: `matmul_parameters`, `flops_per_token`,
`mfu`, `training_memory`; in `sampling.py`: `KVCache`, and the `use_cache` path of
`generate`; and the cache branch of `Attention.forward` in `layers.py`.
**Given:** `fused_causal_attention`, PyTorch's FlashAttention-style kernel, the one place
`torch.nn.functional` is allowed, so you can compare it with yours; `PEAK_TFLOPS`, a few
GPUs' peak speeds.
**Done when** `python -m pytest tests/test_05_performance.py` passes and you have filled in
the benchmark table below for your machine.

## Predict

1. Training `base` costs about 6 floating-point operations per parameter per token.
   Training it for 164M tokens: how many FLOPs? At 20 TFLOPS (trillion per second), how
   long?
2. Which takes more GPU memory when training `base` with a batch of 16: the weights,
   gradients and AdamW state together, or the activations?
3. Your attention keeps the whole context × context score matrix for every head.
   Doubling the context from 512 to 1,024 multiplies that memory by how much?
4. Writing 200 tokens: how many times does the model process the first token without a
   KV cache? With one?

## Build

- **`matmul_parameters` and `flops_per_token`.** The 6 is 2 for the forward pass (a
  multiply and an add per parameter) and 4 for the backward pass. Attention adds its own
  matrix products, which grow with the context.
- **`training_memory`.** Count bytes: float32 weights, gradients, and Adam's two moments
  are 16 bytes per parameter before any activation. The activation estimate is rough;
  you'll check it against the real number.
- **The KV cache.** At each new token, only its query, key and value are new. Keep every
  layer's keys and values, and feed the model just the newest token. Positions continue
  from the cache's length, which is why rope tables are indexed by `start`.
  `test_cached_model_gives_the_same_logits` compares a cached run with a full one.

## Run

```bash
python scripts/benchmark.py
python scripts/benchmark.py configs/base.yaml fused_attention=true
python scripts/benchmark.py configs/base.yaml compile=true
python scripts/benchmark.py configs/base.yaml context=1024 batch_size=8
```

`compile=true` needs PyTorch's compiler, which works on Linux and Colab; on Windows it may
not. Fill in your table:

| config | tokens/s | MFU | memory (measured) | memory (your estimate) | full run |
|---|---|---|---|---|---|
| small | | | | | |
| base | | | | | |
| base, fused attention | | | | | |
| base, compiled | | | | | |

## Investigate

1. How far is your MFU from 100%, and from the 40–50% that well-tuned large training runs
   reach? Where does the rest go? Profile 5 steps with `torch.profiler` and sort by GPU
   time: which operations take the longest?
2. How close was `training_memory` to the measured peak? What does the estimate leave out?
3. Time `write` with and without the cache (`use_cache=False`) for 50, 200 and 500 tokens.
   How does each grow?
4. Find the largest batch of `base` that fits on your GPU, with your attention and with
   the fused one. Why do they differ so much?

## Stretch

- **Write FlashAttention's idea in PyTorch.** Process the keys in blocks of 64, keeping a
  running maximum and a running sum for the softmax ("online softmax"), so the full score
  matrix never exists. Test it against `causal_attention`. It won't be fast in Python;
  the point is that it is exact.
- **Activation checkpointing.** Recompute each block's activations in the backward pass
  instead of keeping them (`torch.utils.checkpoint`). How much memory does it save, and
  how much time does it cost?
