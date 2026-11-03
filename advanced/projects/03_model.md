# Project 3 · The model

The transformer, from tensors up. It is the decoder used, with small changes, by most
open language models today: pre-norm blocks with RMSNorm, rotary position embeddings,
SwiGLU feed-forward layers, and an output layer that shares the embedding table. The entry
track's model had a position table, LayerNorm and a plain feed-forward layer; every change
here has a reason, and finding it is part of the project.

**You write**, in `src/scratchgpt/layers.py`: `Linear`, `Embedding`, `RMSNorm`,
`softmax`, `cross_entropy`, `dropout`, `rope_frequencies`, `apply_rope`,
`causal_attention`, `Attention`, `FeedForward`; in `model.py`: `Block`, `GPT`,
`count_parameters`, `parameter_formula`.
**The rule:** tensor operations and autograd only. No `torch.nn.Linear`, no
`torch.nn.functional`, no `torch.softmax`. `tests/test_rules.py` reads your code and says
where a rule is broken. The tests themselves use PyTorch's versions as the answer key.
**Done when** `python -m pytest tests/test_03_model.py tests/test_rules.py` passes.

## Predict

1. An untrained model with 8,192 tokens guesses about evenly. What loss will it start at?
2. How many parameters does `configs/base.yaml` have? Work it out on paper from the
   config before you write `parameter_formula`. What share is the embedding table?
3. In `test_untrained_gpt_is_close_to_uniform_guessing` the targets are random tokens.
   If the targets were the input tokens themselves, would the untrained loss be higher,
   lower or the same? (Hint: the output layer *is* the embedding table.)
4. Rotary embeddings never add anything to the token vectors. How can the model know
   where a token is?

## Build

The tests go from the smallest part to the whole model; follow them.

- **`Linear`, `Embedding`, `RMSNorm`.** Small, but get the shapes and the initialisation
  exactly right: everything else stands on them.
- **`softmax` and `cross_entropy`.** Subtract the maximum before `exp`. The tests use
  scores of 1,000, which overflow otherwise.
- **`apply_rope`.** `slow_rope` in the tests does it one pair at a time with rotation
  matrices; yours must match it without loops. Keep the pairs as neighbours (0 and 1, 2
  and 3, ...). Other code bases pair the first half with the second half instead; both
  work, but they don't mix.
- **`causal_attention`.** It must handle fewer queries than keys, with the queries at
  the end: you'll need that for the KV cache in project 5. Compute the softmax in float32.
- **`Attention`** builds queries, keys and values with one `Linear`, splits them into
  heads, rotates, attends, and joins the heads back. Leave the cache branch for project 5.
- **`GPT`** ties it together. The rope tables are buffers, so they move to the GPU with
  the model but aren't saved.
- **`parameter_formula`** last. Five configs check it against `count_parameters`.

## Run

No training yet: look at the model on its own.

```bash
python scripts/inspect_model.py
```

It prints each config's parameters by part, checks `parameter_formula`, and gives the
untrained loss on random targets and with the inputs as targets.

## Investigate

1. Check Predict 1 and 3 in that output. Explain the difference between the two losses.
2. **Why the smaller start for output layers?** Build a 24-layer model and print the
   standard deviation of the residual stream after each block, with the output layers at
   `init_std / sqrt(2 * n_layer)` and then at `init_std`. (A forward hook on each block
   does it.) What happens to the activations of a deep model without it?
3. Why pre-norm? Each block adds `attention(norm(x))` to `x` rather than normalising the
   sum. Follow the gradient of the loss with respect to the first block's input through
   the residual additions: what path does it have that post-norm doesn't?
4. What share of `base`'s parameters is attention, feed-forward, embedding? Compare with
   the entry track's model (chapter 5).

## Stretch

- **Grouped-query attention.** Let groups of query heads share one key and value head
  (add `n_kv_head` to the config). How many parameters and how much KV cache does it save?
  Write a test that it gives the same output as full attention when every group has one
  query head.
- Replace RMSNorm with LayerNorm (written by you) and train `small` both ways in project 4.
  Is there a difference you can measure?
