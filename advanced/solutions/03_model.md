# Project 3 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/layers.py`](scratchgpt/layers.py) and
[`scratchgpt/model.py`](scratchgpt/model.py).

## Predict 1 to 3, measured

- **Starting loss.** ln 8,192 = 9.01. The untrained `base` model (seed 0, random targets)
  gives 9.12: random weights make slightly confident random guesses, which score a little
  worse than an even spread.
- **Parameters of `base`:** 29,893,120.

  | part | parameters | share |
  |---|---|---|
  | embedding table (also the output layer) | 4,194,304 | 14% |
  | attention, 8 blocks × 4 × 512² | 8,388,608 | 28% |
  | feed-forward, 8 blocks × 3 × 512 × 1,408 | 17,301,504 | 58% |
  | norms | 8,704 | 0.03% |

- **Targets equal to the inputs.** The untrained `base` model's loss falls from 9.12 to
  7.22. The output layer is the embedding table, so the logit for token *t* is
  h · E[*t*], and h still looks a lot like the input's own embedding E[*t*], so the model
  starts out favouring a copy of the current token. That's harmless (training removes it in
  a few steps), but it means "my loss is already below ln V" isn't proof of learning.

## What's easy to get wrong

- **`softmax` without the maximum.** `exp(1000)` is infinite; the test feeds it scores of
  1,000.
- **Rope pairing.** Neighbours (0, 1), (2, 3), ... here. Code that pairs dimension *i* with
  *i* + D/2 is also correct but gives different numbers: never mix weights between the two.
- **The causal mask with a cache.** With *Tq* queries and *Tk* keys, query *i* sits at
  position *Tk* − *Tq* + *i*. `triu(Tk - Tq + 1)` masks exactly the keys after it.
- **Splitting heads.** `view(B, T, H, D).transpose(1, 2)`, not `view(B, H, T, D)`: the
  second scrambles positions and heads together, and every shape still fits.
- **Float32 where it matters.** RMSNorm squares its input and the softmax exponentiates;
  in 16-bit both overflow or lose precision. Compute them in float32 and cast back.

## Investigate 2: why the smaller start

Each block adds its output to the residual stream. With *L* blocks the sum's variance grows
with the number of additions, about 2*L* of them (attention and feed-forward). Dividing the
output layers' standard deviation by √(2*L*) keeps the stream's scale roughly the same at
any depth (GPT-2 does this). Without it, a 24-layer model's activations grow block by
block, and the last blocks start out saturated.
