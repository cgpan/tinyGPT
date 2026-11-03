# Project 7 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/evaluate.py`](scratchgpt/evaluate.py).

## The probe that never trains

The natural way to write the logistic loss, `z.clamp(min=0) + log1p(exp(-|z|))`, has the
right value everywhere and the wrong gradient at exactly z = 0: PyTorch gives both `clamp`
and `abs` a gradient of 0 there. A probe that starts with all weights at zero has z = 0 for
every review, so its gradient is exactly zero and it never moves. (This happened while the
solution was being written: the probe scored 52% on separable data.) Writing max(z, 0) as
(z + |z|) / 2 gives the true gradient of 1/2 at zero. Starting from small random weights
would also hide the problem, which is a reason to know it is there.

## What we measured

| | RTX 4070 SUPER | M5 Max |
|---|---|---|
| bits per byte, `base`, test reviews | 1.335 | 1.341 |
| bits per byte, token frequencies | 2.524 | 2.524 |
| bits per byte, gzip | 3.252 | 3.252 |
| sentiment: probe on `base`'s features | 85.9% | 85.0% |
| sentiment: probe on word counts | 71.2% | 71.2% |
| training reviews continued exactly | 0.0% | 0.0% |

## Investigate 3: which beats which

`base`'s features beat word counts by about 14 points with this probe, but don't read too
much into that. The word-count probe is weak: 8,192 features learned from 4,000 reviews, with the
same light weight decay as the model's 512. A token that appears in a handful of training
reviews becomes, after standardising, a huge feature that the probe can fit to those
reviews alone; that is the likely reason. Word-count classifiers tuned on all 25,000
labelled reviews reach the high 80s (Maas et al., 2011). What the comparison does show:
with only 4,000 labelled reviews, the features of a model pretrained without labels support
85% accuracy. Whether they beat a tuned word-count classifier is the stretch goal.

## Investigate 1: validation loss against bits per byte

They should agree closely but not exactly: `base`'s validation loss of 3.715 nats per
token is 1.347 bits per byte, and `bits_per_byte` gave 1.335 on the test reviews. Training's validation loss is a mean over random
windows, includes the end-of-text tokens, and is in nats per token; `bits_per_byte` reads
the whole test text once, skips end-of-text tokens, and divides by bytes. Convert one into
the other with bytes per token and ln 2, and the difference that remains is those details.

## Investigate 5: the trick question

Bits per byte measures the probabilities the model *gives* the real text; temperature
changes only how the sampler *draws* from them. Unless you apply the temperature to the
logits inside `bits_per_byte`, it doesn't change at all. The entry track's chapter 6 did
apply it, and found that its character-level model scored best on new text slightly
above temperature 1.

## What's easy to get wrong

- **Counting end-of-text tokens as text.** They stand for no bytes; including their loss
  but not their bytes makes bits per byte too high.
- **Averaging the padding.** Padding with end-of-text tokens and averaging over all
  positions makes a short review's features depend on the longest one in its batch.
- **Testing the probe on the reviews it learned from.** Train on training reviews, test on
  test reviews. The two sets are about different films, which also checks the probe isn't
  just recognising titles.
