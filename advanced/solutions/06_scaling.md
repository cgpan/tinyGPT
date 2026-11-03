# Project 6 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/scaling.py`](scratchgpt/scaling.py).

## Predict 1: why loss flattens

Text is not perfectly predictable: even a perfect model is surprised by what a reviewer
decides to say next. That floor is the c in a·C^(−b) + c. On log-log axes the curve is
straight only after subtracting c; plotted raw, it bends towards c. Fitting a straight line
to raw loss predicts that a big enough model reaches zero, which is impossible.

## Predict 3: IMDB's size

96.4 MB of training text at 3.99 bytes per token is 24.2 million tokens. Four passes give
96.6 million, and at 20 tokens per parameter that is a model of 4.8 million parameters:
about `small`. A bigger model trained on this data still does better, but by less and
less, and repeating the data more often gains little. This is the data-constrained regime,
and it is why `plan_run` moves extra compute into parameters once the data runs out.
`base` (29.9M parameters) makes 6.8 passes: at the end its training loss (3.43) is well
below its validation loss (3.72), which was improving by less than 0.01 per 250 steps.

## What the sweep found

On the RTX 4070 SUPER, in 17 minutes:

| width | parameters | tokens | compute (FLOPs) | validation loss |
|---|---|---|---|---|
| 64 | 0.6M | 12.6M | 5.3 × 10¹³ | 5.221 |
| 96 | 1.0M | 20.2M | 1.3 × 10¹⁴ | 4.811 |
| 128 | 1.5M | 29.5M | 2.8 × 10¹⁴ | 4.568 |
| 192 | 2.9M | 58.0M | 1.1 × 10¹⁵ | 4.202 |
| 256, predicted | 5.3M | 106.2M | 3.7 × 10¹⁵ | 3.979 |
| 256, measured | | | | 3.956 |

The fit was loss = 7,470 × C^(−0.263) + 3.377, and its prediction for the next size was
0.6% off. Stretched 30 times further, to `base`'s 3.35 × 10¹⁶ FLOPs, it says 3.714;
`base` measured 3.715 and 3.735. That is partly luck: `base` reads twice the context (which
lowers loss), uses dropout and repeats its data more often (which raise it), and the
effects partly cancel.

## What's easy to get wrong

- **Fitting in the wrong space.** Least squares on log(y − c) weights small losses heavily.
  The solution fits the line in log space but chooses c by the error in y itself, which is
  what you care about.
- **c too close to the smallest loss.** Then log(y − c) for the best model is very negative
  and dominates. The grid stops at 0.999 × min(y), and the error in y rejects such fits.
- **Extrapolating far.** A fit over a 30-fold range of compute is fair to stretch by a few
  times, not by a thousand. Check the prediction on the next size, as the sweep does,
  before you trust it for the final run.
- **Too few steps for the small models.** Warmup and noise dominate very short runs. The
  sweep script gives every model at least 50 steps and caps warmup at a tenth of the run.
