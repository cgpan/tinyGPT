# Project 6 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/scaling.py`](scratchgpt/scaling.py).

## Predict 1: why loss flattens

Text is not perfectly predictable: even a perfect model is surprised by what a reviewer
decides to say next. That floor is the c in a·C^(−b) + c. On log-log axes the curve is
straight only after subtracting c; plotted raw, it bends towards c. Fitting a straight line
to raw loss predicts that a big enough model reaches zero, which is impossible.

## Predict 3: IMDB's size (a rough estimate)

About 100 MB of training text at 3.5 to 4 bytes per token is 25 to 30 million tokens.
Four passes give 100 to 120 million, and at 20 tokens per parameter that is a model of
5 to 6 million parameters: about `small`. A bigger model trained on this data still does
better, but by less and less, and repeating the data more often gains little. This is the
data-constrained regime, and it is why `plan_run` moves extra compute into parameters once
the data runs out. Your own `meta.json` gives the real token count.

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
