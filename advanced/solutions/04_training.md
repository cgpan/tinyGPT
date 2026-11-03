# Project 4 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/optim.py`](scratchgpt/optim.py),
[`scratchgpt/train.py`](scratchgpt/train.py) and
[`scratchgpt/sampling.py`](scratchgpt/sampling.py).

## Predict 2: resuming exactly

Three things decide what a run does next, and all three must be in the checkpoint:

1. the weights and the optimizer's moments (and the gradient scaler's state);
2. the **batch generator's** state, which decides which windows come next;
3. **PyTorch's random state**, which decides dropout's masks (on a GPU, its own random state
   too).

Leave out 2 or 3 and the resumed run is still a fine run, but a different one: its losses
differ from the uninterrupted run's after the first step. `test_train_resumes_exactly_where_it_stopped`
catches each of these, and removing either line from `load_checkpoint` makes it fail.
That is how you know a resume is right: not that the loss looks reasonable, but that it is
identical.

## Predict 1 and 4, measured

| | validation loss | bits per byte | time |
|---|---|---|---|
| untrained | about 9.1 | about 3.3 | |
| `small`, 3,000 steps | 4.23 | 1.53 | 2.5 min (RTX 4070 SUPER), 4 min (M5 Max) |
| `base`, 5,000 steps | 3.72 (PC), 3.74 (Mac) | 1.35 | 77 min (PC), 115 min (Mac) |

(Bits per byte here is the validation loss divided by ln 2 × 3.98 bytes per token.)

`small` already writes something that reads like a review: sentences about acting, plots and
endings, with invented words ("Propretty Python") and sentences that go nowhere. `base`
writes grammatical reviews whose sentences each make sense, but which drift and contradict
each other. It doesn't continue training reviews word for word: project 7 measures that.

## Predict 3: gradient accumulation

The same, up to rounding, when each batch's loss is divided by the number of batches: the
mean over 64 sequences equals the mean of four means over 16. Without the division the
gradients are four times too big, which AdamW mostly hides (it divides by the gradient's
size) but clipping and the loss you report don't.

## What's easy to get wrong

- **AdamW's weight decay.** Decoupled: `p -= lr * wd * p`, separate from the Adam update.
  Adding `wd * p` to the gradient instead is plain Adam with L2 regularisation, which
  behaves differently.
- **A shared state in `state_dict`.** If `state_dict` returns the optimizer's own tensors
  instead of copies, a second optimizer loaded from it updates the same tensors. The given
  code clones; the test that saves and restores the state caught this while the solution
  was being written.
- **Clipping before unscaling.** With float16 the gradients are multiplied by the loss
  scale (often 65,536) until `unscale_`. Clip them before and the norm is meaningless.
- **Evaluating on new batches every time.** `estimate_loss` uses a fixed seed, so two
  evaluations differ only because the model changed.
- **A checkpoint killed half-written.** `torch.save` straight onto `last.pt` can leave a
  broken file if the run dies while saving. Write `last.pt.tmp`, then `os.replace` it:
  the rename is atomic.
