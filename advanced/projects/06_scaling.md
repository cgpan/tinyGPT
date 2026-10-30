# Project 6 · Scaling, and the final run

How big should the model be, and how long should it train? Labs answer that with scaling
laws: train several small models, fit how loss falls with compute, and predict the big
one before paying for it. You'll do the same, then plan and train your final model.

**You write**, in `src/scratchgpt/scaling.py`: `fit_power_law` and `plan_run`.
**Given:** `power_law`, `sweep_configs` and `scripts/sweep.py`.
**Done when** `python -m pytest tests/test_06_scaling.py` passes, you have run the sweep
and checked your prediction, and your final model is trained.

## Predict

1. Loss against compute, on log-log axes: a straight line, a curve that flattens, or
   something else? Why can't loss keep falling forever?
2. The sweep trains widths 64 to 192, then predicts width 256 before training it. How close
   will the prediction be?
3. IMDB's training text is about 100 MB. At your tokenizer's bytes per token, how many
   tokens is that? A compute-optimal model sees about 20 tokens per parameter (Hoffmann et
   al., 2022). Using all the data four times over, how big a model is that?

## Build

- **`fit_power_law`.** y = a·x^(−b) + c is a straight line in log-log space only once
  you subtract c, the loss no amount of compute removes. Try many values of c, fit a line
  for each, keep the best.
- **`plan_run`.** Two regimes. With plenty of data, grow parameters and tokens together.
  With too little, data repeated more than about four times is worth much less
  (Muennighoff et al., 2023), so once the data runs out, extra compute goes into
  parameters.

## Run

```bash
python scripts/sweep.py configs/small.yaml
```

It trains widths 64, 96, 128 and 192 at 20 tokens per parameter, fits the law, prints its
prediction for width 256, then trains 256 to check. Results go to
`runs/sweep/results.json`; a rerun skips what's done.

## The final run

1. **Choose a budget**, in hours, for your hardware: a night on your own GPU, or up to 24
   hours of paid Colab.
2. **Measure speed.** `python scripts/benchmark.py` on a config near the size you expect.
3. **Plan.** Compute = hours × 3,600 × tokens per second × FLOPs per token. Call
   `plan_run(compute, unique_tokens)`, with unique tokens from `data/imdb/meta.json`.
4. **Write `configs/final.yaml`.** Pick a width and depth near the planned parameter count
   (`parameter_formula` checks it), and set `max_steps` from the planned tokens. Keep
   `dropout: 0.1`: you'll repeat the data.
5. **Train it,** with `out_dir` on Google Drive if you use Colab, and resume as often as
   you have to. `best.pt` keeps the step with the lowest validation loss, so training too
   long costs time, not quality.

## Investigate

1. How far off was the sweep's prediction for width 256? Is your fit's c believable?
2. Plot your final run's validation loss together with the sweep's points. Is it on the
   line? If not, why not? (How many epochs did it make? Look at when `best.pt` was saved.)
3. **The data wall.** Train `base` for twice the steps. Does validation loss keep falling,
   flatten, or rise? This is why labs pay so much for data.

## Stretch

- **More data.** The pipeline reads any JSON-lines file of `{"text": ...}` documents.
  Pre-train on a few hundred MB of a general English corpus (for example a sample of
  FineWeb-Edu, which is published for this purpose), then continue training on IMDB.
  Compare bits per byte on the IMDB test reviews with your IMDB-only model. Mind the
  corpus's license, and run project 1's checks on it too.
- Fit loss against parameters at fixed tokens, and against tokens at fixed parameters.
  Do the two fits agree with each other?
