# Project 4 · Training

The infrastructure around the model: the optimizer, the learning-rate schedule, batches,
mixed precision, and checkpoints that let a run survive being cut off. At an AI company a
training run is shared, expensive and long, and it *will* be interrupted: a lost GPU, a
restarted machine, a Colab session that ends. A run that can't resume exactly wastes all
that compute. At the end of this project you'll train your first IMDB model and read what
it writes.

**You write**, in `src/scratchgpt/optim.py`: `parameter_groups`, `AdamW.step`,
`learning_rate`, `clip_gradients`; in `train.py`: `get_batch`, `estimate_loss`,
`train_step`, `save_checkpoint`, `load_checkpoint`, `train`; in `sampling.py`:
`sample_next`, and `generate` without the cache.
**Given:** the rest of `AdamW` (its state can be saved and loaded), `load_tokens`,
`load_model`, the watermark (project 8) and `write`.
**Done when** `python -m pytest tests/test_04_training.py` passes and you have trained
`small` and read its writing.

## Predict

1. Your model starts at about ln 8,192 = 9.01 nats per token. Where will `small` be after
   3,000 steps? Write a number, and the same number in bits per byte (project 2,
   Investigate 4).
2. A run is cut off at step 1,100, and the last checkpoint was at 1,000. You resume. Will
   the loss at step 2,000 be exactly the same as in a run that was never cut off? What would
   you have to save for that?
3. Gradient accumulation: 4 batches of 16, one optimizer step. Is that the same as one
   batch of 64? Exactly, or only nearly?
4. What will the model write after 3,000 steps: words, sentences, or something that reads
   like a review?

## Build

- **`AdamW.step`.** The test runs yours and `torch.optim.AdamW` side by side for five
  steps; the parameters must agree. The usual mistakes are forgetting the bias correction
  and applying weight decay to Adam's update instead of to the weights.
- **`learning_rate`**: linear warmup, then half a cosine. **`clip_gradients`**: the
  norm over all the gradients together, not per tensor.
- **`get_batch`.** Draw the windows with the generator you are given, and nothing else:
  saving that generator is what lets a resumed run see the same batches.
- **`train_step`.** Divide each batch's loss by the number of batches before `backward`,
  so the gradients add up to one big batch's. Unscale before you clip.
- **`save_checkpoint` and `load_checkpoint`.** Everything that changes during training
  must be saved, including two kinds of random state. Write to a temporary file and
  rename it, so a run killed while saving still has its last good checkpoint.
- **`train`.** The docstring is the whole plan. `test_train_resumes_exactly_where_it_stopped`
  trains with dropout, stops after 11 steps, resumes, and demands the very same numbers as
  a run that didn't stop.
- **`sample_next` and `generate`.** Temperature, top-k and top-p, then the loop. Leave
  `use_cache` for project 5.

**Mixed precision.** `device.amp_dtype` picks the number format: bfloat16 on NVIDIA GPUs
that have it (RTX 30 and 40 series, A100, L4), float16 on older ones such as Colab's T4,
where the gradient scaler keeps small gradients from rounding to zero, and float32 on
Macs, which is the safe default there. You'll see why some parts of the model compute in
float32 regardless: the norms, the softmax and the loss.

## Run

```bash
python scripts/train.py configs/small.yaml
python scripts/sample.py runs/small --n 3
python scripts/sample.py runs/small --prompt "I watched this movie last night and" --n 3
```

**On Colab:** your runs must live on Google Drive or they vanish with the session. Use
`out_dir=/content/drive/MyDrive/tinyGPT-runs/small` (the Colab notebook sets this up).

**The drill.** Start `configs/base.yaml`, stop it with Ctrl+C (or *Runtime > Interrupt*)
after the first checkpoint, and run the same command again. Check `log.jsonl`: does the
curve carry on as if nothing happened?

## Investigate

1. Train `small` with `lr=3e-4`, `lr=1e-3` (the default) and `lr=3e-3`
   (`python scripts/train.py configs/small.yaml lr=3e-3 out_dir=runs/small-lr3e-3`).
   Plot or tabulate validation loss against step from each `log.jsonl`. What does too high
   look like? Too low?
2. Look at `grad_norm` in the log. What does it do in the first 100 steps, and why does
   warmup help?
3. Read 10 samples from `small`. What has it learned? What hasn't it? Keep them: you'll
   compare them with your final model.
4. Remove `torch.set_rng_state` from `load_checkpoint` and run the resume test. Which number
   differs, and why only that one?

## Stretch

- Write your own loss scaling for float16 instead of `torch.amp.GradScaler`: scale the
  loss, check the gradients for inf and NaN, skip the step and halve the scale when you
  find one, double it after 2,000 good steps.
- Add z-loss (a small penalty on log(sum(exp(logits)))²), used to keep large models'
  logits from drifting. Does it change anything at this size?
