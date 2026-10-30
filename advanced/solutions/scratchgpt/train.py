"""Project 4: batches, the training loop, and checkpoints that survive being cut off."""

import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from scratchgpt.config import Config
from scratchgpt.device import amp_dtype, autocast, describe, get_device
from scratchgpt.model import GPT, count_parameters
from scratchgpt.optim import AdamW, clip_gradients, learning_rate, parameter_groups


def load_tokens(path):
    """A token file from project 2, as a read-only array that stays on disk. Given."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(tokens, batch_size, context, generator, device="cpu"):
    """batch_size random windows of the token array. Returns (x, y), each (batch_size, context),
    int64, on device: y is x moved one token on, so y[:, t] is the token after x[:, t].
    Draw the window starts with torch.randint(len(tokens) - context, (batch_size,),
    generator=generator), so that saving the generator's state saves where the data is."""
    # BEGIN SOLUTION 4
    starts = torch.randint(len(tokens) - context, (batch_size,), generator=generator).tolist()
    windows = torch.from_numpy(np.stack([tokens[s : s + context + 1] for s in starts]).astype(np.int64))
    x, y = windows[:, :-1], windows[:, 1:]
    device = torch.device(device)
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
    # END SOLUTION


@torch.no_grad()
def estimate_loss(model, tokens, cfg, steps, device, dtype=None, seed=0):
    """The model's mean loss over `steps` batches of cfg.batch_size windows, in eval mode (no
    dropout), drawn with a generator seeded with `seed`, so every evaluation sees the same
    batches. Puts the model back in the mode it was in."""
    # BEGIN SOLUTION 4
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(steps):
        x, y = get_batch(tokens, cfg.batch_size, cfg.context, generator, device)
        with autocast(device, dtype):
            _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / steps
    # END SOLUTION


def train_step(model, optimizer, batches, scaler, device, dtype=None, grad_clip=1.0):
    """One optimizer step over several batches (gradient accumulation).

    For each (x, y): compute the loss in autocast(device, dtype), divide it by len(batches)
    and backpropagate it through scaler.scale(...). The gradients add up, so the step is the
    same as one big batch. Then scaler.unscale_(optimizer), clip_gradients to grad_clip,
    scaler.step(optimizer), scaler.update() and optimizer.zero_grad().

    The scaler matters only in float16, whose small range can round gradients to zero; for
    other dtypes, pass torch.amp.GradScaler(enabled=False) and it does nothing.
    Returns (mean loss over the batches, gradient norm before clipping), as floats.
    """
    # BEGIN SOLUTION 4
    total = 0.0
    for x, y in batches:
        with autocast(device, dtype):
            _, loss = model(x, y)
        loss = loss / len(batches)
        scaler.scale(loss).backward()
        total += loss.detach()
    scaler.unscale_(optimizer)
    norm = clip_gradients([p for g in optimizer.param_groups for p in g["params"]], grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return float(total), norm
    # END SOLUTION


def save_checkpoint(path, model, optimizer, scaler, generator, step, best_val, cfg):
    """Save everything needed to carry on exactly where training stopped: the model's and the
    optimizer's state, the scaler's state, the batch generator's state, PyTorch's random
    state (dropout draws from it; with a GPU, also torch.cuda.get_rng_state_all() or
    torch.mps.get_rng_state()), the step, the best validation loss so far, and the config
    as a dict.

    Write to a temporary file next to path first, then os.replace it onto path: if the run
    is cut off while saving, the old checkpoint is still whole.
    """
    # BEGIN SOLUTION 4
    path = Path(path)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "generator": generator.get_state(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "step": step,
        "best_val": best_val,
        "config": dataclasses.asdict(cfg),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    # END SOLUTION


def load_checkpoint(path, model, optimizer, scaler, generator):
    """Restore what save_checkpoint saved into the given objects. Returns (step, best_val)."""
    # BEGIN SOLUTION 4
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state.get("mps_rng") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"])
    return state["step"], state["best_val"]
    # END SOLUTION


def train(cfg, stop_after=None, log=print):
    """Train a GPT on cfg.data_dir's train.bin, checking it on val.bin. Returns the history:
    one dict per evaluation, {"step", "train_loss", "val_loss", "lr", "grad_norm", "tokens",
    "seconds", "tokens_per_second"}, also appended as JSON lines to out_dir/log.jsonl.

    Setup: torch.manual_seed(cfg.seed); take the vocabulary size from data_dir/meta.json;
    build the model on get_device(cfg.device) and the AdamW optimizer on parameter_groups;
    a GradScaler, enabled only for float16; a batch generator seeded with cfg.seed.
    If out_dir/last.pt exists, resume from it.

    Each step: set every group's lr to learning_rate(step, ...); draw cfg.grad_accum batches
    and call train_step. Every cfg.eval_interval steps, and at the last one, estimate the
    validation loss over cfg.eval_steps batches, log a line, and when it is the best so far
    save the model alone to out_dir/best.pt as {"model": ..., "config": ...}. Every
    cfg.checkpoint_interval steps, at the end, and before stopping early, save_checkpoint to
    out_dir/last.pt.

    Stop early, after saving, once cfg.max_hours have passed in this run or, for tests, once
    this run has taken stop_after steps.
    """
    # BEGIN SOLUTION 4
    data_dir, out_dir = Path(cfg.data_dir), Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if cfg.vocab_size != meta["vocab_size"]:
        cfg = dataclasses.replace(cfg, vocab_size=meta["vocab_size"])
    train_tokens, val_tokens = load_tokens(data_dir / "train.bin"), load_tokens(data_dir / "val.bin")

    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = amp_dtype(device, cfg.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = GPT(cfg).to(device)
    optimizer = AdamW(parameter_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    generator = torch.Generator().manual_seed(cfg.seed)

    step, best_val = 0, math.inf
    last = out_dir / "last.pt"
    if last.exists():
        step, best_val = load_checkpoint(last, model, optimizer, scaler, generator)
        log(f"resuming from step {step:,}")
    else:
        log(f"{count_parameters(model):,} parameters, {cfg.tokens_per_step:,} tokens per step, "
            f"training on {describe(device)}" + (f" in {str(dtype).split('.')[-1]}" if dtype else ""))
    runner = torch.compile(model) if cfg.compile else model
    model.train()

    history, started, steps_this_run = [], time.perf_counter(), 0
    window_start, window_tokens = started, 0
    while step < cfg.max_steps:
        lr = learning_rate(step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batches = [get_batch(train_tokens, cfg.batch_size, cfg.context, generator, device)
                   for _ in range(cfg.grad_accum)]
        loss, norm = train_step(runner, optimizer, batches, scaler, device, dtype, cfg.grad_clip)
        step += 1
        steps_this_run += 1
        window_tokens += cfg.tokens_per_step

        out_of_time = (time.perf_counter() - started) / 3600 > cfg.max_hours
        stopping = out_of_time or (stop_after is not None and steps_this_run >= stop_after)
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val = estimate_loss(runner, val_tokens, cfg, cfg.eval_steps, device, dtype)
            now = time.perf_counter()
            row = {"step": step, "train_loss": round(loss, 4), "val_loss": round(val, 4), "lr": lr,
                   "grad_norm": round(norm, 3), "tokens": step * cfg.tokens_per_step,
                   "seconds": round(now - started, 1),
                   "tokens_per_second": round(window_tokens / (now - window_start))}
            window_start, window_tokens = now, 0
            history.append(row)
            with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(f"step {step:>6,}  train {loss:.4f}  val {val:.4f}  lr {lr:.2e}  "
                f"{row['tokens_per_second']:,} tokens/s")
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, out_dir / "best.pt")
        if step % cfg.checkpoint_interval == 0 or step == cfg.max_steps or stopping:
            save_checkpoint(last, model, optimizer, scaler, generator, step, best_val, cfg)
        if stopping:
            log(f"stopping at step {step:,}" + (f": {cfg.max_hours} hours have passed" if out_of_time else ""))
            break
    return history
    # END SOLUTION


def load_model(path, device=None):
    """Load a model saved by train (best.pt or last.pt), in eval mode. Returns (model, cfg). Given."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**state["config"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    return model.to(device or get_device()).eval(), cfg
