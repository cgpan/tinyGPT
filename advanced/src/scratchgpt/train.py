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
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


@torch.no_grad()
def estimate_loss(model, tokens, cfg, steps, device, dtype=None, seed=0):
    """The model's mean loss over `steps` batches of cfg.batch_size windows, in eval mode (no
    dropout), drawn with a generator seeded with `seed`, so every evaluation sees the same
    batches. Puts the model back in the mode it was in."""
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


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
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


def save_checkpoint(path, model, optimizer, scaler, generator, step, best_val, cfg):
    """Save everything needed to carry on exactly where training stopped: the model's and the
    optimizer's state, the scaler's state, the batch generator's state, PyTorch's random
    state (dropout draws from it; with a GPU, also torch.cuda.get_rng_state_all() or
    torch.mps.get_rng_state()), the step, the best validation loss so far, and the config
    as a dict.

    Write to a temporary file next to path first, then os.replace it onto path: if the run
    is cut off while saving, the old checkpoint is still whole.
    """
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


def load_checkpoint(path, model, optimizer, scaler, generator):
    """Restore what save_checkpoint saved into the given objects. Returns (step, best_val)."""
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


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
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


def load_model(path, device=None):
    """Load a model saved by train (best.pt or last.pt), in eval mode. Returns (model, cfg). Given."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**state["config"])
    model = GPT(cfg)
    model.load_state_dict(state["model"])
    return model.to(device or get_device()).eval(), cfg
