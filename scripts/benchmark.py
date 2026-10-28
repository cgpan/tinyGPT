"""Time a few training steps on this machine and estimate how long each full run takes.

    python scripts/benchmark.py                      # every config in configs/
    python scripts/benchmark.py configs/base.yaml    # just one
"""

import sys
import time
from pathlib import Path

import torch

from tinygpt import GPT, count_params, get_batch, get_device, load_config, load_data, train_step


def wait_for(device):
    """GPUs run asynchronously; wait until they have really finished before reading the clock."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def seconds_per_step(cfg, train_ids, device, warmup=3, steps=10):
    torch.manual_seed(cfg.seed)
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for i in range(warmup + steps):
        if i == warmup:
            wait_for(device)
            start = time.time()
        x, y = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
        train_step(model, x, y, optimizer)
    wait_for(device)
    return (time.time() - start) / steps, count_params(model)


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in Path("configs").glob("*.yaml"))
    device = get_device()
    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"torch {torch.__version__} on {name}\n")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")               # the same setting train() uses
    _, train_ids, _ = load_data()

    print(f"{'config':<22}{'params':>9}{'s/step':>9}{'est. minutes':>14}")
    for path in paths:
        cfg = load_config(path)
        per_step, n = seconds_per_step(cfg, train_ids, device)
        # each measurement runs 2 x eval_iters forward passes; a forward pass costs about 1/3 of a training step
        n_evals = cfg.max_iters // cfg.eval_interval + 1
        total_steps = cfg.max_iters + n_evals * 2 * cfg.eval_iters / 3
        minutes = per_step * total_steps / 60
        note = "" if minutes <= cfg.max_minutes else (
            f"  <- over max_minutes; would stop near step {int(cfg.max_minutes * 60 / per_step)}")
        print(f"{path:<22}{n / 1e6:>8.2f}M{per_step:>9.3f}{minutes:>14.1f}{note}")


if __name__ == "__main__":
    main()
