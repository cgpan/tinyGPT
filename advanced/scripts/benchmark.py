"""Project 5: how fast does training run here, and how long would a whole run take?

    python scripts/benchmark.py                                  # every config
    python scripts/benchmark.py configs/base.yaml
    python scripts/benchmark.py configs/base.yaml fused_attention=true compile=true

Times 10 real training steps (after 3 to warm up) and prints tokens per second, the
fraction of the GPU's peak you use (MFU), the memory used, and the time a full run takes.
Random token ids stand in if the data isn't prepared yet: speed doesn't depend on them.
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scratchgpt.config import load_config, parse_overrides
from scratchgpt.device import amp_dtype, describe, get_device, synchronize
from scratchgpt.model import GPT, count_parameters
from scratchgpt.optim import AdamW, parameter_groups
from scratchgpt.perf import mfu, peak_tflops, training_memory
from scratchgpt.train import get_batch, load_tokens, train_step


def benchmark(cfg, steps=10, warmup=3):
    device = get_device(cfg.device)
    dtype = amp_dtype(device, cfg.dtype)
    data = Path(cfg.data_dir)
    if (data / "meta.json").exists():
        cfg = dataclasses.replace(cfg, vocab_size=json.loads((data / "meta.json").read_text(encoding="utf-8"))["vocab_size"])
        tokens = load_tokens(data / "train.bin")
    else:
        tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, 2_000_000).astype(np.uint16)
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    runner = torch.compile(model) if cfg.compile else model
    optimizer = AdamW(parameter_groups(model, cfg.weight_decay), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    generator = torch.Generator().manual_seed(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            synchronize(device)
            start = time.perf_counter()
        batches = [get_batch(tokens, cfg.batch_size, cfg.context, generator, device) for _ in range(cfg.grad_accum)]
        train_step(runner, optimizer, batches, scaler, device, dtype, cfg.grad_clip)
    synchronize(device)
    seconds = (time.perf_counter() - start) / steps
    result = {"device": describe(device), "dtype": str(dtype or torch.float32).split(".")[-1],
              "parameters": count_parameters(model), "seconds_per_step": seconds,
              "tokens_per_second": cfg.tokens_per_step / seconds,
              "minutes_for_run": seconds * cfg.max_steps / 60,
              "estimated_gb": training_memory(cfg, cfg.batch_size)["total"] / 1e9}
    peak = peak_tflops(result["device"])
    result["mfu"] = mfu(result["tokens_per_second"], cfg, peak) if peak else None
    result["peak_gb"] = None
    if device.type == "cuda":
        result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    elif device.type == "mps":                      # what the Apple GPU holds now, not its peak
        result["peak_gb"] = torch.mps.driver_allocated_memory() / 1e9
    return result


def main():
    sys.stdout.reconfigure(line_buffering=True, errors="replace")   # print as it happens; never crash on a character
    paths = [a for a in sys.argv[1:] if "=" not in a] or sorted(str(p) for p in Path("configs").glob("*.yaml"))
    overrides = parse_overrides([a for a in sys.argv[1:] if "=" in a])
    rows = []
    for path in paths:
        cfg = load_config(path, **overrides)
        print(f"timing {path} ...", flush=True)
        rows.append((Path(path).stem, cfg, benchmark(cfg)))
    device = rows[0][2]["device"]
    print(f"\non {device}, in {rows[0][2]['dtype']}:\n")
    print(f"{'config':<8}{'params':>8}{'tokens/s':>11}{'MFU':>7}{'memory GB':>11}{'estimate':>10}{'full run':>12}")
    for name, cfg, r in rows:
        used = f"{r['peak_gb']:.1f}" if r["peak_gb"] is not None else "-"
        mfu_text = f"{r['mfu']:.0%}" if r["mfu"] is not None else "-"
        print(f"{name:<8}{r['parameters'] / 1e6:>7.1f}M{r['tokens_per_second']:>11,.0f}{mfu_text:>7}{used:>11}"
              f"{r['estimated_gb']:>10.1f}{r['minutes_for_run']:>9.1f} min")
    print("\nfull run: max_steps at this speed. memory GB: the peak measured on NVIDIA GPUs, what is held on Apple"
          "\nGPUs (none on CPUs); estimate: project 5's. MFU needs the GPU's peak speed, in perf.PEAK_TFLOPS.")


if __name__ == "__main__":
    main()
