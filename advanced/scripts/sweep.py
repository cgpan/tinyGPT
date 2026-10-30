"""Project 6: train models of several sizes, fit a scaling law, and predict the next size.

    python scripts/sweep.py configs/small.yaml
    python scripts/sweep.py configs/small.yaml --widths 64 96 128 192 --predict 256

Each model sees tokens_per_parameter tokens per parameter (default 20), so compute grows
with size. Results go to runs/sweep/results.json; rerunning skips sizes already trained.
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

from scratchgpt.config import load_config, parse_overrides
from scratchgpt.model import parameter_formula
from scratchgpt.perf import flops_per_token
from scratchgpt.scaling import fit_power_law, power_law, sweep_configs
from scratchgpt.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="key=value settings, as for train.py")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96, 128, 192])
    parser.add_argument("--predict", type=int, default=256, help="the width to predict, then check")
    parser.add_argument("--tokens-per-parameter", type=float, default=20)
    parser.add_argument("--out", default="runs/sweep")
    args = parser.parse_args()

    base = dataclasses.replace(load_config(args.config, **parse_overrides(args.overrides)), out_dir=args.out)
    meta = json.loads((Path(base.data_dir) / "meta.json").read_text(encoding="utf-8"))
    base = dataclasses.replace(base, vocab_size=meta["vocab_size"])
    results_path = Path(args.out) / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    for cfg in sweep_configs(base, args.widths + [args.predict]):
        key = str(cfg.d_model)
        params = parameter_formula(cfg)
        steps = max(50, math.ceil(args.tokens_per_parameter * params / cfg.tokens_per_step))
        cfg = dataclasses.replace(cfg, max_steps=steps, warmup_steps=min(cfg.warmup_steps, steps // 10),
                                  eval_interval=steps, checkpoint_interval=steps)
        if cfg.d_model == args.predict and len(results) >= 3 and key not in results:
            fitted = [r for k, r in results.items() if int(k) != args.predict]
            a, b, c = fit_power_law([r["compute"] for r in fitted], [r["val_loss"] for r in fitted])
            compute = flops_per_token(cfg) * steps * cfg.tokens_per_step
            print(f"\nfit: loss = {a:.3g} * C^-{b:.3f} + {c:.3f}")
            print(f"prediction for width {args.predict} ({params / 1e6:.1f}M parameters): "
                  f"{power_law(compute, a, b, c):.4f}. Now training it to check ...\n")
        if key in results:
            continue
        print(f"width {cfg.d_model}: {params / 1e6:.1f}M parameters, {steps:,} steps")
        history = train(cfg, log=lambda *_: None)
        results[key] = {"parameters": params, "steps": steps, "tokens": steps * cfg.tokens_per_step,
                        "compute": flops_per_token(cfg) * steps * cfg.tokens_per_step,
                        "val_loss": history[-1]["val_loss"], "minutes": history[-1]["seconds"] / 60}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'width':>6}{'params':>9}{'tokens':>10}{'compute':>11}{'val loss':>10}{'minutes':>9}")
    for key, r in sorted(results.items(), key=lambda kv: int(kv[0])):
        print(f"{key:>6}{r['parameters'] / 1e6:>8.1f}M{r['tokens'] / 1e6:>9.1f}M{r['compute']:>11.2e}"
              f"{r['val_loss']:>10.4f}{r['minutes']:>9.1f}")


if __name__ == "__main__":
    main()
