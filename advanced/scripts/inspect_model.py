"""Project 3: look at your model before training it.

    python scripts/inspect_model.py                     # every config
    python scripts/inspect_model.py configs/base.yaml

Prints where the parameters are, and the untrained model's loss on random targets and with
the inputs as targets (Predict 1 to 3 in the brief).
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

from scratchgpt.config import load_config
from scratchgpt.model import GPT, count_parameters, parameter_formula


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in Path("configs").glob("*.yaml"))
    for path in paths:
        cfg = load_config(path)
        torch.manual_seed(0)
        model = GPT(cfg).eval()
        parts = defaultdict(int)
        for name, p in model.named_parameters():
            part = "norms" if "norm" in name else name.split(".")[2] if name.startswith("blocks") else name.split(".")[0]
            parts[{"attn": "attention", "ff": "feed-forward", "embed": "embedding", "head": "output layer"}.get(part, part)] += p.numel()
        total = count_parameters(model)
        print(f"\n{path}: {total:,} parameters (parameter_formula says {parameter_formula(cfg):,})")
        for part, n in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"  {part:<14}{n:>12,}  {n / total:6.1%}")
        with torch.no_grad():
            ids, other = torch.randint(cfg.vocab_size, (2, 2, cfg.context))
            random_targets = model(ids, other)[1].item()
            same_targets = model(ids, ids)[1].item()
        print(f"  untrained loss: {random_targets:.3f} on random targets, {same_targets:.3f} with the inputs "
              f"as targets; ln {cfg.vocab_size:,} = {math.log(cfg.vocab_size):.3f}")


if __name__ == "__main__":
    main()
