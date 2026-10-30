"""Project 4: train a model.

    python scripts/train.py configs/small.yaml
    python scripts/train.py configs/base.yaml lr=3e-4 out_dir=runs/base-lr3e-4

Change any setting with key=value. If the run is cut off (Colab disconnects, you press
Ctrl+C), run the same command again: it carries on from the last checkpoint.
"""

import sys

from scratchgpt.config import load_config, parse_overrides
from scratchgpt.train import train


def main():
    sys.stdout.reconfigure(line_buffering=True, errors="replace")   # print as it happens; never crash on a character
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cfg = load_config(sys.argv[1], **parse_overrides(sys.argv[2:]))
    history = train(cfg)
    if history:
        best = min(history, key=lambda row: row["val_loss"])
        print(f"\nbest validation loss {best['val_loss']:.4f} at step {best['step']:,}; "
              f"saved in {cfg.out_dir}/best.pt")


if __name__ == "__main__":
    main()
