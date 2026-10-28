"""Train a model from a config file and save it.

    python scripts/train.py configs/base.yaml
    python scripts/train.py configs/small.yaml n_layer=2 dropout=0.1     # override any value
    python scripts/train.py configs/small.yaml --out runs/no-mask/model.pt causal=false
"""

import argparse
from dataclasses import fields
from pathlib import Path

from tinygpt import Config, load_config, train


def parse_override(pair):
    """'n_layer=2' -> ('n_layer', 2), converted to the type the Config field expects."""
    key, _, text = pair.partition("=")
    types = {f.name: type(f.default) for f in fields(Config)}
    if key not in types:
        raise SystemExit(f"unknown setting {key!r}; choose from: {', '.join(types)}")
    if types[key] is bool:
        return key, text.lower() in ("true", "1", "yes")
    return key, types[key](text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML file in configs/")
    parser.add_argument("overrides", nargs="*", help="settings to change, like n_layer=2")
    parser.add_argument("--out", help="where to save the model (default: runs/<config name>/model.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config, **dict(parse_override(p) for p in args.overrides))
    out = args.out or Path("runs") / Path(args.config).stem / "model.pt"
    train(cfg, save_path=out)


if __name__ == "__main__":
    main()
