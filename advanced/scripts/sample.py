"""Project 4: read what your model writes.

    python scripts/sample.py runs/base
    python scripts/sample.py runs/base --prompt "I watched this film" --temperature 0.7 --n 3

Everything it writes carries the watermark (project 8). It is for reading, not for posting:
see Responsible use in the README. It writes at most 5 texts at a time, on purpose.
"""

import argparse
import sys
from pathlib import Path

from scratchgpt.device import get_device
from scratchgpt.sampling import write
from scratchgpt.tokenizer import Tokenizer
from scratchgpt.train import load_model


def main():
    sys.stdout.reconfigure(line_buffering=True, errors="replace")   # print as it happens; never crash on a character
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="a run folder, such as runs/base")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n", type=int, default=1, choices=range(1, 6), metavar="1-5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-watermark", action="store_true", help="for experiments in project 8 only")
    args = parser.parse_args()

    model, cfg = load_model(Path(args.run) / "best.pt", get_device())
    tokenizer = Tokenizer.load(Path(cfg.data_dir) / "tokenizer.json")
    print("Generated text, for study only. Never post it as a real review.\n")
    for i in range(args.n):
        seed = None if args.seed is None else args.seed + i
        kwargs = dict(max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, watermark=not args.no_watermark, seed=seed)
        try:
            text = write(model, tokenizer, args.prompt, **kwargs)
        except NotImplementedError:                           # no KV cache yet (project 5)
            text = write(model, tokenizer, args.prompt, use_cache=False, **kwargs)
        print(f"--- {i + 1} ---\n{args.prompt}{text}\n")


if __name__ == "__main__":
    main()
