"""Write text with a trained model.

    python scripts/sample.py runs/base/model.pt
    python scripts/sample.py runs/base/model.pt --prompt "ROMEO:" --length 300 --temperature 0.8
"""

import argparse

from tinygpt import load_checkpoint, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a model.pt file saved by scripts/train.py")
    parser.add_argument("--prompt", default="\n", help="text to continue (default: a new line)")
    parser.add_argument("--length", type=int, default=500, help="characters to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="below 1: safer; above 1: wilder")
    parser.add_argument("--top-k", type=int, default=None, help="only ever pick from the k likeliest characters")
    parser.add_argument("--seed", type=int, default=None, help="fix the dice, for repeatable output")
    args = parser.parse_args()

    model, tok, _ = load_checkpoint(args.checkpoint)
    print(sample(model, tok, args.prompt, args.length, args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
