"""Project 8: does the watermark work on your model, and what does it cost?

    python scripts/watermark_check.py runs/base

Writes 20 short texts with the watermark and 20 without, scores them and 20 real test
reviews with your detector, and tries to wash the watermark out by replacing a share of
the tokens at random. Nothing is saved but the scores (watermark.json in the run folder).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from scratchgpt.device import get_device
from scratchgpt.sampling import generate
from scratchgpt.tokenizer import Tokenizer
from scratchgpt.train import load_model
from scratchgpt.watermark import detect


def main():
    sys.stdout.reconfigure(line_buffering=True, errors="replace")   # print as it happens; never crash on a character
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("--tokens", type=int, default=150)
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_model(Path(args.run) / "best.pt", device)
    tokenizer = Tokenizer.load(Path(cfg.data_dir) / "tokenizer.json")
    vocab = tokenizer.vocab_size
    start = torch.tensor([[tokenizer.eot_id]], device=device)

    def texts(watermark):
        out = []
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            ids = generate(model, start, args.tokens, temperature=0.8, top_p=0.95, watermark=watermark,
                           use_cache=True, generator=g)[0, 1:].tolist()
            out.append(ids)
        return out

    marked, plain = texts(True), texts(False)
    with open(Path(cfg.data_dir) / "test.jsonl", encoding="utf-8") as f:
        human = [tokenizer.encode(json.loads(line)["text"], allow_special=False)[: args.tokens]
                 for _, line in zip(range(20), f)]

    def scores(group):
        return [detect(ids, vocab) for ids in group]

    def caught(group, threshold=4.0):
        return sum(z > threshold for z in scores(group)) / len(group)

    rng = random.Random(0)

    def edited(ids, share):
        return [rng.randrange(vocab) if rng.random() < share else i for i in ids]

    results = {
        "mean_z": {"watermarked": sum(scores(marked)) / 20, "not_watermarked": sum(scores(plain)) / 20,
                   "real_reviews": sum(scores(human)) / 20},
        "detected": {"watermarked": caught(marked), "not_watermarked": caught(plain), "real_reviews": caught(human)},
        "detected_after_editing": {f"{int(s * 100)}%": caught([edited(ids, s) for ids in marked]) for s in (0.1, 0.25, 0.5)},
    }
    (Path(args.run) / "watermark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{'20 texts each':<28}{'mean z':>8}{'flagged (z > 4)':>17}")
    for name in ["watermarked", "not_watermarked", "real_reviews"]:
        print(f"{name.replace('_', ' '):<28}{results['mean_z'][name]:>8.2f}{results['detected'][name]:>17.0%}")
    print("\nwatermarked, after replacing tokens at random:")
    for share, rate in results["detected_after_editing"].items():
        print(f"  {share:>4} replaced: {rate:.0%} still flagged")


if __name__ == "__main__":
    main()
