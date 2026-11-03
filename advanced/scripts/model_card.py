"""Project 8: write the model card for a trained model.

    python scripts/model_card.py runs/base

Fills in everything measured (the data, the model, training, evaluation, the watermark)
from the run folder, and leaves the parts only you can write marked TODO. Writes
MODEL_CARD.md into the run folder; it will not overwrite one you have already edited.
"""

import argparse
import json
from pathlib import Path

import torch

from scratchgpt.config import Config
from scratchgpt.model import parameter_formula
from scratchgpt.watermark import DELTA, GAMMA


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("--force", action="store_true", help="overwrite an existing MODEL_CARD.md")
    args = parser.parse_args()
    run = Path(args.run)
    out = run / "MODEL_CARD.md"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; edit it, or use --force to start again")

    cfg = Config(**torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"])
    data = Path(cfg.data_dir)
    stats, meta = load(data / "stats.json"), load(data / "meta.json")
    evaluation, mark = load(run / "eval.json"), load(run / "watermark.json")
    log = [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    best = min(log, key=lambda row: row["val_loss"])

    def number(d, key, fmt="{:,}"):
        return fmt.format(d[key]) if key in d else "not measured"

    card = f"""# Model card: {run.name}

## What it is
A {parameter_formula(cfg) / 1e6:.1f}M-parameter decoder-only transformer ({cfg.n_layer} layers,
width {cfg.d_model}, {cfg.n_head} heads, context {cfg.context} tokens), written from scratch
and trained on movie reviews for learning purposes.

## Intended use
Studying how language models are built, trained and evaluated.

TODO: say in your own words what you built it to find out.

## Out of scope: never use it for
- writing reviews, comments or ratings to post anywhere as if a person wrote them;
- posing as a real person, or as a group of people;
- writing misinformation, or anything meant to deceive.

Text it writes carries a watermark (gamma {GAMMA}, delta {DELTA}; the key is kept private).

## Training data
The IMDB Large Movie Review Dataset (Maas et al., 2011): {number(stats, 'reviews')} reviews
read; {number(stats, 'train')} training documents after cleaning,
{number(stats, 'exact_duplicates')} exact and {number(stats, 'near_duplicates')} near duplicates
removed, and {number(stats, 'contaminated')} training reviews removed because they repeated a
test review. Email addresses ({number(stats, 'email')}), web addresses ({number(stats, 'url')})
and phone numbers ({number(stats, 'phone')}) were replaced by placeholders.
The reviews are English, about films, written by IMDB users before 2011.

Tokenizer: byte-level BPE, {number(meta, 'vocab_size')} tokens, about
{number(meta, 'bytes_per_token', '{:.2f}')} bytes of text per token.

## Training
{log[-1]['step']:,} steps at {cfg.tokens_per_step:,} tokens each; best validation loss
{best['val_loss']:.4f} (nats per token) at step {best['step']:,}; {log[-1]['seconds'] / 3600:.2f} hours
for the last run.

TODO: the hardware, and anything that went wrong along the way.

## Evaluation
| measure | value |
|---|---|
| bits per byte, test reviews | {number(evaluation, 'bits_per_byte', '{:.3f}')} |
| bits per byte, token frequencies only | {number(evaluation, 'unigram_bits_per_byte', '{:.3f}')} |
| bits per byte, gzip | {number(evaluation, 'gzip_bits_per_byte', '{:.3f}')} |
| sentiment accuracy, a probe on its features | {number(evaluation, 'sentiment_accuracy_model', '{:.1%}')} |
| sentiment accuracy, a probe on word counts | {number(evaluation, 'sentiment_accuracy_bag_of_words', '{:.1%}')} |
| training reviews continued word for word | {number(evaluation, 'extraction_rate', '{:.1%}')} |
| watermarked texts flagged | {number(mark.get('detected', {}), 'watermarked', '{:.0%}')} |
| real reviews flagged | {number(mark.get('detected', {}), 'real_reviews', '{:.0%}')} |

## Limitations
TODO: what it gets wrong. Read twenty of its texts first. Does it make up films, actors or
facts? Does it repeat itself? How does it treat the people and groups it writes about?

## Risks
TODO: who could be harmed if its text were posted as real reviews, and how the watermark
helps (and where it does not: see the editing results).
"""
    out.write_text(card, encoding="utf-8")
    print(f"wrote {out}: now fill in the TODO parts")


if __name__ == "__main__":
    main()
