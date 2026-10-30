"""Project 7: how good is the model? Writes eval.json into the run folder.

    python scripts/evaluate.py runs/base
    python scripts/evaluate.py runs/base --max-tokens 0      # the whole test set (slower)

  bits per byte    on the test reviews, against two baselines: gzip, and a model that
                   knows only how common each token is
  sentiment        a logistic regression on the model's features, against one on word counts
  memorization     how often the model continues a training review word for word
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from scratchgpt import evaluate
from scratchgpt.device import amp_dtype, get_device
from scratchgpt.tokenizer import Tokenizer
from scratchgpt.train import load_model, load_tokens


def read_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def labelled(docs, n, seed):
    """n documents with a label, half positive and half negative."""
    rng = random.Random(seed)
    pos = [d for d in docs if d["label"] == "pos"]
    neg = [d for d in docs if d["label"] == "neg"]
    picked = rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n // 2, len(neg)))
    rng.shuffle(picked)
    return [d["text"] for d in picked], torch.tensor([d["label"] == "pos" for d in picked], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("--max-tokens", type=int, default=2_000_000, help="test tokens for bits per byte (0: all)")
    parser.add_argument("--probe-train", type=int, default=4000)
    parser.add_argument("--probe-test", type=int, default=2000)
    parser.add_argument("--memorization", type=int, default=500, help="training reviews to try")
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_model(Path(args.run) / "best.pt", device)
    dtype = amp_dtype(device, cfg.dtype)
    data = Path(cfg.data_dir)
    tokenizer = Tokenizer.load(data / "tokenizer.json")
    lengths = evaluate.token_byte_lengths(tokenizer)
    results = {}

    print("bits per byte on the test reviews ...", flush=True)
    test_tokens = load_tokens(data / "test.bin")
    results["bits_per_byte"] = evaluate.bits_per_byte(model, test_tokens, lengths, device, dtype,
                                                      max_tokens=args.max_tokens or None)
    results["unigram_bits_per_byte"] = evaluate.unigram_bits_per_byte(load_tokens(data / "train.bin"),
                                                                      test_tokens[: args.max_tokens or None], lengths)
    test_docs = read_docs(data / "test.jsonl")
    results["gzip_bits_per_byte"] = evaluate.gzip_bits_per_byte("\n".join(d["text"] for d in test_docs[:5000]))

    train_docs = read_docs(data / "train.jsonl")
    if any(d["label"] for d in train_docs) and any(d["label"] for d in test_docs):
        print("sentiment probe ...", flush=True)
        x_text, y = labelled(train_docs, args.probe_train, seed=0)
        t_text, t_y = labelled(test_docs, args.probe_test, seed=1)
        for name, features in [("model", lambda texts: evaluate.review_features(model, tokenizer, texts, device, dtype)),
                               ("bag_of_words", lambda texts: evaluate.bag_of_words(tokenizer, texts))]:
            predict = evaluate.train_probe(features(x_text), y)
            results[f"sentiment_accuracy_{name}"] = (predict(features(t_text)) == t_y).float().mean().item()

    print("memorization ...", flush=True)
    rng = random.Random(2)
    docs = [tokenizer.encode(d["text"], allow_special=False) for d in rng.sample(train_docs, min(args.memorization, len(train_docs)))]
    results["extraction_rate"] = evaluate.extraction_rate(model, docs, device)
    results["memorization_documents"] = sum(len(d) >= 64 for d in docs)

    (Path(args.run) / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{'':<36}{'value':>10}")
    rows = [("bits per byte: your model", "bits_per_byte"), ("bits per byte: token frequencies", "unigram_bits_per_byte"),
            ("bits per byte: gzip", "gzip_bits_per_byte"), ("sentiment accuracy: your model", "sentiment_accuracy_model"),
            ("sentiment accuracy: word counts", "sentiment_accuracy_bag_of_words"),
            ("training reviews continued exactly", "extraction_rate")]
    for label, key in rows:
        if key in results:
            value = results[key]
            print(f"{label:<36}{value:>10.1%}" if "accuracy" in key or "rate" in key else f"{label:<36}{value:>10.3f}")
    print(f"\nwritten to {Path(args.run) / 'eval.json'}")


if __name__ == "__main__":
    main()
