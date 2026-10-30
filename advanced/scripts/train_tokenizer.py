"""Project 2: train a BPE tokenizer on the training reviews, then encode every split.

    python scripts/train_tokenizer.py                       # data/imdb, 8,192 tokens
    python scripts/train_tokenizer.py --vocab-size 16384
    python scripts/train_tokenizer.py --sample-mb 10        # learn merges from 10 MB only

Writes tokenizer.json, train.bin, val.bin, test.bin and meta.json next to the data.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from scratchgpt.tokenizer import Tokenizer, count_words, encode_documents, train_bpe


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def main():
    sys.stdout.reconfigure(line_buffering=True)   # show progress as it happens, in Colab too
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/imdb")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--sample-mb", type=float, default=0, help="learn merges from this much text (0: all)")
    args = parser.parse_args()
    data = Path(args.data)

    texts = read_texts(data / "train.jsonl")
    sample, size = texts, 0
    if args.sample_mb:
        sample = []
        for text in texts:
            sample.append(text)
            size += len(text.encode("utf-8"))
            if size >= args.sample_mb * 1e6:
                break
    start = time.perf_counter()
    words = count_words(sample)
    print(f"{len(sample):,} documents, {len(words):,} different pieces; learning "
          f"{args.vocab_size - 257:,} merges ...")
    tokenizer = Tokenizer(train_bpe(words, args.vocab_size, verbose=True))
    trained = time.perf_counter() - start
    tokenizer.save(data / "tokenizer.json")
    print(f"learned {len(tokenizer.merges):,} merges in {trained:.0f} s")

    meta = {"vocab_size": tokenizer.vocab_size, "seconds_to_train": round(trained, 1), "splits": {}}
    for split in ["train", "val", "test"]:
        start = time.perf_counter()
        split_texts = texts if split == "train" else read_texts(data / f"{split}.jsonl")
        stats = encode_documents(tokenizer, split_texts, data / f"{split}.bin")
        stats["seconds"] = round(time.perf_counter() - start, 1)
        meta["splits"][split] = stats
    meta["bytes_per_token"] = round(meta["splits"]["val"]["bytes"] / meta["splits"]["val"]["tokens"], 3)
    (data / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'split':<8}{'documents':>12}{'tokens':>14}{'MB of text':>12}{'bytes/token':>13}")
    for split, s in meta["splits"].items():
        print(f"{split:<8}{s['documents']:>12,}{s['tokens']:>14,}{s['bytes'] / 1e6:>12.1f}"
              f"{s['bytes'] / s['tokens']:>13.2f}")
    longest = sorted(tokenizer.vocab.values(), key=len)[-8:]
    print("\nsome of the longest tokens:", ", ".join(repr(t.decode("utf-8", "replace")) for t in longest))
    example = "This movie was absolutely wonderful, I loved it!"
    print(f"\n{example!r} ->")
    print(" | ".join(tokenizer.decode([i]) for i in tokenizer.encode(example)))


if __name__ == "__main__":
    main()
