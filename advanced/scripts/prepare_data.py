"""Project 1: download the IMDB reviews and turn them into clean training, validation and
test documents.

    python scripts/prepare_data.py                 # writes data/imdb/
    python scripts/prepare_data.py --data other/   # somewhere else

Writes train.jsonl, val.jsonl and test.jsonl (one {"id", "text", "label"} per line) and
stats.json (what the pipeline did), then prints the numbers for your data card.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from scratchgpt import data, imdb


def main():
    sys.stdout.reconfigure(line_buffering=True, errors="replace")   # print as it happens; never crash on a character
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data", help="where to download and write (default: data)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.8, help="near-duplicate Jaccard similarity")
    args = parser.parse_args()

    root = imdb.download(args.data)
    out = Path(args.data) / "imdb"
    out.mkdir(parents=True, exist_ok=True)
    print("Cleaning, scrubbing and deduplicating the reviews (a few minutes for all of IMDB) ...")
    start = time.perf_counter()
    splits, stats = data.build_splits(imdb.read_reviews(root), args.val_fraction, threshold=args.threshold)
    stats["seconds"] = round(time.perf_counter() - start, 1)

    for name, docs in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        stats[f"{name}_bytes"] = sum(len(d["text"].encode("utf-8")) for d in docs)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n{'':<28}{'count':>12}")
    for key in ["reviews", "exact_duplicates", "near_duplicates", "contaminated", "email", "url", "phone",
                "train", "val", "test"]:
        print(f"{key.replace('_', ' '):<28}{stats.get(key, 0):>12,}")
    for name in ["train", "val", "test"]:
        print(f"{name + ' text (MB)':<28}{stats[name + '_bytes'] / 1e6:>12.1f}")
    print(f"\nWritten to {out} in {stats['seconds']} s.")


if __name__ == "__main__":
    main()
