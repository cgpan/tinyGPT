"""Download and read the IMDB Large Movie Review Dataset. Given: nothing to write here.

Maas, Daly, Pham, Huang, Ng and Potts (2011), Learning Word Vectors for Sentiment
Analysis. https://ai.stanford.edu/~amaas/data/sentiment/

100,000 reviews: 25,000 for training and 25,000 for testing, each half positive (rating
7 to 10) and half negative (1 to 4), plus 50,000 without a label. The training and test
reviews are about different films.
"""

import re
import tarfile
import urllib.request
from pathlib import Path

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
FOLDERS = [("train", "pos"), ("train", "neg"), ("train", "unsup"), ("test", "pos"), ("test", "neg")]


def download(dest="data"):
    """Download (84 MB) and unpack the reviews into dest/aclImdb, once. Returns that folder."""
    dest = Path(dest)
    root = dest / "aclImdb"
    if (root / "test" / "neg").is_dir():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "aclImdb_v1.tar.gz"
    if not archive.exists():
        print(f"Downloading {URL} (84 MB) ...")
        part = archive.with_name(archive.name + ".part")
        urllib.request.urlretrieve(URL, part)
        part.replace(archive)
    print("Unpacking ...")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:                                         # older Python: check the paths ourselves
            for member in tar.getmembers():
                if not (dest / member.name).resolve().is_relative_to(dest.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
            tar.extractall(dest)
    return root


def read_reviews(root):
    """Yield every review as a dict:

        id      'train/pos/123', the folder and the review's number
        split   'train', 'unsup' or 'test'
        label   'pos', 'neg', or None for unlabelled reviews
        rating  1 to 10, or None for unlabelled reviews
        movie   the film's IMDB id ('tt0453418'), from the urls file
        text    the review exactly as written, HTML and all
    """
    root = Path(root)
    for split, label in FOLDERS:
        folder = root / split / label
        urls_file = root / split / f"urls_{label}.txt"
        urls = urls_file.read_text(encoding="utf-8").split() if urls_file.exists() else []
        files = sorted(folder.glob("*.txt"), key=lambda p: int(p.stem.split("_")[0]))
        for path in files:
            number, rating = (int(part) for part in path.stem.split("_"))
            found = re.search(r"tt\d+", urls[number]) if number < len(urls) else None
            yield {
                "id": f"{split}/{label}/{number}",
                "split": "unsup" if label == "unsup" else split,
                "label": None if label == "unsup" else label,
                "rating": rating or None,
                "movie": found.group(0) if found else None,
                "text": path.read_text(encoding="utf-8", errors="replace"),
            }
