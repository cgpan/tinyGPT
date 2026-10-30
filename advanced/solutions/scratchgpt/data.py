"""Project 1: turn raw reviews into clean, deduplicated, decontaminated documents."""

import hashlib
import html
import re
import unicodedata
import zlib
from collections import defaultdict

import numpy as np

PRIME = 4294967311                       # the smallest prime above 2**32


def clean(text):
    """Turn one raw review into clean text.

    1. Every HTML line break (<br />, <br/>, <br>, in any case) becomes a newline.
    2. HTML entities are decoded: &amp; becomes &, &#39; becomes '.
    3. Unicode is put in NFC form, so an accented letter is always one character.
    4. Each run of spaces, tabs and non-breaking spaces becomes one space, and spaces at
       the start and end of every line are removed.
    5. Three or more newlines in a row become two: at most one blank line.
    6. Whitespace at the start and end is removed.
    """
    # BEGIN SOLUTION 1
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
    # END SOLUTION


def scrub(text):
    """Replace personal details: email addresses with [email], web addresses with [url],
    and phone numbers with [phone]. Returns (text, counts), where counts says how many of
    each were replaced, e.g. {"email": 1, "url": 0, "phone": 0}.

    A web address starts with http://, https:// or www. and runs to the next whitespace.
    A phone number is a North American one: 555-123-4567, (555) 123-4567, 555.123.4567 or
    +1 555 123 4567. Years, ratings like 7/10 and other numbers are left alone.
    Emails are replaced first, then web addresses, then phone numbers.
    """
    # BEGIN SOLUTION 1
    patterns = {
        "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
        "url": r"(?:https?://|www\.)\S+",
        "phone": r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w-])",
    }
    counts = {}
    for name, pattern in patterns.items():
        text, counts[name] = re.subn(pattern, f"[{name}]", text, flags=re.IGNORECASE)
    return text, counts
    # END SOLUTION


def fingerprint(text):
    """A fingerprint for exact duplicates: reviews that differ only in upper and lower case,
    whitespace or punctuation get the same one. Lower-case the text, replace every run of
    characters that are not letters or digits with one space, strip it, and return the
    SHA-1 hex digest of its UTF-8 bytes."""
    # BEGIN SOLUTION 1
    normal = re.sub(r"[\W_]+", " ", text.lower()).strip()
    return hashlib.sha1(normal.encode("utf-8")).hexdigest()
    # END SOLUTION


def shingles(text, n=5):
    """The set of word n-grams in text: lower-cased runs of letters and digits (regex \\w+),
    joined by single spaces. A text with fewer than n words gives one shingle of all its
    words; a text with no words gives the empty set."""
    # BEGIN SOLUTION 1
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    # END SOLUTION


def hash_functions(num_hashes, seed=0):
    """The (a, b) pairs of the hash functions h(x) = (a * x + b) mod PRIME. Given."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**32 - 1, size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, 2**32 - 1, size=num_hashes, dtype=np.uint64)
    return a, b


def minhash(shingle_set, num_hashes=128, seed=0):
    """The MinHash signature of a set of shingles: an array of num_hashes unsigned 64-bit
    integers. Each shingle becomes a number x = zlib.crc32 of its UTF-8 bytes; entry i of
    the signature is the smallest h_i(x) over all shingles, with the hash functions from
    hash_functions(num_hashes, seed). The empty set's signature is PRIME everywhere.

    Two sets agree in each entry with probability equal to their Jaccard similarity
    (shared shingles divided by all shingles), so the fraction of matching entries
    estimates it.
    """
    # BEGIN SOLUTION 1
    a, b = hash_functions(num_hashes, seed)
    if not shingle_set:
        return np.full(num_hashes, PRIME, dtype=np.uint64)
    x = np.array([zlib.crc32(s.encode("utf-8")) for s in shingle_set], dtype=np.uint64)
    return ((a[:, None] * x[None, :] + b[:, None]) % PRIME).min(axis=1)
    # END SOLUTION


def estimated_jaccard(signature_a, signature_b):
    """The fraction of entries where the two signatures agree."""
    # BEGIN SOLUTION 1
    return float(np.mean(signature_a == signature_b))
    # END SOLUTION


def candidate_pairs(signatures, bands=16):
    """Locality-sensitive hashing. Cut each signature into `bands` equal bands; two documents
    are candidates when all of one band's entries match. Returns a set of pairs (i, j) with
    i < j, indexing into signatures. Pairs with high similarity are very likely to be found;
    pairs with low similarity rarely are, so you never compare every pair."""
    # BEGIN SOLUTION 1
    rows = len(signatures[0]) // bands
    pairs = set()
    for band in range(bands):
        buckets = defaultdict(list)
        for index, signature in enumerate(signatures):
            buckets[signature[band * rows : (band + 1) * rows].tobytes()].append(index)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs
    # END SOLUTION


def near_duplicates(texts, threshold=0.8, num_hashes=128, bands=16, n=5):
    """Pairs of texts that are nearly the same: a list of (i, j, similarity), i < j, sorted,
    for every candidate pair whose estimated Jaccard similarity is at least threshold."""
    # BEGIN SOLUTION 1
    signatures = [minhash(shingles(t, n), num_hashes) for t in texts]
    found = []
    for i, j in sorted(candidate_pairs(signatures, bands)):
        similarity = estimated_jaccard(signatures[i], signatures[j])
        if similarity >= threshold:
            found.append((i, j, similarity))
    return found
    # END SOLUTION


def build_splits(reviews, val_fraction=0.1, seed=0, threshold=0.8):
    """Turn the reviews from imdb.read_reviews into training, validation and test documents.

    1. Clean and scrub every review.
    2. Training documents come from the 'train' and 'unsup' reviews; held-out documents from
       the 'test' reviews.
    3. Drop exact duplicates (same fingerprint) within training and within held-out,
       keeping the first one.
    4. Drop near-duplicates (near_duplicates with this threshold) within training, keeping
       the first one of each pair.
    5. Decontaminate: drop every training document that is an exact or near duplicate of a
       held-out document. A model must never train on its own test.
    6. Shuffle the held-out documents with np.random.default_rng(seed).permutation; the
       first round(val_fraction * count) are validation, the rest test.

    Returns (splits, stats). splits maps 'train', 'val' and 'test' to lists of
    {"id", "text", "label"} in the order above. stats counts what happened: "reviews",
    "email", "url", "phone", "exact_duplicates", "near_duplicates", "contaminated", and the
    documents in each split ("train", "val", "test").
    """
    # BEGIN SOLUTION 1
    stats = defaultdict(int)
    train, held_out = [], []
    for review in reviews:
        stats["reviews"] += 1
        text, counts = scrub(clean(review["text"]))
        for name, count in counts.items():
            stats[name] += count
        doc = {"id": review["id"], "text": text, "label": review["label"]}
        (held_out if review["split"] == "test" else train).append(doc)

    def drop_exact(docs):
        seen, kept = set(), []
        for doc in docs:
            key = fingerprint(doc["text"])
            if key in seen:
                stats["exact_duplicates"] += 1
            else:
                seen.add(key)
                kept.append(doc)
        return kept, seen

    train, _ = drop_exact(train)
    held_out, held_out_keys = drop_exact(held_out)

    everything = train + held_out
    drop = set()
    for i, j, _ in near_duplicates([d["text"] for d in everything], threshold):
        if j < len(train):                          # both in training: keep the first
            if j not in drop:
                drop.add(j)
                stats["near_duplicates"] += 1
        elif i < len(train) and i not in drop:      # training near a held-out document
            drop.add(i)
            stats["contaminated"] += 1
    for i, doc in enumerate(train):
        if i not in drop and fingerprint(doc["text"]) in held_out_keys:
            drop.add(i)
            stats["contaminated"] += 1
    train = [doc for i, doc in enumerate(train) if i not in drop]

    order = np.random.default_rng(seed).permutation(len(held_out))
    n_val = round(val_fraction * len(held_out))
    splits = {
        "train": train,
        "val": [held_out[i] for i in order[:n_val]],
        "test": [held_out[i] for i in order[n_val:]],
    }
    for name, docs in splits.items():
        stats[name] = len(docs)
    return splits, dict(stats)
    # END SOLUTION
