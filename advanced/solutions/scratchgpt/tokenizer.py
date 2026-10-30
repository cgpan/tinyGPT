"""Project 2: a byte-level BPE tokenizer, trained on your data.

Every text is UTF-8 bytes, so the 256 byte values are the first 256 tokens and any text
at all can be encoded. Training then merges the most frequent pair of neighbouring tokens
into a new token, again and again, until the vocabulary is full.
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EOT = "<|endoftext|>"     # marks the end of each document

# Splits text into words before BPE, so merges never cross a word boundary. GPT-2's
# pattern, rewritten for Python's own re module. Given.
SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


def pretokenize(text):
    """Cut text into the pieces BPE works on. Joining them gives back the text. Given."""
    return SPLIT_PATTERN.findall(text)


def count_words(texts):
    """How often each piece occurs across texts: a Counter of strings. Given."""
    counts = Counter()
    for text in texts:
        counts.update(pretokenize(text))
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of pair in the list ids by new_id, left to right, without
    overlapping: merge([1, 1, 1], (1, 1), 9) is [9, 1]. Returns a new list."""
    # BEGIN SOLUTION 2
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
    # END SOLUTION


def train_bpe(word_counts, vocab_size, special_tokens=(EOT,), verbose=False):
    """Learn BPE merges from word_counts (a dict from piece to how often it occurs).

    Every piece starts as its list of UTF-8 bytes. Then, vocab_size - 256 - len(special_tokens)
    times: count every pair of neighbouring tokens (a piece that occurs 10 times counts 10
    times), take the most frequent pair, and merge it into a new token. Merge i (counting
    from 0) creates token 256 + i. Ties go to the pair that is largest when compared as a
    tuple of token ids, which is what max(counts, key=lambda p: (counts[p], p)) does. Stop
    early if no pair is left.

    Returns the merges: a list of (left id, right id) pairs, in the order they were learned.

    The simple way recounts every pair after every merge, which is fine for a megabyte but
    takes hours on all of IMDB. Stretch: after a merge, update only the counts of the
    pieces that contained the merged pair.
    """
    # BEGIN SOLUTION 2
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")
    words = [list(w.encode("utf-8")) for w in word_counts]
    freqs = list(word_counts.values())
    pair_counts = Counter()
    where = defaultdict(set)                         # pair -> the words that contain it
    for index, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            where[pair].add(index)

    # A heap finds the most frequent pair without looking at all of them. Entries go stale
    # when a count changes; a popped entry is used only if it still matches the count.
    def key(pair):
        return (-pair_counts[pair], -pair[0], -pair[1])   # most frequent, then largest pair
    heap = [key(pair) for pair in pair_counts]
    heapq.heapify(heap)

    merges = []
    for step in range(num_merges):
        best = None
        while heap:
            negative_count, a, b = heapq.heappop(heap)
            pair = (-a, -b)
            if pair_counts.get(pair) == -negative_count:
                best = pair
                break
        if best is None:
            break
        new_id = 256 + step
        merges.append(best)
        changed = set()
        for index in list(where[best]):
            word, freq = words[index], freqs[index]
            for pair in zip(word, word[1:]):         # forget the old word's pairs ...
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                where[pair].discard(index)
                changed.add(pair)
            word = merge(word, best, new_id)
            words[index] = word
            for pair in zip(word, word[1:]):         # ... and count the new word's
                pair_counts[pair] += freq
                where[pair].add(index)
                changed.add(pair)
        for pair in changed:
            if pair in pair_counts:
                heapq.heappush(heap, key(pair))
        if verbose and (step + 1) % 1000 == 0:
            print(f"  {step + 1:,} of {num_merges:,} merges")
    return merges
    # END SOLUTION


class Tokenizer:
    """Encodes text as token ids and decodes them back."""

    def __init__(self, merges, special_tokens=(EOT,)):
        # Given: the vocabulary is the 256 bytes, then one token per merge, then the
        # special tokens.
        self.merges = [tuple(pair) for pair in merges]
        self.special_tokens = list(special_tokens)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (left, right) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[left] + self.vocab[right]
        self.special_ids = {s: 256 + len(self.merges) + i for i, s in enumerate(self.special_tokens)}
        for s, i in self.special_ids.items():
            self.vocab[i] = s.encode("utf-8")
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}   # earlier merges first
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eot_id(self):
        return self.special_ids[EOT]

    def encode_piece(self, piece):
        """Encode one piece from pretokenize: start from its UTF-8 bytes, then repeatedly
        apply the merge with the lowest rank (the earliest learned) among its neighbouring
        pairs, until no pair in it has been merged in training."""
        # BEGIN SOLUTION 2
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, 256 + self.ranks[pair])
        return ids
        # END SOLUTION

    def encode(self, text, allow_special=True):
        """Encode text as a list of token ids. With allow_special, a special token written in
        the text (such as <|endoftext|>) becomes its own id; otherwise it is encoded like any
        other text. Everything else is cut by pretokenize and each piece encoded with
        encode_piece. Keep the pieces you have already encoded in self.cache: most words
        occur many times."""
        # BEGIN SOLUTION 2
        if allow_special and self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        ids = []
        for part in parts:
            if part in self.special_ids and allow_special:
                ids.append(self.special_ids[part])
                continue
            for piece in pretokenize(part):
                if piece not in self.cache:
                    self.cache[piece] = self.encode_piece(piece)
                ids.extend(self.cache[piece])
        return ids
        # END SOLUTION

    def decode(self, ids):
        """Turn token ids back into text: join their bytes, then decode as UTF-8, replacing
        any invalid bytes with U+FFFD (a model can write a byte sequence that isn't valid
        UTF-8)."""
        # BEGIN SOLUTION 2
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")
        # END SOLUTION

    def save(self, path):
        """Given."""
        data = {"merges": self.merges, "special_tokens": self.special_tokens}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Given."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["merges"], data["special_tokens"])


def encode_documents(tokenizer, texts, path):
    """Encode every text, each followed by <|endoftext|>, and write all the ids to path as
    one array of unsigned 16-bit integers (so the vocabulary must be under 65,536). Returns
    {"documents", "tokens", "bytes"}, where bytes counts the UTF-8 bytes of the texts. Given.
    """
    if tokenizer.vocab_size > 2**16:
        raise ValueError("vocabularies above 65,536 tokens don't fit in 16 bits")
    chunks, n_bytes, n_docs = [], 0, 0
    for text in texts:
        ids = tokenizer.encode(text, allow_special=False)
        ids.append(tokenizer.eot_id)
        chunks.append(np.array(ids, dtype=np.uint16))
        n_bytes += len(text.encode("utf-8"))
        n_docs += 1
    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    tokens.tofile(path)
    return {"documents": n_docs, "tokens": int(len(tokens)), "bytes": n_bytes}
