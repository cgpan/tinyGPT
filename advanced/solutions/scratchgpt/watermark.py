"""Project 8: a hidden watermark in generated text, and the test that finds it.

Kirchenbauer, Geiping, Wen, Katz, Miers and Goldstein (2023), A Watermark for Large
Language Models. Before each new token, the previous token and a secret key pick a random
"green" fraction gamma of the vocabulary, and the sampler adds delta to the green tokens'
scores. A reader can't see the difference, but text with far more green tokens than
gamma would give by chance was almost certainly written by the model.

green_ids and mark are given: the sampler from project 4 uses them by default. You write
detect.
"""

import functools
import hashlib
import math

import torch

KEY = "scratchgpt"       # the secret: change it for your own model, and keep it private
GAMMA = 0.25             # the green fraction of the vocabulary
DELTA = 2.0              # how much the green tokens' scores are raised


@functools.lru_cache(maxsize=65536)
def green_ids(previous, vocab_size, key=KEY, gamma=GAMMA):
    """The green tokens after token `previous`: the first round(gamma * vocab_size) entries of
    a random permutation of the vocabulary, seeded by the key and the previous token. Given."""
    seed = int.from_bytes(hashlib.sha256(f"{key}:{previous}".encode("utf-8")).digest()[:8], "little")
    generator = torch.Generator().manual_seed(seed % 2**63)
    return torch.randperm(vocab_size, generator=generator)[: round(gamma * vocab_size)]


def mark(logits, previous, key=KEY, gamma=GAMMA, delta=DELTA):
    """Raise the green tokens' scores by delta. logits: (B, V); previous: (B,) the last token
    of each row. Returns new logits. Given."""
    logits = logits.clone()
    for row, prev in enumerate(previous.tolist()):
        logits[row, green_ids(prev, logits.shape[-1], key, gamma).to(logits.device)] += delta
    return logits


def detect(ids, vocab_size, key=KEY, gamma=GAMMA):
    """How strongly the token ids carry the watermark: a z-score.

    For every token after the first, check whether it is in green_ids(the token before it).
    With T tokens checked and G of them green, unmarked text has about gamma * T green
    tokens by chance, so
        z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma)).
    Around 0 means no watermark; above 4 almost certainly means one. Fewer than 2 tokens
    give 0.0.
    """
    # BEGIN SOLUTION 8
    ids = [int(i) for i in ids]
    if len(ids) < 2:
        return 0.0
    green = sum(1 for prev, tok in zip(ids, ids[1:]) if tok in set(green_ids(prev, vocab_size, key, gamma).tolist()))
    checked = len(ids) - 1
    return (green - gamma * checked) / math.sqrt(checked * gamma * (1 - gamma))
    # END SOLUTION
