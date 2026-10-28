"""Counting bigram model: the number the transformer has to beat."""

import numpy as np


class CountBigram:
    """Predict the next character by counting which characters follow which.

    counts[a, b] is how many times character b came right after character a.
    No neural network, no training loop: just counting.
    """

    def __init__(self, vocab_size, alpha=1.0):
        self.vocab_size = vocab_size
        self.alpha = alpha    # smoothing: pretend every pair was seen `alpha` extra times
        self.counts = np.zeros((vocab_size, vocab_size))

    def fit(self, ids):
        """Count every pair of neighbouring characters in `ids`."""
        ids = np.asarray(ids).tolist()
        for a, b in zip(ids[:-1], ids[1:]):
            self.counts[a, b] += 1
        return self

    def probs(self):
        """Turn each row of counts into probabilities that sum to 1."""
        smoothed = self.counts + self.alpha
        return smoothed / smoothed.sum(axis=1, keepdims=True)

    def loss(self, ids):
        """Average surprise, -log p(next character), in nats.

        This is the same quantity the transformer's loss measures, so the two
        numbers can be compared directly.
        """
        ids = np.asarray(ids)
        p = self.probs()[ids[:-1], ids[1:]]
        return float(-np.log(p).mean())

    def generate(self, n, start=0, seed=0):
        """Sample `n` characters, each one drawn using only the character before it."""
        rng = np.random.default_rng(seed)
        probs = self.probs()
        out = [start]
        for _ in range(n):
            out.append(int(rng.choice(self.vocab_size, p=probs[out[-1]])))
        return out
