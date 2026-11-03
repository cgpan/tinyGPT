"""Shared test fixtures: a tiny fake IMDB folder, tiny configs, tiny token files."""

import json
import random

import numpy as np
import pytest
import torch

torch.set_num_threads(1)        # tiny models: threads only add overhead


def long_review(seed, n_words=100):
    """A long made-up review, the same for the same seed."""
    words = ("the film story acting music scene ending director plot camera actor role night "
             "city love war friend family house car dog old young first last good bad slow").split()
    rng = random.Random(seed)
    return " ".join(rng.choice(words) for _ in range(n_words))


# Made-up reviews in IMDB's folder layout. What the tests expect from them:
#   train/unsup/1 repeats train/pos/0 (case and punctuation aside): an exact duplicate
#   train/unsup/0 is train/pos/1 with one word changed: a near-duplicate
#   train/neg/1 repeats test/neg/0: contamination
#   train/neg/0 has an email address and a phone number; train/pos/1 a web address
LONG = long_review(1)
FAKE_REVIEWS = {
    ("train", "pos", 0, 9): "A wonderful film.<br /><br />The acting was great &amp; the music was lovely.",
    ("train", "pos", 1, 8): LONG + " More at www.example.com/reviews",
    ("train", "neg", 0, 2): "Terrible. Write to fan@example.com or call 555-123-4567 to complain.",
    ("train", "neg", 1, 1): "The worst movie I have seen in years; avoid it!",
    ("train", "unsup", 0, 0): LONG.replace("the", "a", 1) + " More at www.example.com/reviews",
    ("train", "unsup", 1, 0): "a wonderful film. the acting was great & the music was lovely",
    ("test", "pos", 0, 10): "Simply the best picture of the decade, a true classic.",
    ("test", "pos", 1, 7): long_review(2),
    ("test", "neg", 0, 3): "the worst movie i have seen in years. avoid it",
    ("test", "neg", 1, 4): "Dull and far too long; I left halfway through.",
}


@pytest.fixture
def imdb_root(tmp_path):
    root = tmp_path / "aclImdb"
    for (split, label, number, rating), text in FAKE_REVIEWS.items():
        folder = root / split / label
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{number}_{rating}.txt").write_text(text, encoding="utf-8")
    for split, label in [("train", "pos"), ("train", "neg"), ("train", "unsup"), ("test", "pos"), ("test", "neg")]:
        code = {"pos": 1, "neg": 2, "unsup": 3}[label] + (0 if split == "train" else 3)
        lines = [f"http://www.imdb.com/title/tt{code}{n:06d}/usercomments" for n in range(2)]
        (root / split / f"urls_{label}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def tiny_config(**changes):
    from scratchgpt import Config
    settings = dict(vocab_size=64, context=16, n_layer=2, n_head=2, d_model=32, dropout=0.0,
                    batch_size=4, grad_accum=1, max_steps=20, warmup_steps=2, eval_interval=10,
                    eval_steps=2, checkpoint_interval=10, lr=1e-2, min_lr=1e-3, device="cpu",
                    dtype="float32")
    settings.update(changes)
    return Config(**settings)


@pytest.fixture
def token_dir(tmp_path):
    """train.bin, val.bin and meta.json holding an easy pattern: counting 0 to 31, again and
    again, with token 63 (the end-of-text token) in between."""
    pattern = np.array(list(range(32)) + [63], dtype=np.uint16)
    np.tile(pattern, 300).tofile(tmp_path / "train.bin")
    np.tile(pattern, 30).tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")
    return tmp_path
