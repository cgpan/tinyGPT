"""Project 7: how good is the model? Bits per byte, sentiment, and memorization."""

import gzip
import math

import torch

from scratchgpt.device import autocast
from scratchgpt.optim import AdamW
from scratchgpt.sampling import generate


def token_byte_lengths(tokenizer):
    """A long tensor of shape (vocab_size,): how many bytes of text each token stands for,
    and 0 for special tokens, which stand for no text."""
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


@torch.no_grad()
def bits_per_byte(model, tokens, byte_lengths, device, dtype=None, batch_size=8, max_tokens=None):
    """The model's loss in bits per byte of text: the number that compares models with
    different tokenizers fairly.

    Cut tokens (an array of ids; only the first max_tokens if given) into windows of
    context + 1 tokens that overlap by one (starting at 0, context, 2 * context, ...; drop a
    last window shorter than 2 tokens); in each, the model predicts tokens 1..end from the
    ones before. Sum the surprise -log p (in nats) at every target that stands for text
    (byte length above 0), and divide by ln 2 times those targets' total bytes.
    """
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


def unigram_bits_per_byte(train_tokens, val_tokens, byte_lengths):
    """A baseline: a model that ignores context and gives each token its frequency in the
    training tokens, with one extra count for every token (add-one smoothing). Bits per byte
    on val_tokens, counted the same way as bits_per_byte."""
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


def gzip_bits_per_byte(text):
    """Another baseline: how small gzip makes the text, in bits per byte. Given."""
    data = text.encode("utf-8")
    return 8 * len(gzip.compress(data, compresslevel=9)) / len(data)


@torch.no_grad()
def review_features(model, tokenizer, texts, device, dtype=None, batch_size=32):
    """One vector per text: the model's final hidden states (model(ids, hidden=True)),
    averaged over the text's first cfg.context tokens. Returns a float32 CPU tensor
    (len(texts), d_model). Pad shorter texts in a batch, and leave the padding out of the
    average."""
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


def bag_of_words(tokenizer, texts):
    """A baseline's features: for each text, how often each vocabulary token occurs in it,
    divided by its number of tokens. A float32 tensor (len(texts), vocab_size)."""
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


def train_probe(features, labels, steps=300, lr=0.01, weight_decay=1e-3, seed=0):
    """Logistic regression from features (N, d) to labels (N,) of 0 and 1, the classifier that
    asks whether the features already contain the answer.

    Standardise each feature with the training features' mean and standard deviation (plus
    1e-6). Start w (d,) and b at zero and take `steps` full-batch steps of your AdamW
    (weight_decay on w only) on the mean of log(1 + exp(-s * (x w + b))), with s = +1 for
    label 1 and -1 for label 0. Write that loss so it can't overflow:
    log(1 + exp(z)) = max(z, 0) + log(1 + exp(-|z|)), with max(z, 0) written as (z + |z|) / 2.
    (Why not z.clamp(min=0)? At the start every z is exactly 0, where PyTorch gives clamp, and
    |z|, a gradient of 0: the probe would never move. (z + |z|) / 2 gets the true 1/2.)
    Returns a function from features to predicted labels (a long tensor of 0s and 1s).
    """
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")


@torch.no_grad()
def extraction_rate(model, documents, device, prefix=32, suffix=32, batch_size=32):
    """Memorization: give the model the first `prefix` tokens of each training document and
    let it continue greedily (temperature 0, no watermark) for `suffix` tokens. Returns the
    fraction of documents whose next `suffix` tokens it reproduces exactly. documents: lists
    of token ids; ones shorter than prefix + suffix are skipped."""
    raise NotImplementedError("write this in project 7: see projects/07_evaluation.md")
