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
    # BEGIN SOLUTION 7
    lengths = torch.tensor([len(tokenizer.vocab[i]) for i in range(tokenizer.vocab_size)], dtype=torch.long)
    lengths[list(tokenizer.special_ids.values())] = 0
    return lengths
    # END SOLUTION


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
    # BEGIN SOLUTION 7
    context = model.cfg.context
    ids = torch.as_tensor(tokens[: max_tokens or len(tokens)].astype("int64"))
    windows = [ids[s : s + context + 1] for s in range(0, len(ids) - 1, context)]
    windows = [w for w in windows if len(w) >= 2]
    nats, n_bytes = 0.0, 0
    model.eval()
    for i in range(0, len(windows), batch_size):
        group = windows[i : i + batch_size]
        for length in sorted({len(w) for w in group}):        # only the last window is short
            same = torch.stack([w for w in group if len(w) == length]).to(device)
            x, y = same[:, :-1], same[:, 1:]
            with autocast(device, dtype):
                logits, _ = model(x)
            logits = logits.float()
            m = logits.amax(dim=-1, keepdim=True)
            logsumexp = m.squeeze(-1) + (logits - m).exp().sum(dim=-1).log()
            surprise = logsumexp - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            weights = byte_lengths.to(device)[y]
            nats += surprise[weights > 0].sum().item()
            n_bytes += weights.sum().item()
    return nats / (math.log(2) * n_bytes)
    # END SOLUTION


def unigram_bits_per_byte(train_tokens, val_tokens, byte_lengths):
    """A baseline: a model that ignores context and gives each token its frequency in the
    training tokens, with one extra count for every token (add-one smoothing). Bits per byte
    on val_tokens, counted the same way as bits_per_byte."""
    # BEGIN SOLUTION 7
    vocab_size = len(byte_lengths)
    counts = torch.bincount(torch.as_tensor(train_tokens.astype("int64")), minlength=vocab_size).double() + 1
    log_p = counts.log() - counts.sum().log()
    val = torch.as_tensor(val_tokens[1:].astype("int64"))
    weights = byte_lengths[val]
    nats = -log_p[val][weights > 0].sum().item()
    return nats / (math.log(2) * weights.sum().item())
    # END SOLUTION


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
    # BEGIN SOLUTION 7
    model.eval()
    context = model.cfg.context
    out = []
    for i in range(0, len(texts), batch_size):
        encoded = [tokenizer.encode(t, allow_special=False)[:context] or [tokenizer.eot_id]
                   for t in texts[i : i + batch_size]]
        length = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), length), tokenizer.eot_id, dtype=torch.long)
        keep = torch.zeros(len(encoded), length, 1)
        for row, e in enumerate(encoded):
            ids[row, : len(e)] = torch.tensor(e)
            keep[row, : len(e)] = 1
        with autocast(device, dtype):
            h = model(ids.to(device), hidden=True)
        keep = keep.to(device)
        out.append(((h.float() * keep).sum(dim=1) / keep.sum(dim=1)).cpu())
    return torch.cat(out)
    # END SOLUTION


def bag_of_words(tokenizer, texts):
    """A baseline's features: for each text, how often each vocabulary token occurs in it,
    divided by its number of tokens. A float32 tensor (len(texts), vocab_size)."""
    # BEGIN SOLUTION 7
    out = torch.zeros(len(texts), tokenizer.vocab_size)
    for row, text in enumerate(texts):
        ids = torch.tensor(tokenizer.encode(text, allow_special=False) or [tokenizer.eot_id])
        out[row] = torch.bincount(ids, minlength=tokenizer.vocab_size).float() / len(ids)
    return out
    # END SOLUTION


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
    # BEGIN SOLUTION 7
    torch.manual_seed(seed)
    mean, std = features.mean(dim=0), features.std(dim=0) + 1e-6
    x = (features - mean) / std
    s = labels.float() * 2 - 1
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = AdamW([{"params": [w], "weight_decay": weight_decay}, {"params": [b], "weight_decay": 0.0}], lr=lr)
    for _ in range(steps):
        z = -s * (x @ w + b)
        loss = ((z + z.abs()) / 2 + (-z.abs()).exp().log1p()).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    w, b = w.detach(), b.detach()
    return lambda f: (((f - mean) / std) @ w + b > 0).long()
    # END SOLUTION


@torch.no_grad()
def extraction_rate(model, documents, device, prefix=32, suffix=32, batch_size=32):
    """Memorization: give the model the first `prefix` tokens of each training document and
    let it continue greedily (temperature 0, no watermark) for `suffix` tokens. Returns the
    fraction of documents whose next `suffix` tokens it reproduces exactly. documents: lists
    of token ids; ones shorter than prefix + suffix are skipped."""
    # BEGIN SOLUTION 7
    usable = [d[: prefix + suffix] for d in documents if len(d) >= prefix + suffix]
    if not usable:
        return 0.0
    hits = 0
    for i in range(0, len(usable), batch_size):
        batch = torch.tensor(usable[i : i + batch_size], device=device)
        out = generate(model, batch[:, :prefix], suffix, temperature=0, watermark=False, use_cache=True)
        hits += (out[:, prefix:] == batch[:, prefix:]).all(dim=1).sum().item()
    return hits / len(usable)
    # END SOLUTION
