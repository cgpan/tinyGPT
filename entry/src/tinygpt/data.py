"""The corpus: load it, split it, and cut it into training batches."""

from importlib import resources

import torch

from tinygpt.tokenizer import CharTokenizer


def load_text():
    """All of tinyshakespeare as one string (1,115,394 characters)."""
    return resources.files("tinygpt").joinpath("shakespeare.txt").read_text(encoding="utf-8")


def load_data(val_fraction=0.1):
    """Tokenize the corpus and split it into training and validation parts.

    Returns (tokenizer, train_ids, val_ids), where the ids are 1-D tensors of
    integers. The first 90% of the text is for training. The last 10% is held
    out, so the validation loss measures how well the model copes with text it
    has never seen.
    """
    text = load_text()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * (1 - val_fraction))
    return tok, ids[:n], ids[n:]


def get_batch(ids, block_size, batch_size, device="cpu"):
    """Cut `batch_size` random windows out of `ids`.

    x is a window of `block_size` characters. y is the same window moved one
    character to the right, so y[b, t] is the character that came right after
    x[b, t]: the answer the model has to predict at that position.
    """
    starts = torch.randint(len(ids) - block_size, (batch_size,))
    x = torch.stack([ids[s : s + block_size] for s in starts])
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)
