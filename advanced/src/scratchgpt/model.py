"""Project 3: the transformer."""

from torch import nn

from scratchgpt.layers import Attention, Embedding, FeedForward, Linear, RMSNorm, cross_entropy, rope_frequencies


class Block(nn.Module):
    """One transformer block, pre-norm: x + attention(norm(x)), then x + feed_forward(norm(x))."""

    def __init__(self, cfg, layer):
        super().__init__()
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, x, cos, sin, cache=None):
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


class GPT(nn.Module):
    """Token embedding -> n_layer blocks -> RMSNorm -> output layer, which gives a score
    (logit) for every token in the vocabulary at every position.

    With tie_embeddings the output layer is the embedding table itself (logits = h E^T),
    which saves vocab_size * d_model parameters. Otherwise it is a separate Linear.
    The rope tables are registered as buffers (persistent=False: not saved, rebuilt here),
    so they move to the GPU with the model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("write this in project 3: see projects/03_model.md")

    def forward(self, ids, targets=None, cache=None, hidden=False):
        """ids: (B, T) token ids. Returns (logits, loss): logits (B, T, vocab_size), and the
        cross-entropy against targets (B, T), or None without targets. With hidden=True,
        return only the final normalised hidden states (B, T, d_model) (project 7).
        Raises ValueError if the positions would go past cfg.context."""
        raise NotImplementedError("write this in project 3: see projects/03_model.md")


def count_parameters(model):
    """The number of numbers the model learns (each shared parameter counted once)."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")


def parameter_formula(cfg):
    """The same number, from the config alone, without building the model. Work it out on
    paper first: the embedding table, then per block two norms, the attention's two Linears
    and the feed-forward's three, then the final norm and, if not tied, the output layer."""
    raise NotImplementedError("write this in project 3: see projects/03_model.md")
