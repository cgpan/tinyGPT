"""Project 3: the transformer."""

from torch import nn

from scratchgpt.layers import Attention, Embedding, FeedForward, Linear, RMSNorm, cross_entropy, rope_frequencies


class Block(nn.Module):
    """One transformer block, pre-norm: x + attention(norm(x)), then x + feed_forward(norm(x)).
    forward(x, cos, sin, cache) passes cos, sin and the cache (project 5) on to attention."""

    def __init__(self, cfg, layer):
        super().__init__()
        # BEGIN SOLUTION 3
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
        # END SOLUTION

    def forward(self, x, cos, sin, cache=None):
        # BEGIN SOLUTION 3
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.ff(self.norm2(x))
        # END SOLUTION


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
        # BEGIN SOLUTION 3
        self.embed = Embedding(cfg.vocab_size, cfg.d_model, cfg.init_std)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = None if cfg.tie_embeddings else Linear(cfg.d_model, cfg.vocab_size, cfg.init_std)
        cos, sin = rope_frequencies(cfg.head_dim, cfg.context, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # END SOLUTION

    def forward(self, ids, targets=None, cache=None, hidden=False):
        """ids: (B, T) token ids. Returns (logits, loss): logits (B, T, vocab_size), and the
        cross-entropy against targets (B, T), or None without targets. With hidden=True,
        return only the final normalised hidden states (B, T, d_model) (project 7).
        cache (project 5): a KVCache passed to every block; the new positions then start at
        cache.length(0). Raises ValueError if the positions would go past cfg.context."""
        # BEGIN SOLUTION 3
        start = 0 if cache is None else cache.length(0)
        if start + ids.shape[1] > self.cfg.context:
            raise ValueError(f"{start + ids.shape[1]} positions, but the context is {self.cfg.context}")
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, cache)
        h = self.norm(x)
        if hidden:
            return h
        logits = h @ self.embed.weight.T if self.head is None else self.head(h)
        loss = None
        if targets is not None:
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
        # END SOLUTION


def count_parameters(model):
    """The number of numbers the model learns (each shared parameter counted once)."""
    # BEGIN SOLUTION 3
    return sum(p.numel() for p in model.parameters())
    # END SOLUTION


def parameter_formula(cfg):
    """The same number, from the config alone, without building the model. Work it out on
    paper first: the embedding table, then per block two norms, the attention's two Linears
    and the feed-forward's three, then the final norm and, if not tied, the output layer."""
    # BEGIN SOLUTION 3
    d, f = cfg.d_model, cfg.ff_dim
    per_block = 2 * d + 4 * d * d + 3 * d * f
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    return cfg.vocab_size * d + cfg.n_layer * per_block + d + head
    # END SOLUTION
