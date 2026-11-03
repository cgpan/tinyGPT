"""Project 4: let the model write. Project 5: the KV cache that makes it fast.

The sampler marks everything it writes with the watermark from watermark.py, unless you
turn it off. See Responsible use in the README.
"""

import torch

from scratchgpt.layers import softmax
from scratchgpt.watermark import KEY, mark


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Choose the next token for each row of logits (B, V). Returns ids of shape (B, 1).

    temperature 0: the highest-scoring token (greedy). Otherwise divide the logits by the
    temperature; with top_k, keep only the k highest scores; with top_p, keep the smallest
    set of highest-probability tokens whose probabilities add up to at least top_p (the
    most likely token is always kept). Set the rest to -inf, take the softmax, and draw with
    torch.multinomial. With a generator, draw on the generator's device (a CPU generator
    gives the same text on any machine) and move the result back to the logits' device.
    """
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


class KVCache:
    """Project 5. Keeps every layer's keys and values, so each new token needs only its own.
    Keys and values have shape (B, H, T, head_dim); T grows by one per generated token."""

    def __init__(self, n_layer):
        raise NotImplementedError("write this in project 5: see projects/05_performance.md")

    def length(self, layer=0):
        """How many positions this layer holds (0 when empty)."""
        raise NotImplementedError("write this in project 5: see projects/05_performance.md")

    def update(self, layer, k, v):
        """Add the new positions' keys and values (B, H, T_new, head_dim) to this layer's, and
        return all of them: (keys, values), each (B, H, T_old + T_new, head_dim)."""
        raise NotImplementedError("write this in project 5: see projects/05_performance.md")


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None,
             watermark=True, key=KEY, use_cache=False, generator=None):
    """Continue ids (B, T) by max_new_tokens tokens. Returns ids (B, T + new), on ids' device.

    Each step: run the model on at most the last cfg.context tokens, take the last position's
    logits (in float32), mark them with the watermark (mark(logits, ids[:, -1], key)) unless
    watermark is False, and choose the next token with sample_next. Stop early once every
    row has written eos_id, if given.

    Project 5, use_cache: feed the whole prompt once, then only the newest token, keeping the
    keys and values in a KVCache. When the sequence reaches cfg.context, start a new cache
    from the last cfg.context - 1 tokens. Greedy output must be the same with and without
    the cache.
    """
    raise NotImplementedError("write this in project 4: see projects/04_training.md")


def write(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=None, top_p=0.95,
          watermark=True, seed=None, use_cache=True):
    """Text in, text out: encode the prompt, generate until <|endoftext|> or max_new_tokens,
    and decode what was written after the prompt. Given."""
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eot_id]], device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    out = generate(model, ids, max_new_tokens, temperature, top_k, top_p, tokenizer.eot_id,
                   watermark, use_cache=use_cache, generator=generator)
    new = out[0, ids.shape[1]:].tolist()
    if tokenizer.eot_id in new:
        new = new[: new.index(tokenizer.eot_id)]
    return tokenizer.decode(new)
