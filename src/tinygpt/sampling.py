"""Generation: let a trained model write, one character at a time."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Extend each row of `idx` (a (B, T) tensor of character ids) by `max_new_tokens` characters."""
    was_training = model.training
    model.eval()                                                 # dropout off while writing
    for _ in range(max_new_tokens):
        context = idx[:, -model.cfg.block_size:]                 # the model only sees the last block_size characters
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature                  # scores for the next character only
        if top_k is not None:
            cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))   # keep only the top_k candidates
        probs = F.softmax(logits, dim=-1)                        # scores -> probabilities
        next_id = torch.multinomial(probs, num_samples=1)        # roll the weighted dice
        idx = torch.cat([idx, next_id], dim=1)
    model.train(was_training)
    return idx


def sample(model, tok, prompt="\n", length=500, temperature=1.0, top_k=None, seed=None):
    """Return `prompt` followed by `length` generated characters, as a string."""
    if not prompt:
        raise ValueError("the prompt needs at least one character; try prompt='\\n'")
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, length, temperature, top_k)
    return tok.decode(out[0].tolist())
