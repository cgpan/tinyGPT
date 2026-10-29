"""Comparing tinyGPT with GPT-2, fairly (chapter 8).

The functions here take any model and its own tokenizer: tinyGPT with its characters,
or GPT-2 with its word pieces.
"""

import importlib
import subprocess
import sys
import warnings

import torch
from torch.nn import functional as F

from tinygpt.config import get_device

GPT2 = "openai-community/gpt2"          # GPT-2 small on Hugging Face: 124M parameters


def load_gpt2(device=None):
    """Download GPT-2 small (about 500 MB, the first time only). Returns (model, tokenizer)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("Installing the transformers library first...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)
        importlib.invalidate_caches()
    with warnings.catch_warnings():                  # notes meant for library developers, not for us
        warnings.simplefilter("ignore")
        from huggingface_hub.utils import logging as hub_logging
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
    hub_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(GPT2)
    tokenizer.model_max_length = 10**9  # we cut long texts into windows ourselves
    model = AutoModelForCausalLM.from_pretrained(GPT2).to(device or get_device()).eval()
    return model, tokenizer


def _context_length(model):
    return model.cfg.block_size if hasattr(model, "cfg") else model.config.n_positions


def _logits(model, x):
    out = model(x)
    return out[0] if hasattr(model, "cfg") else out.logits    # tinyGPT returns (logits, loss)


@torch.no_grad()
def token_surprises(text, model, tokenizer):
    """Read `text` in the model's own tokens. Returns the tokens it had to predict, and
    its surprise (-log p, in nats) at each one."""
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(text)
    window = _context_length(model)                          # how far back the model can read
    device = next(model.parameters()).device
    tokens, surprises = [], []
    for start in range(0, len(ids) - 1, window):
        chunk = torch.tensor([ids[start : start + window + 1]], device=device)
        x, y = chunk[:, :-1], chunk[:, 1:]                   # every token predicts the next one
        logits = _logits(model, x)
        surprises += F.cross_entropy(logits[0], y[0], reduction="none").tolist()
        tokens += [tokenizer.decode([t]) for t in y[0].tolist()]
    model.train(was_training)
    return tokens, surprises


def per_token_and_per_character(text, model, tokenizer):
    """The same total surprise, averaged two ways: per token, and per character."""
    tokens, surprises = token_surprises(text, model, tokenizer)
    return sum(surprises) / len(tokens), sum(surprises) / sum(len(t) for t in tokens)


def gpt2_sample(model, tokenizer, prompt="\n", length=100, temperature=1.0, seed=None):
    """Let GPT-2 continue `prompt` for `length` tokens."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, do_sample=True, max_new_tokens=length, temperature=temperature, top_k=0,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0])
