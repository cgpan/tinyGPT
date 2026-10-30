"""Save a trained model to one file, and load it back on any machine."""

from pathlib import Path

import torch

from tinygpt.config import Config, get_device
from tinygpt.model import GPT, BigramLM
from tinygpt.tokenizer import CharTokenizer

MODELS = {"GPT": GPT, "BigramLM": BigramLM}


def save_checkpoint(path, model, tok, cfg, history=None):
    """Everything needed to use the model later: its weights, its config, and its vocabulary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": type(model).__name__,
                "config": cfg.to_dict(),
                "chars": tok.chars,
                "model": model.state_dict(),
                "history": history or []}, path)


def load_checkpoint(path, device=None):
    """Returns (model, tokenizer, history). The model goes to the best device on *this* machine."""
    ck = torch.load(path, map_location="cpu")
    model = MODELS[ck["kind"]](Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device or get_device()).eval()
    return model, CharTokenizer(ck["chars"]), ck["history"]
