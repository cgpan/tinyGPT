"""All the knobs in one place."""

from dataclasses import asdict, dataclass

import torch
import yaml


@dataclass
class Config:
    # the model
    vocab_size: int = 65         # how many distinct characters there are
    block_size: int = 128        # how many characters the model can look back at
    n_layer: int = 6             # how many transformer blocks are stacked
    n_head: int = 6              # attention heads in each block
    n_embd: int = 384            # length of the vector that represents each character
    dropout: float = 0.2         # fraction of activations randomly zeroed while training
    causal: bool = True          # False lets attention see future characters (a deliberate bug)
    pos_emb: bool = True         # False takes away the model's sense of position
    # training
    batch_size: int = 64         # windows of text per training step
    learning_rate: float = 0.001
    max_iters: int = 2500        # training steps
    eval_interval: int = 250     # measure the loss every this many steps
    eval_iters: int = 20         # batches averaged for each measurement
    max_minutes: float = 9.0     # stop early if training runs longer than this
    seed: int = 1337

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    def to_dict(self):
        return asdict(self)


def load_config(path, **overrides):
    """Read a YAML file into a Config. Keyword arguments replace values from the file.

    >>> cfg = load_config("configs/small.yaml", n_layer=1)
    """
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    values.update(overrides)
    return Config(**values)


def get_device():
    """The fastest hardware available: an NVIDIA GPU, an Apple Silicon GPU, or the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
