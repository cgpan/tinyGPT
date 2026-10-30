"""Every setting in one place. Given: nothing to write here."""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # the model (project 3)
    vocab_size: int = 8192
    context: int = 512                # how many tokens the model can read at once
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: int | None = None           # None: about 8/3 of d_model, rounded up to a multiple of 64
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_embeddings: bool = True       # the output layer reuses the embedding table
    init_std: float = 0.02
    fused_attention: bool = False     # project 5: PyTorch's fused attention instead of yours

    # training (project 4)
    batch_size: int = 16              # sequences per forward pass
    grad_accum: int = 4               # forward passes per optimizer step
    max_steps: int = 6000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_steps: int = 20
    checkpoint_interval: int = 250
    max_hours: float = 24.0           # stop (and save) after this long, wherever the run is
    dtype: str = "auto"               # auto, float32, bfloat16 or float16
    compile: bool = False             # project 5: torch.compile
    seed: int = 1337
    device: str = "auto"              # auto, cuda, mps or cpu

    # where things live
    data_dir: str = "data/imdb"
    out_dir: str = "runs/base"

    @property
    def ff_dim(self):
        return self.d_ff or 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    @property
    def tokens_per_step(self):
        return self.batch_size * self.grad_accum * self.context

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.head_dim % 2:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for rotary embeddings")


def _coerce(name, value):
    """Turn a command-line string into the field's type: 'lr=3e-4' gives 0.0003."""
    if not isinstance(value, str):
        return value
    parsed = yaml.safe_load(value)
    if name in ("lr", "min_lr", "dropout", "weight_decay", "max_hours", "rope_theta",
                "init_std", "beta1", "beta2", "grad_clip") and isinstance(parsed, int):
        parsed = float(parsed)
    return parsed


def load_config(path=None, **overrides):
    """Read a YAML config, then apply overrides (keyword arguments or 'key=value' strings)."""
    values = {}
    if path is not None:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = {f.name for f in dataclasses.fields(Config)}
    for name, value in {**values, **overrides}.items():
        if name not in fields:
            raise ValueError(f"unknown setting {name!r}; the settings are: {', '.join(sorted(fields))}")
        values[name] = _coerce(name, value)
    return Config(**values)


def parse_overrides(pairs):
    """['n_layer=4', 'lr=3e-4'] -> {'n_layer': '4', 'lr': '3e-4'}"""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out
