"""scratchgpt: a language model you write yourself.

Each module belongs to one project (see projects/ in the advanced folder):

    data.py        project 1   clean, deduplicate and split the reviews
    tokenizer.py   project 2   byte-level BPE
    layers.py      project 3   the transformer's parts
    model.py       project 3   the transformer
    optim.py       project 4   AdamW, the learning-rate schedule, gradient clipping
    train.py       project 4   batches, the training loop, checkpoints
    sampling.py    projects 4-5 writing text, and the KV cache
    perf.py        project 5   compute, memory and speed
    scaling.py     project 6   scaling laws, and planning the final run
    evaluate.py    project 7   bits per byte, sentiment, memorization
    watermark.py   project 8   marking and detecting generated text

Given, nothing to write: imdb.py (download and read the reviews), config.py, device.py.
"""

from scratchgpt.config import Config, load_config

__all__ = ["Config", "load_config"]
