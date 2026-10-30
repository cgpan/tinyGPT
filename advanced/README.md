# tinyGPT · Advanced track: a language model from scratch

Write every piece of a small language model yourself: the data pipeline, the tokenizer,
the transformer, the training infrastructure, the evaluation, and the safeguards that go
with a release. Then train it on 100,000 IMDB movie reviews and find out what it learned.
This is the work of the people who build language models at AI companies, at a scale that
fits one GPU.

> **Status:** designed and tested; not yet run on a GPU. The reference solution passes the
> whole test suite, and the pipeline has been run end to end on a CPU with stand-in data
> (the real download was blocked where this was written). Still to do: a full run on
> IMDB with `scripts/verify.py` on an NVIDIA GPU, a Mac and Colab, and the measured numbers
> that come from it.

Please read [Responsible use](../README.md#responsible-use) first. In short: this is for
learning. Never use it to write fake reviews or comments, to pose as real people, or to
spread misinformation.

## The projects

| | project | you write | you'll be able to say |
|---|---|---|---|
| 1 | [The data](projects/01_data.md) | cleaning, scrubbing personal details, MinHash deduplication, decontamination | what's in my training data, and why I trust my test set |
| 2 | [The tokenizer](projects/02_tokenizer.md) | byte-level BPE: training, encoding, decoding | how text becomes tokens, and why per-token losses don't compare |
| 3 | [The model](projects/03_model.md) | every layer: RMSNorm, rotary positions, attention, SwiGLU, the GPT | every number in a modern decoder |
| 4 | [Training](projects/04_training.md) | AdamW, the schedule, clipping, accumulation, mixed precision, checkpoints, sampling | my run survives being cut off, and resumes exactly |
| 5 | [Performance](projects/05_performance.md) | FLOP and memory accounting, MFU, the KV cache | how long a run takes and where the time goes |
| 6 | [Scaling, and the final run](projects/06_scaling.md) | fitting a scaling law, planning a run from it | how big my model should be for my budget, and why |
| 7 | [Evaluation](projects/07_evaluation.md) | bits per byte, baselines, a sentiment probe, a memorization test | how good my model is, in numbers that can't fool me |
| 8 | [Responsible release](projects/08_release.md) | a watermark detector, the model card | what my model must not be used for, and how its text can be told apart |

Each brief follows the entry track's habit: **Predict** before you run anything, **Build**
against the tests, **Run** the project's script, **Investigate** what came out, and then a
**Stretch** goal or two.

## The rules

You write it from scratch, with PyTorch's tensors, autograd and GPU support. You may use
`torch.nn.Module`, `nn.Parameter`, `nn.ModuleList` and `nn.init`, NumPy, and Python's
standard library. You may not use PyTorch's ready-made layers, losses or optimizers
(`torch.nn.Linear`, `torch.nn.functional`, `torch.softmax`, `torch.optim`, ...), or anyone
else's tokenizer or model (`tiktoken`, `tokenizers`, `transformers`). `tests/test_rules.py`
reads your code and says where a rule is broken. One exception, in project 5: PyTorch's
fused attention, to compare yours with.

## How it works

**The tests are the specification.** Every function you write has a docstring that says
exactly what it must do, and tests that check it, often against PyTorch's own version.
Until you write a function it raises `NotImplementedError` and names its project:

```bash
python -m pytest tests/test_02_tokenizer.py            # one project
python -m pytest tests/test_02_tokenizer.py -k merge   # one function's tests
```

**Solutions** are in [`solutions/`](solutions/): the complete reference package, and a page
of notes per project on the ideas and the usual mistakes. Use them after you have tried.

## What you need

- **An NVIDIA GPU with 12 GB or more**, **a Mac with Apple Silicon and 24 GB or more**, or
  **Google Colab**. Paid Colab is best for the final run in project 6: its sessions last
  longer and its GPUs are faster.
- PyTorch with GPU support. On Windows, install the CUDA build from
  [pytorch.org](https://pytorch.org/get-started/locally/); the default is CPU-only.
- About 2 GB of disk for the data and checkpoints.

### On your own computer

```bash
git clone https://github.com/cgpan/tinyGPT.git
cd tinyGPT/advanced
python -m pip install -e .
python -m pytest tests/test_rules.py
```

Then open [projects/01_data.md](projects/01_data.md). Run everything from the `advanced`
folder.

### In Colab

Open the workspace notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/advanced/colab.ipynb).
It keeps your code and checkpoints in Google Drive, so your work survives the end of a
session, and has one cell per command. You edit the `.py` files in Colab's file browser.

## How long training takes

The compute is exact (project 5 shows how it's counted). The times are **rough estimates**,
assuming the GPU sustains about 20 TFLOPS for a fast gaming card and about 10 for Colab's
T4. Once you have done project 5, `python scripts/benchmark.py` measures yours in a minute.

| config | parameters | tokens | compute | RTX 4070-class | Colab T4 |
|---|---|---|---|---|---|
| `tiny` | 0.6M | 0.1M | a smoke test | seconds | seconds |
| `small` | 5.3M | 24.6M | 8.6 × 10¹⁴ FLOPs | 1–2 min | 2–3 min |
| `base` | 29.9M | 164M | 3.4 × 10¹⁶ FLOPs | about 30 min | about 1 hour |
| your final model (project 6) | you decide | | | up to 24 hours | |

Measured so far, on a 4-core CPU only: `small` runs at 3,800 tokens per second, which
would take 107 minutes. A GPU should be tens of times faster.

`base` needs about 4.6 GB of GPU memory with a batch of 16 by project 5's rough estimate,
well under 12 GB; the real number is part of project 5.

## The data

The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et
al., 2011): 50,000 reviews labelled positive or negative, half for training and half for
testing, and 50,000 unlabelled ones; 133 MB of text. The training and test reviews are
about different films. Project 1 downloads it; it isn't part of this repository, and it
remains under its authors' terms.

## What's in this folder

```
projects/             one brief per project: start here
src/scratchgpt/       your package: the files you write in
    data.py           project 1        sampling.py   projects 4, 5
    tokenizer.py      project 2        perf.py       project 5
    layers.py         project 3        scaling.py    project 6
    model.py          project 3        evaluate.py   project 7
    optim.py          project 4        watermark.py  project 8
    train.py          project 4
    imdb.py, config.py, device.py      given: nothing to write
tests/                the specification, one file per project, plus the rules
configs/              tiny.yaml, small.yaml, base.yaml
scripts/              one or two per project: prepare_data, train_tokenizer, inspect_model,
                      train, sample, benchmark, sweep, evaluate, watermark_check, model_card;
                      verify.py and make_stubs.py are for instructors
solutions/            the reference package, and notes per project
colab.ipynb           the Colab workspace
```

Teaching with it? See [INSTRUCTOR.md](INSTRUCTOR.md).
