# tinyGPT · Advanced track: a language model from scratch

Write every piece of a small language model yourself: the data pipeline, the tokenizer,
the transformer, the training infrastructure, the evaluation, and the safeguards that go
with a release. Then train it on 100,000 IMDB movie reviews and find out what it learned.
This is the work of the people who build language models at AI companies, at a scale that
fits one GPU.

> **Tested** end to end, on all of IMDB with the reference solution, on an NVIDIA
> RTX 4070 SUPER (Windows, bfloat16) and an Apple M5 Max (float32). The numbers below come
> from those runs unless they are marked as estimates. Colab hasn't been tested yet: its
> T4 trains in float16, which has only been checked on a CPU. If something doesn't work
> for you, please [open an issue](https://github.com/cgpan/tinyGPT/issues).

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

## What your model will write

The number, in bits per byte of held-out reviews (lower is better; project 7 explains it):

| | bits per byte | |
|---|---|---|
| knowing nothing | 8 | every byte value equally likely |
| gzip | 3.25 | the compression program, as a baseline |
| token frequencies | 2.52 | each token's share of the training text, ignoring context |
| `small` (5.3M parameters) | 1.53 | after 3,000 steps (validation reviews) |
| `base` (29.9M parameters) | 1.34 | after 5,000 steps (test reviews: 1.335 and 1.341 on the two machines) |

`base` writes reviews like this one (its first sample on the Mac, watermarked, not a real review):

> This movie has many of the most impressive effects ever to come across on film. The acting
> is superb, the special effects are great, and there is a lot of atmosphere in it. The
> movie's plot has been told many times and too often is given a very strong central idea.

Its sentences are grammatical and about films, but a review drifts and contradicts itself
("the acting was fantastic ... You will be disappointed by the acting"), repeats stock
phrases, and invents names. It learned from unfiltered reviews: in our runs it wrote
profanity and, once, an ableist slur. That belongs in your model card (project 8).

## How long training takes

Measured with the reference solution. Your own code may be faster or slower; once you have
done project 5, `python scripts/benchmark.py` measures it in a minute.

| config | parameters | tokens | compute | RTX 4070 SUPER | Apple M5 Max | Colab T4 |
|---|---|---|---|---|---|---|
| `tiny` | 0.6M | 0.1M | a smoke test | seconds | seconds | seconds |
| `small` | 5.3M | 24.6M | 8.6 × 10¹⁴ FLOPs | 2.5 min | 4 min | about 5 min |
| `base` | 29.9M | 164M | 3.4 × 10¹⁶ FLOPs | 77 min | 115 min | about 2–3 hours |
| your final model (project 6) | you decide | | | | | up to 24 hours |
| | | | | measured | measured | estimate |

The 4070 SUPER trained in bfloat16 at 35,000 tokens per second for `base`, 10% of its
peak; the M5 Max in float32 at 23,000. The T4 column is scaled from those by its memory
speed and isn't measured. Paid Colab's L4 and A100 are faster GPUs; we haven't timed them either.

`base` with a batch of 16 used 6.6 GB of GPU memory at its peak on the 4070 SUPER, and the
Apple GPU held 9.8 GB in float32: it fits a 12 GB card and a 24 GB Mac.

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

Teaching with it? See [INSTRUCTOR.md](INSTRUCTOR.md). The papers behind each project, and
everything else this track builds on, are listed in [CREDITS.md](../CREDITS.md).
