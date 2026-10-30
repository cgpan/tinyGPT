# tinyGPT · Entry track

Build and train a small GPT yourself: a character-level transformer that learns to
write Shakespeare-like text from one 1 MB file. No `transformers` library. Every line
of the model, including the attention code, is in this repository and meant to be read.

> **Status:** in progress. The package, its tests, the measured numbers below, and
> all eight chapters are written, and everything has been run on Windows (NVIDIA GPU),
> macOS (Apple Silicon) and Linux. Still to do: a check on Colab's T4 GPU.
>
> This is the entry track. When you finish it, the [advanced track](../advanced/README.md)
> has you write every piece of a language model yourself. Please read
> [Responsible use](../README.md#responsible-use) before you share anything a model wrote.

```
ROMEO:
Call'd here! me ungly that means a bloody own.

DUKE OF YORK:

DORSET:
Clifford, all gainst he shall return us our desert.
```
<sub>The first sample from the `base` model trained for 3,000 steps (validation loss 1.50), at temperature 1.0.</sub>

## The number

One number runs through the whole project: the model's **loss** on text it has never
seen. Loss is the average surprise at the real next character, `-log p`, measured in
nats. Lower is better.

| model | validation loss | perplexity | what it means |
|---|---|---|---|
| knowing nothing | **4.17** | 65 | `ln 65`: every one of the 65 characters equally likely |
| counting bigram | **2.48** | 12.0 | count which character follows which; ~30 lines of NumPy |
| tinyGPT `small` (0.8M parameters) | **1.69** | 5.4 | 2,000 training steps |
| tinyGPT `base` (10.7M parameters) | **1.49** | 4.4 | 2,500 training steps |
| GPT-2 small (124M parameters), for comparison | 1.29 | 3.6 | trained on about 40 GB of web text; per character (chapter 8) |

Every number in this table was measured by running the code in this repository
(see [How these numbers were measured](#how-these-numbers-were-measured)).

## Chapters

Work through them in order. Each chapter opens one part of the model, and asks how that
part moves the number.

| | chapter | open box | the number | GPU |
|---|---|---|---|---|
| 1 | [Text is numbers](notebooks/01_text_is_numbers.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/01_text_is_numbers.ipynb) | the tokenizer | 4.17 → 3.35 | no |
| 2 | [Counting beats guessing](notebooks/02_counting.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/02_counting.ipynb) | the counting bigram | 3.35 → 2.48 | no |
| 3 | [Learning instead of counting](notebooks/03_learning.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/03_learning.ipynb) | one training step | 4.17 → 2.48, by learning | no |
| 4 | [Looking back: attention](notebooks/04_attention.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/04_attention.ipynb) | one attention head | 2.48 → 1.76 | recommended |
| 5 | [The full run](notebooks/05_full_run.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/05_full_run.ipynb) | one transformer block | 1.76 → 1.49 | yes |
| 6 | [Talking to your model](notebooks/06_talking.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/06_talking.ipynb) | the writing loop | 1.49, and what temperature does to it | only to retrain |
| 7 | [How do you know it works?](notebooks/07_testing.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/07_testing.ipynb) | the model's tests | 4.17, as a test | no |
| 8 | [Meet a real one](notebooks/08_meet_gpt2.ipynb) (optional) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgpan/tinyGPT/blob/main/entry/notebooks/08_meet_gpt2.ipynb) | how a model reads a text | tinyGPT against GPT-2, per character | recommended |

Every chapter follows the same five steps: **Predict** what will happen, **Run** it,
**Investigate** the code that ran, **Modify** one thing and explain the result, and,
as an optional challenge, **Make** that chapter's piece yourself.

**In Colab:** click a badge and run the first cell. It downloads and installs everything.
**On your own computer:** clone the repository (see [Quick start](#quick-start)), open
a notebook from the `notebooks/` folder in Jupyter or VS Code, and run the first cell.

Teaching with it? See [INSTRUCTOR.md](INSTRUCTOR.md).

## What you need

A GPU. Any one of these works:

- **Google Colab** (free tier, T4 GPU)
- **A computer with an NVIDIA GPU** (for example a gaming laptop or desktop)
- **A Mac with Apple Silicon** (M1 or later; PyTorch uses its GPU through "MPS")

Check that PyTorch can see your GPU. This should print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"
```

If it prints `False` on a machine with an NVIDIA GPU, you have the CPU-only build of
PyTorch (the default on Windows). Install the CUDA build using the selector on
[pytorch.org](https://pytorch.org/get-started/locally/).

## Quick start

From a terminal, in the environment you want to use:

```bash
git clone https://github.com/cgpan/tinyGPT.git
cd tinyGPT/entry
python -m pip install -e .
python -m pytest                             # the tests: a few seconds
python scripts/train.py configs/tiny.yaml    # whole pipeline in seconds
python scripts/benchmark.py                  # how long each config will take on this machine
python scripts/train.py configs/base.yaml    # the full model
python scripts/sample.py runs/base/model.pt --prompt "ROMEO:"
```

Change any setting without editing a file: `python scripts/train.py configs/small.yaml n_layer=2 causal=false`.

Or use it from Python:

```python
import tinygpt

cfg = tinygpt.load_config("configs/small.yaml")
model, history = tinygpt.train(cfg)
tok, _, _ = tinygpt.load_data()
print(tinygpt.sample(model, tok, prompt="ROMEO:", length=300))
```

## How long training takes

Every run stops by itself after `max_minutes` (9 minutes), so no run goes past
10 minutes on any machine. On slower hardware the run stops early with a higher loss.

| config | parameters | steps | NVIDIA RTX 4070 SUPER | Apple M5 Max | 4-core CPU | Colab T4 | Apple M1/M2 |
|---|---|---|---|---|---|---|---|
| `tiny` | 0.11M | 200 | under 1 min | under 1 min | 6 s | under 1 min | under 1 min |
| `small` | 0.82M | 2,000 | 0.6 min | 0.6 min | 2.3 min | about 1 min | 1–2 min |
| `base` | 10.7M | 2,500 | **1.9 min** | **4.1 min** | about 1 h 50 min | 6–8 min | stops at the 9-min limit, around step 900–1,500 |
| | | | measured | measured | measured | estimate | estimate |

The first three columns were measured with `scripts/verify.py` (Windows 11, PyTorch
2.15 with CUDA; macOS with PyTorch's MPS backend; Linux, no GPU). The last two are
estimates, scaled by each chip's typical speed. `python scripts/benchmark.py` times 10
real steps on your machine and prints a projection in about a minute.

A `base` run that stops early still gets most of the way: validation loss is already
1.56 at step 1,500 and 1.51 at step 1,750 (see the curve below). On a base M1/M2/M3
Mac, expect roughly 1.55–1.75; use Colab instead if you want the full 1.49.

## What's in the repository

```
src/tinygpt/          the package: everything you import
    tokenizer.py      text <-> integers
    data.py           load the corpus, split it, cut it into batches
    bigram.py         the counting baseline (NumPy only)
    model.py          the learned bigram and the transformer
    training.py       train_step (one step of learning) and train (the loop around it)
    sampling.py       generate text one character at a time
    config.py         all the settings in one place
    checkpoint.py     save a trained model; load it on any machine
    notebook.py       show() displays a chapter's open box; print_setup()
    plots.py          the scoreboard and other charts
    check.py          checks for the Make challenges
    shakespeare.txt   the corpus (1,115,394 characters)
notebooks/            the chapters
solutions/            worked solutions to each chapter's Make challenge
configs/              tiny.yaml, small.yaml, base.yaml
scripts/              train.py, sample.py, benchmark.py, verify.py (check a machine end to end)
tests/                pytest suite; runs in seconds on a CPU
INSTRUCTOR.md         notes for teaching with it
```

## How these numbers were measured

On a 4-core Intel Xeon CPU (no GPU), PyTorch 2.15, Python 3.13, seed 1337, using the
configs in `configs/` unchanged. The training and validation split is the first 90%
and last 10% of the corpus (1,003,854 and 111,540 characters).

- Counting bigram (smoothing 1): training 2.4546, validation 2.4819, perplexity 11.96.
  Without smoothing the validation loss is infinite: 187 character pairs in the
  validation text never occur in the training text.
- `small`: validation 1.6945 at step 2,000 (training 1.5026).
- `base`: trained for 3,000 steps to see where the curve flattens. The learning rate
  is constant, so the first 2,500 steps of that run are exactly the 2,500-step run
  that `configs/base.yaml` now specifies.

  | step | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | **2500** | 2750 | 3000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | train | 4.34 | 2.40 | 1.93 | 1.64 | 1.48 | 1.39 | 1.34 | 1.30 | 1.26 | 1.24 | **1.21** | 1.19 | 1.18 |
  | validation | 4.33 | 2.43 | 2.01 | 1.79 | 1.67 | 1.59 | 1.56 | 1.51 | 1.52 | 1.52 | **1.49** | 1.49 | 1.50 |

  After about step 1,750 validation loss improves only slowly (1.51 to 1.49) while
  training loss keeps falling: the model has started to memorise its training text.
  Measurements vary by about ±0.015 between nearby steps because each uses 20 random
  batches.

  The untrained model starts at 4.33, not 4.17: its random starting weights make
  random, slightly confident guesses, which score worse than an even spread. Smaller
  models start closer (`tiny` 4.19, `small` 4.20).

Loss values come out essentially the same on any hardware (small floating-point
differences aside). Training times do not.

## Credits

The model follows the decoder-only transformer as taught in Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture). The corpus is
`tinyshakespeare` from his [char-rnn](https://github.com/karpathy/char-rnn) repository.
