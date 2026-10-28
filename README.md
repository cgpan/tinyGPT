# tinyGPT

Build and train a small GPT yourself: a character-level transformer that learns to
write Shakespeare-like text from one 1 MB file. No `transformers` library. Every line
of the model, including the attention code, is in this repository and meant to be read.

> **Status:** Milestone 1 of 4. The Python package, its tests, and the measured
> numbers below are done. The chapter notebooks (the handbook itself) come next.

## The number

One number runs through the whole project: the model's **loss** on text it has never
seen. Loss is the average surprise at the real next character, `-log p`, measured in
nats. Lower is better.

| model | validation loss | perplexity | what it means |
|---|---|---|---|
| knowing nothing | **4.17** | 65 | `ln 65`: every one of the 65 characters equally likely |
| counting bigram | **2.48** | 12.0 | count which character follows which; ~30 lines of NumPy |
| tinyGPT `small` (0.8M parameters) | **1.69** | 5.4 | 2,000 training steps |
| tinyGPT `base` (10.7M parameters) | **TBD** | TBD | 3,000 training steps |

Every number in this table was measured by running the code in this repository
(see [How these numbers were measured](#how-these-numbers-were-measured)).

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
cd tinyGPT
python -m pip install -e ".[dev]"
python -m pytest                             # 41 tests, a few seconds
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

| config | parameters | steps | this CPU (measured) | Colab T4 (estimate) | NVIDIA RTX 30/40 (estimate) | Apple M1/M2 (estimate) | Apple M Pro/Max (estimate) |
|---|---|---|---|---|---|---|---|
| `tiny` | 0.11M | 200 | 6 s | under 1 min | under 1 min | under 1 min | under 1 min |
| `small` | 0.82M | 2,000 | 2.3 min | about 1 min | under 1 min | 1–2 min | about 1 min |
| `base` | 10.7M | 3,000 | TBD | 6–9 min | 2–4 min | hits the 9-min limit around step 1,200–1,800 | 4–9 min |

The estimates scale this CPU's measured speed by each chip's typical throughput. They
are rough. `python scripts/benchmark.py` times 10 real steps on your machine and
prints a projection in about a minute.

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
    shakespeare.txt   the corpus (1,115,394 characters)
configs/              tiny.yaml, small.yaml, base.yaml
scripts/              train.py, sample.py, benchmark.py
tests/                pytest suite; runs in seconds on a CPU
```

## How these numbers were measured

On a 4-core Intel Xeon CPU (no GPU), PyTorch 2.15, Python 3.13, seed 1337, using the
configs in `configs/` unchanged. The training and validation split is the first 90%
and last 10% of the corpus (1,003,854 and 111,540 characters).

- Counting bigram (smoothing 1): training 2.4546, validation 2.4819, perplexity 11.96.
  Without smoothing the validation loss is infinite: 187 character pairs in the
  validation text never occur in the training text.
- `small`: validation 1.6945 at step 2,000 (training 1.5026).
- `base`: TBD.

Loss values do not depend on the hardware. Training times do.

## Credits

The model follows the decoder-only transformer as taught in Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture). The corpus is
`tinyshakespeare` from his [char-rnn](https://github.com/karpathy/char-rnn) repository.
