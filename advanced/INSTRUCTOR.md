# Instructor note: the advanced track

For students who want to build language models for a living. They write every piece
against a test suite, in eight projects; expect several weeks of work. This page is for
you.

## Before you share it

- [x] **`scripts/verify.py` on a GPU machine and a Mac.** Done on an NVIDIA RTX 4070 SUPER
      (Windows 11, PyTorch 2.11, bfloat16) and an Apple M5 Max (macOS, PyTorch 2.15, float32):
      every step OK. The numbers are at the end of this page.
- [x] **Is IMDB enough?** Yes; see below.
- [ ] **Colab** (not done for the first release). Run the workspace notebook
      (`colab.ipynb`) on a T4 up to project 4, or `python scripts/verify.py --quick` in a
      Colab cell. Its T4 trains in float16 with the gradient scaler: tested on a CPU, but
      not yet on a real GPU. The README's T4 times are estimates.
- [x] **Make the repository public**: the Colab workspace clones it.

```bash
cd advanced
python -m pip install -e .
python scripts/verify.py --quick     # a minute or two: the whole pipeline on a 2 MB slice
python scripts/verify.py             # 1.5 to 2 hours: small and base on all of IMDB
python scripts/verify.py --sweep     # also the scaling sweep (17 minutes on the 4070 SUPER)
```

## How it is built

- **`solutions/scratchgpt/` is the source of truth.** The parts students write sit between
  `# BEGIN SOLUTION n` and `# END SOLUTION` markers. `scripts/make_stubs.py` writes
  `src/scratchgpt/` from it, replacing each marked block with a `NotImplementedError` that
  names project n. Never edit `src/` by hand.
- **The tests are the specification.** They use PyTorch's own functions as the answer key
  where one exists (`torch.optim.AdamW`, `F.scaled_dot_product_attention`,
  `clip_grad_norm_`, ...), and test properties elsewhere (a resumed run must match an
  uninterrupted one exactly; a cached generation must match an uncached one).
- **`tests/test_rules.py`** reads the students' code and rejects ready-made layers, losses,
  optimizers and tokenizers. It reads the code, so a determined student can get past it;
  it's there to keep honest students honest.
- **`tests/test_stubs.py`** (maintainers only) checks that `src/` matches the solution.

To change the reference solution:

```bash
# edit solutions/scratchgpt/..., then
python scripts/make_stubs.py
PYTHONPATH=solutions python -m pytest      # PowerShell: $env:PYTHONPATH="solutions"; python -m pytest
python -m pytest                           # with the stubs: most tests fail, by design
```

With the solution, all 166 tests pass. With the stubs, 33 pass (the rules, the layout, and
tests of the given code), 2 are skipped (the maintainers' stub check) and the rest fail or
error with "write this in project n".

## Is IMDB enough?

Yes. You asked whether IMDB can teach the model to write meaningful text rather than
nonsense. After 77 minutes on the 4070 SUPER, `base` writes grammatical sentences about
films, such as:

> This movie has many of the most impressive effects ever to come across on film. The
> acting is superb, the special effects are great, and there is a lot of atmosphere in it.

Its limits are those of a 30M-parameter model trained on 24 million tokens: a review drifts
and contradicts itself, repeats stock phrases ("The acting was ..."), and invents names.
`small` (5.3M parameters, 2.5 minutes) already writes review-like sentences, with more
nonsense words. The IMDB training text is 24.2 million tokens, so the data is small for
the 24-hour budget: `base` makes 6.8 passes over it, its validation loss was still
improving slowly at the last step (3.724 at step 4,500, 3.715 at 5,000), and training loss
(3.43) is already below validation loss (3.72). A bigger model or a longer run gains little
from the same data. Project 6's data-wall experiment and its "more data" stretch are where
students meet that.

**Content.** The model learned from unfiltered reviews. In the test runs it wrote profanity
and, once, an ableist slur. Project 8's model card asks students to look for this.

## What has and hasn't been run

Run, with the reference solution:

- the whole test suite on Linux, Windows and macOS (165 tests then; 166 now);
- `verify.py` in full on the RTX 4070 SUPER (bfloat16, with the scaling sweep) and the
  M5 Max (float32): the IMDB download, data preparation, the tokenizer, `small`, `base`,
  samples, evaluation, the watermark check and the model card. The two machines give the
  same data counts and tokenizer, and nearly the same losses;
- float16 training with the gradient scaler, on a CPU only: the path a T4 takes;
- planted bugs: removing either random-state restore, the gradient-accumulation division,
  or AdamW's weight decay, or clipping before unscaling, each fails exactly the test meant
  to catch it;
- an independent review of the briefs against the code, whose findings are fixed.

Not run yet:

- Colab, and so a T4 in float16 on a real GPU;
- `fused_attention=true` and `compile=true` on a GPU (project 5's comparison);
- a student's own implementation from the stubs, end to end. The tests stand in for it.

## Notes for each project

1. **Data.** The pipeline takes about a minute. The near-duplicate threshold (0.8) and the
   phone pattern (North American only) are choices; good students question them.
2. **Tokenizer.** The simple BPE is far too slow for all of IMDB, so the brief has
   students train on a 5 MB sample first. The fast version is the stretch goal; the
   reference learns all 7,935 merges from IMDB in 9 to 13 seconds. Training the tokenizer
   on a sample changes the merges, and so every later number, a little.
3. **Model.** The most tests. The tied-embedding surprise (an untrained model scores below
   ln V when targets equal inputs) is a good discussion: in `base`, 7.22 against 9.12.
4. **Training.** The resume test is strict on purpose: it catches forgotten random states.
   Colab users must put `out_dir` on Drive; the workspace notebook does.
5. **Performance.** MFU is low, 8% to 10% on the 4070 SUPER, because our attention isn't
   fused and keeps float32 scores. The fused-attention and compile flags show how much
   that costs. Ask for the filled-in table.
6. **Scaling.** The sweep takes 17 minutes on the 4070 SUPER, and its prediction for the
   next size was within 0.6%. The final run is the student's own plan; ask to see it
   before they spend the hours.
7. **Evaluation.** The probe's zero-gradient trap is real (it happened while writing the
   reference); the docstring warns about it. The word-count baseline (71%) is weak: 4,000
   labelled reviews for 8,192 features, with the same light probe as the model's 512.
   Tuned word-count classifiers do much better (Maas et al. report the high 80s with all
   25,000 reviews). Don't let students conclude more than the comparison supports; the
   brief's Investigate 3 and the stretch ask them to tune it.
8. **Release.** Insist on a complete model card. It is the point of the project.

## Measured numbers

With the reference solution, on the RTX 4070 SUPER (PC) and the M5 Max (Mac). Where the two
machines agree, one number is given.

| project | measure | value |
|---|---|---|
| 1 | reviews read | 100,000 |
| 1 | exact duplicates removed (training / held-out) | 1,151 (949 / 202) |
| 1 | near-duplicates removed from training | 43 |
| 1 | training reviews removed as copies of test reviews | 389 |
| 1 | emails / web addresses / phone numbers replaced | 95 / 425 / 6 |
| 1 | documents: training / validation / test | 73,619 / 2,480 / 22,318 |
| 1 | text: training / validation / test | 96.4 / 3.2 / 28.4 MB |
| 1 | time | 57 s (PC), 40 s (Mac) |
| 2 | different pieces in the training text | 196,319 |
| 2 | time to learn 7,935 merges (the reference's fast version) | 13 s (PC), 9 s (Mac) |
| 2 | bytes per token at 8,192 tokens | 3.99 (training), 3.98 (validation, test) |
| 2 | training / validation / test tokens | 24,162,373 / 816,211 / 7,119,838 |
| 4 | `small` validation loss at step 3,000 | 4.225 (PC), 4.228 (Mac): 1.53 bits per byte |
| 4 | `small` training time | 2.5 min (PC), 4.0 min (Mac) |
| 4 | `base` validation loss at step 5,000 | 3.715 (PC), 3.735 (Mac): 1.35 bits per byte |
| 4 | `base` training loss at step 5,000 | 3.43 (PC), 3.45 (Mac) |
| 4 | `base` training time | 77 min (PC), 115 min (Mac) |
| 5 | `small` tokens per second | 170,604 (PC, MFU 8%), 111,311 (Mac) |
| 5 | `base` tokens per second | 35,431 (PC, MFU 10%, 7.2 TFLOPS), 23,436 (Mac, 4.8 TFLOPS) |
| 5 | `base` memory, batch 16 | 6.6 GB peak (PC, bfloat16), 9.8 GB held (Mac, float32) |
| 5 | project 5's estimate of that memory | 6.0 GB in bfloat16, 8.8 GB in float32 |
| 6 | sweep: widths 64 / 96 / 128 / 192, validation loss | 5.221 / 4.811 / 4.568 / 4.202 |
| 6 | fit | loss = 7,470 × C^(−0.263) + 3.377 |
| 6 | width 256: predicted, then measured | 3.979, then 3.956 |
| 6 | the fit, stretched 30 times further, for `base`'s compute | 3.714 (`base` measured 3.715 and 3.735) |
| 7 | bits per byte, test reviews: `base` / token frequencies / gzip | 1.335 (PC), 1.341 (Mac) / 2.524 / 3.252 |
| 7 | sentiment, 4,000 training and 2,000 test reviews: `base` features / word counts | 85.9% (PC), 85.0% (Mac) / 71.2% |
| 7 | training reviews continued exactly (the 32 tokens after their first 32) | 0.0% |
| 8 | mean z-score: watermarked / unmarked / real reviews | 11.3 and 11.1 / 0.2 and −0.3 / −0.3 |
| 8 | flagged (z > 4): watermarked / unmarked / real | 100% / 0% / 0% |
| 8 | watermarked, still flagged after replacing 10% / 25% / 50% of tokens | 100% / 100% and 95% / 0% and 5% |

The fit's agreement with `base` is partly luck: `base` reads twice the context of the
sweep's models (lower loss), uses dropout and repeats its data more often (6.8 passes,
against at most 4.4 in the sweep; higher loss), and the effects partly cancel. Use it to show that a fit can extrapolate, not that it always will.
