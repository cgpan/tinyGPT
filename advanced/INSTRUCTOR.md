# Instructor note: the advanced track

For students who want to build language models for a living. They write every piece
against a test suite, in eight projects; expect several weeks of work. This page is for
you.

## Before you share it

- [ ] **Run `scripts/verify.py` on a GPU machine** (and on a Mac, and in Colab if you can).
      It runs the whole track with the reference solution on the real IMDB data and writes
      `runs/report-advanced-<computer>.md`. This hasn't been done yet: the IMDB download
      was blocked where the track was written, and there was no GPU. See *What has and
      hasn't been run* below.
- [ ] **Read the samples in that report** and decide whether IMDB is enough (see
      *Is IMDB enough?*).
- [ ] **Replace the estimates with measured numbers**: the time table in the README, and
      the expected numbers listed at the end of this page.
- [ ] **Make the repository public** (or give students access): the Colab workspace clones
      it. Until then, test in Colab with a token, as in the entry track's instructor note.

```bash
cd advanced
python -m pip install -e .
python scripts/verify.py --quick     # minutes: the whole pipeline on a 2 MB slice
python scripts/verify.py             # about 1 to 2 hours with a GPU: small and base on all of IMDB
python scripts/verify.py --sweep     # also the scaling sweep
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

With the solution, all 159 tests pass. With the stubs, 26 pass (the rules, the layout, and
tests of the given code) and the rest fail or error with "write this in project n".

## Is IMDB enough?

You asked whether IMDB can teach the model to write meaningful text, or whether a bigger
dataset is needed. Our estimate, which `verify.py` will test:

- IMDB's training text is about 100 MB, roughly 25 to 30 million tokens. That is small:
  a compute-optimal model for four passes over it has only about 5 to 6 million parameters.
  `base` (29.9M) uses five or six passes, with dropout, and keeps its best checkpoint.
- Reviews are a narrow, repetitive kind of text, which helps a small model. We expect
  grammatical, on-topic review sentences, with ideas that drift and facts that are made up
  (actors, plots). We haven't seen it yet.
- **How to decide:** read the ten samples from `base` in the report. If most sentences are
  grammatical and about films, IMDB is enough for this track. If they are not, the
  stretch goal in project 6 is the way out: pre-train on a general English corpus (a
  sample of FineWeb-Edu), then continue on IMDB. That would need a download script for
  the corpus, which isn't written yet.

The 24-hour budget is far more than IMDB needs (`base` should take under an hour). It
matters for students who take project 6's stretch and bring more data.

## What has and hasn't been run

Run, on a 4-core Linux CPU with PyTorch 2.15:

- the whole test suite against the reference (159 passed), and against the stubs;
- every script, end to end (`verify.py --quick`), on a fake IMDB folder made from
  Shakespeare's speeches, with the real folder layout;
- planted bugs: removing either random-state restore, the gradient-accumulation division,
  or AdamW's weight decay each fails exactly the test meant to catch it.

Not run yet, and worth watching in the first reports:

- the real IMDB download and its sizes, duplicate counts and timings;
- CUDA: bfloat16 on RTX 30/40 series, float16 with the gradient scaler on a T4;
- Apple's MPS backend (float32 by default), and Windows;
- `torch.compile` (`compile=true`), which may not work on Windows;
- `base` and the sweep at full size, and every quality number: bits per byte, sentiment
  accuracy, memorization.

## Notes for each project

1. **Data.** The pipeline runs in minutes. The near-duplicate threshold (0.8) and the
   phone pattern (North American only) are choices; good students question them.
2. **Tokenizer.** The simple BPE is far too slow for all of IMDB, so the brief has
   students train on a 5 MB sample first. The fast version is the stretch goal; the
   reference learns 7,935 merges from Shakespeare in about a second. Training the
   tokenizer on a sample changes the merges, and so every later number, a little.
3. **Model.** The most tests. The tied-embedding surprise (an untrained model scores below
   ln V when targets equal inputs) is a good discussion: in `base`, 7.22 against 9.12.
4. **Training.** The resume test is strict on purpose: it catches forgotten random states.
   Colab users must put `out_dir` on Drive; the workspace notebook does.
5. **Performance.** MFU will be low (our attention is not fused); the fused-attention and
   compile flags show how much. Ask for the filled-in table.
6. **Scaling.** The sweep takes roughly 10 to 30 minutes on a GPU (estimate). The final run is the
   student's own plan; ask to see it before they spend the hours.
7. **Evaluation.** The probe's zero-gradient trap is real (it happened while writing the
   reference); the docstring warns about it.
8. **Release.** Insist on a complete model card. It is the point of the project.

## Numbers to fill in after `verify.py`

From the report, for the briefs and your own reference: the counts from
`prepare_data.py` (duplicates, contamination, personal details); bytes per token at 8,192;
`small` and `base` validation loss and minutes; bits per byte for `base`, gzip and the
token-frequency baseline; both sentiment accuracies; the extraction rate; the watermark
detection rates; the benchmark table.
