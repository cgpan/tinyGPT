# Instructor note

tinyGPT is a self-paced handbook: students work through the chapter notebooks in order,
in Colab or on their own GPU machine. This page is for you.

## Before you share it

- [ ] **Make the repository public.** Every notebook's setup cell clones it. While it is
      private, the setup cell fails in Colab for everyone except you (see below).
- [ ] **Run each chapter in Colab.** Open it from the links in the README and choose
      Runtime > Run all. It should finish without errors. Chapters 1 and 2 take under a
      minute and need no GPU.
- [ ] **From chapter 3 on, check the training times.** Run `python scripts/benchmark.py`
      once on a Colab T4, and on a Mac if you have one.

### Checking it on your own machine

`scripts/verify.py` runs everything: the tests, the benchmark, the three training configs,
and every chapter notebook from top to bottom (chapter 8 downloads GPT-2). It writes
`runs/report-<computer name>.md`. Nothing in the repository is changed.

```bash
python -m pip install -e . nbclient nbformat ipykernel transformers
python scripts/verify.py
```

Install a GPU build of PyTorch first. It takes about 20 to 40 minutes with a GPU;
`--quick` takes a minute or two.

### Testing in Colab while the repository is private

Add a GitHub token with read access to this repository to Colab's Secrets panel (the key
icon) under the name `GITHUB_TOKEN`. Then run this in a new cell *before* the setup cell:

```python
from google.colab import userdata
!git clone -q https://{userdata.get('GITHUB_TOKEN')}@github.com/cgpan/tinyGPT.git /content/tinyGPT
```

The setup cell sees that the folder exists and skips its own clone. Don't save that
extra cell into the repository.

## How every chapter is built

- **Setup cell first.** It is collapsed in Colab and identical in every chapter.
- **Five steps:** Predict, Run and Investigate (`everyone`), Modify (`most people`),
  Make (`optional challenge`). The tags in the headings are the three tiers.
- **One open box.** Each chapter has exactly one `tinygpt.show(...)`, which displays
  the code being studied, exactly as it runs. Everything else is called, not shown.
- **The number.** Each chapter ends with `tinygpt.scoreboard(...)`, which places the
  student's result between 4.17 (knowing nothing) and 1.49 (tinyGPT).
- **Make challenges** are checked by `tinygpt.check`, which prints ✓ or one specific
  hint and never stops the notebook. Solutions, with a short explanation, are in
  `solutions/`, one file per chapter, linked from the end of each Make section. A test
  runs every solution against its check.

## Editing a notebook

Edit in Colab or Jupyter, then clear all outputs before committing (Edit > Clear all
outputs): saved outputs would give away the answers to the Predict steps.
`python -m pytest` checks this, and the rest of the template above.

## Chapter by chapter

The numbers students should see (all measured), and where they tend to get stuck.

### Chapter 1 · Text is numbers (no GPU)

- 1,115,394 characters, 65 distinct. Most common: the space (169,892). Rarest: `$` (once).
- Knowing nothing: 4.17. Knowing frequencies: 3.35. Lower-case vocabulary: 39
  characters, knowing nothing 3.66.
- `'\n'` and `' '` look blank in the printed vocabulary; point them out.
- The tokenizer is a class. Students who haven't met classes yet can read `encode` and
  `decode` as two plain functions.
- Modify B is the first *units trap*: a smaller vocabulary lowers the number without
  making the model any better. It comes back whenever models with different
  vocabularies are compared.

### Chapter 2 · Counting beats guessing (no GPU)

- Counting bigram: training 2.4546, validation 2.4819 (perplexity 12.0).
- Every "q" in the training text is followed by "u" (563 of 563), yet the model says
  0.898: the smoothing keeps some probability for the other 64 characters.
- Smoothing: alpha 0 gives an infinite loss (187 validation pairs never occur in
  training); alpha 0.01 gives 2.4875; alpha 100 gives 2.6307.
- Less data: 1,000 characters give 3.28; 10,000 give 2.75; 100,000 give 2.62.
- Make, part 2 (counting triples, alpha 0.1): 2.0458.
- Hand-written loops over a tensor are slow; the cells pass `train_ids.tolist()`.
  A student who builds a 5-character table runs out of memory (it needs 9 GB): restart
  the runtime and run the setup cell again.

### Chapter 3 · Learning instead of counting (no GPU needed)

- The learned bigram (learning rate 0.01, 2,000 steps) starts at exactly 4.17, because
  its scores start at zero, and ends at about 2.49: 2.489 on the whole validation text,
  against counting's 2.482.
- Learning rates: 0.001 is still at 2.75 after 2,000 steps; 0.01 and 0.1 reach about
  2.49; 1.0 stalls near 2.58; 10 bounces around 3.1.
- Forgetting `zero_grad`: the loss gets *worse* as training goes on, from 2.61 to 2.85.
- The untrained transformer (`base`, seed 1337) starts at 4.33. Its favourite next
  character gets 4.2% on average where an even spread gives 1.5%.
- This is the first chapter with PyTorch. Students don't need to understand how
  gradients are computed; the five lines of `train_step` are the whole story here.
- Expect "learning gave the same number, so what was the point?" The answer is in
  Common questions, and it pays off in chapter 4.
- A GPU makes no difference in this chapter: the model is too small to benefit.

### Chapter 4 · Looking back: attention (GPU recommended)

- One block with 64 characters of memory (`small` with `n_layer=1`, 223,041
  parameters): 1.76, in about 35 seconds on a 4-core CPU.
- Without the mask: 0.04, and its writing is gibberish (in our run, `ROMEO::EO::O::O::::`
  and colons from then on).
- Memory length: 8 characters give 2.10, 16 give 1.92, 64 give 1.76.
- The attention pictures differ from run to run: which head does what depends on
  training. In our run one head looked almost only at the previous character. The
  questions are worded so that any pattern works.
- The mask experiment is the chapter's key moment: a near-perfect loss from a model that
  can't write. Give students time with "why did the low loss mislead us?". It is the same
  lesson as a leaky evaluation in their own research.

### Chapter 5 · The full run (GPU needed)

- `base`: 10,739,777 parameters. Validation 1.49 at step 2,500 (training 1.21); it
  passes 2.48 at about step 250.
- Measured training time for `base`: 1.9 minutes on an NVIDIA RTX 4070 SUPER (Windows)
  and 4.1 minutes on an Apple M5 Max. On a Colab T4 it is still an estimate (6 to 8
  minutes); time one run there before you share the chapter. The whole chapter took 7 and
  11 minutes on those two machines.
- Validation loss at step 2,500 across our runs: 1.49 to 1.52 (dropout makes each run a
  little different).
- Feed-forward holds two-thirds of the parameters (7.09M of 10.74M); attention one-third.
- `small` ablations (2,000 steps): one block 1.76, two blocks 1.70, four blocks 1.69.
  Without positional embeddings: 1.81. Dropout 0.2: about 1.83 to 1.85 (training about
  1.67), worse than none: it narrows the gap between training and validation loss a
  little, but a small model trained this briefly loses more from it than it gains. `base`
  uses dropout 0.2 because a 10.7M-parameter model memorises far more.
- Runs with dropout don't repeat exactly: two of our runs of the same dropout setting
  gave 1.83 and 1.85. Runs without dropout repeated to four decimal places on the same
  machine. Expect students' numbers to differ from these by a few hundredths.
- Make, part 1: each block has 12d² + 10d parameters. Part 2 has no fixed answer.
- In Colab the trained model in `runs/base/model.pt` disappears when the session ends.

### Chapter 6 · Talking to your model (GPU only if the model must be retrained)

- Students load the model they kept at the end of chapter 5: from Google Drive
  (`MyDrive/tinyGPT/model.pt`) on Colab, from `runs/base/model.pt` on their own computer.
  If none is found, the chapter trains one first.
- Measured with our `base` model (3,000 steps, validation 1.50), averaged over 10
  batches: validation loss at temperature 0.5 is 2.07, at 1.0 1.49, at 1.2 1.46, at 1.3
  1.45 (the best), at 2.0 1.58. On *training* text the best temperature is exactly 1.0. The model is overconfident on new text because it has
  partly memorised its training text. This is a calibration result; it surprised us too.
- Greedy writing (`top_k=1`) gets stuck in a loop ("the way of the way of the way").
  Temperature 0.1 is nearly the same; 2.0 is gibberish.
- Longest piece copied from the training text, in 1,000 characters: 15 to 24 characters
  at temperature 1.0, 23 to 34 at 0.5; mostly speaker names such as `LADY ANNE:`. The
  counting bigram's longest is 8.
- Writing 300 characters takes about 4 seconds on a CPU.

### Chapter 7 · How do you know it works? (no GPU needed)

- The five planted bugs, and the tests that catch them:
  - `no_mask` and `softmax_wrong_way`: `test_attention_cannot_see_the_future` and
    `test_attention_weights_look_only_backwards_and_sum_to_one`.
  - `no_shift`: the two batch tests in `tests/test_data.py`.
  - `big_init`: `test_untrained_gpt_loss_is_close_to_uniform_guessing`. The untrained
    `small` model starts at about 27 instead of 4.2.
  - `no_position`: none. Yet it costs a lot: the one-block model reaches 2.38 instead
    of 1.76. The Make challenge is to write the test that catches it.
- With `no_shift` the one-block model's loss falls to 0.000 within 250 steps, and it
  writes nothing but colons.
- A run of the suite takes 5 to 10 seconds. Runs with a bug planted leave out the tests
  of the bug-planting and of the solutions, which need tinyGPT without a bug, so they
  report a dozen or so fewer tests than the clean run.
- Modify B suits a "coding with AI" lesson: students ask an AI assistant for a test, then
  check whether it really fails when the bug is planted. The example test in the cell
  only checks shapes, and doesn't catch the bug.
- Bugs are planted with `tinygpt.bugs.planted(...)` inside a `with` block, and removed
  when the block ends. `tinygpt.run_tests(bug=...)` plants one for a whole run of the
  suite, in a separate process.

### Chapter 8 · Meet a real one (optional; GPU recommended)

- Downloads GPT-2 small (`openai-community/gpt2`, about 500 MB) from Hugging Face with
  the `transformers` library. Colab has the library already; `load_gpt2()` installs it
  anywhere else.
- Measured for tinyGPT (`base`, 3,000 steps): 1.486 nats per character on the whole
  validation text, read in 128-character windows, and 1.88 on the modern-English example
  in Modify A.
- GPT-2, measured on Windows and macOS (the same on both): **4.00 nats per token, 1.29
  per character** on the validation text. One GPT-2 token covers 3.09 characters of our
  Shakespeare; the Romeo line is 16 GPT-2 tokens. So GPT-2 looks almost as bad as knowing
  nothing per token (4.00 against 4.17), yet beats tinyGPT (1.49 to 1.51) per character.
  That is the units trap, and the reason the chapter exists.
- On the modern-English example in Modify A: GPT-2 0.73 per character, tinyGPT 1.88 to 1.99.
- GPT-2 was most surprised by the start of the play format: 10.1 nats for "EO" in
  "ROMEO" and 11.7 for "But" at the start of a line. Passages dense with speaker names
  are the likeliest place for tinyGPT to win the "going further" challenge; we haven't
  searched for one.
- **Content warning.** GPT-2 learned from unfiltered web pages. In one of our test runs,
  its first sample at temperature 1.0 was sexually explicit. GPT-2's writing now appears
  only in Modify B, with a warning; delete that cell if you'd rather students not see
  unfiltered output.
- The comparison code was tested with a small, untrained GPT-2 built locally from the
  same `transformers` classes, and `load_gpt2` was tested loading from a local folder.
- Text pasted into Modify A often has curly quotes (’ “ ”), which tinyGPT can't read.
