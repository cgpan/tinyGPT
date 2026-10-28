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
