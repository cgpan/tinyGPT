# Project 8 · Responsible release

Before a lab releases a model, it decides who may use it for what, measures what can go
wrong, and builds ways to tell its text from people's. Your model writes movie reviews:
the very thing the FTC's 2024 rule on fake reviews forbids passing off as real.
In this project you build the watermark detector, measure how well it works and how it
fails, and write the model card that goes with your model.

**You write**, in `src/scratchgpt/watermark.py`: `detect`.
**Given:** `green_ids` and `mark`, which your sampler has used since project 4, and
`scripts/watermark_check.py` and `scripts/model_card.py`.
**Done when** `python -m pytest tests/test_08_release.py` passes and your model card has no
TODO left.

## How the watermark works

Before each new token, the previous token and a secret key pick a random quarter of the
vocabulary, the "green" tokens, and the sampler adds 2 to their scores (Kirchenbauer et
al., 2023). The text still reads naturally: when one token is clearly right, a small push
changes nothing. But over many tokens, far more of them are green than the quarter chance
would give. Anyone with the key can count.

## Predict

1. In 150 tokens of unmarked text, how many green tokens do you expect? In watermarked
   text, more or fewer than half?
2. Someone replaces a quarter of the tokens at random. Will the detector still flag the
   text? Half of the tokens?
3. A review that repeats "great great great": strong or weak watermark? Why?
4. A real review, never seen by your model: how often will the detector wrongly flag it?

## Build

`detect` counts green tokens and turns the count into a z-score: how many standard
deviations above chance. Mind the edge cases: a text of 0 or 1 tokens has nothing to check.

## Run

```bash
python scripts/watermark_check.py runs/final
python scripts/model_card.py runs/final
```

The first writes 20 watermarked and 20 unmarked texts and scores them with 20 real
reviews; it saves only the scores. The second writes `MODEL_CARD.md` with every measured
number filled in.

## Investigate

1. Run the check on `small` and on your final model. A model that is surer of each token
   leaves the push less room to change anything. Which model's text carries the stronger
   watermark, and is the difference what that predicts?
2. Change `DELTA` to 1 and to 4. What happens to detection, and to the text? (Read a few
   samples of each. A bigger push means more green tokens, but at what cost?)
3. Detection needs the key. What happens if the key leaks? If it is never shared at all?
4. Random replacement is a weak attack. A person rewriting the text by hand, or another
   model paraphrasing it, is a strong one. What would that do to the score, and what does
   that mean for how much to trust "no watermark found"?

## The model card

Fill in every TODO in `MODEL_CARD.md`, in your own words:

- **Intended use:** what you built it to find out.
- **Out of scope:** the three rules are already there. Add anything else your testing
  showed it must not be used for.
- **Limitations:** read 20 of its texts first. Does it invent films, actors or facts?
  Repeat itself? Contradict itself within a review? How does it write about the people
  and groups that appear in reviews?
- **Risks:** who could be harmed if its text were posted as real reviews, and how far the
  watermark goes towards preventing it.

This is the README's Responsible use section, made specific to the model you built.

## Stretch

- **Find a watermark inside a longer text.** Mix a watermarked paragraph into a real
  review. Score a sliding window of 50 tokens and find where the generated part is.
- **Measure the watermark's cost.** Compute your model's own loss on watermarked and on
  unmarked samples. How much less likely, by its own judgement, is the watermarked text?
