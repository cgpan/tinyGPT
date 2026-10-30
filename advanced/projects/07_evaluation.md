# Project 7 · Evaluation

How good is your model, measured in ways that can't fool you? One number that compares
fairly across tokenizers, two baselines that say what "good" means, a test of what the
model understood, and a test of what it copied.

**You write**, in `src/scratchgpt/evaluate.py`: `token_byte_lengths`, `bits_per_byte`,
`unigram_bits_per_byte`, `review_features`, `bag_of_words`, `train_probe`,
`extraction_rate`.
**Given:** `gzip_bits_per_byte`, and `scripts/evaluate.py`, which runs it all.
**Done when** `python -m pytest tests/test_07_evaluation.py` passes and you have evaluated
`small` and your final model.

## Predict

1. **Bits per byte** on the test reviews. Knowing nothing about bytes is 8 bits per byte.
   Where will gzip be? A model that knows only how common each token is? Your final model?
2. **Sentiment.** Your model never saw a label. A logistic regression on its hidden
   states, trained on 4,000 labelled reviews: what accuracy on new reviews? And a logistic
   regression on plain word counts, the classic baseline? (Maas et al., 2011 report
   word-count classifiers in the high 80s, in percent, with all 25,000 training reviews.)
3. **Memorization.** Give the model the first 32 tokens of a training review. How often
   will it write the next 32 exactly?

## Build

- **`bits_per_byte`.** Cut the test tokens into windows of the context length, sum the
  surprise at every target, and divide by the bytes those targets stand for. End-of-text
  tokens stand for no text: leave them out of both sums.
- **`unigram_bits_per_byte`.** The simplest real model: every token gets its training
  frequency, whatever came before. It is the floor any model that reads context must beat.
- **`review_features`.** Average the final hidden states over each review's tokens. Pad
  short reviews in a batch, and keep the padding out of the average (the test checks).
- **`train_probe`.** Logistic regression with your own AdamW. Read the docstring's note on
  `clamp`: the obvious way to write the loss never trains.
- **`extraction_rate`.** Greedy continuation with the KV cache, no watermark. Exact matches
  only: a near-copy isn't counted.

## Run

```bash
python scripts/evaluate.py runs/small
python scripts/evaluate.py runs/final
```

| | small | final |
|---|---|---|
| bits per byte: your model | | |
| bits per byte: token frequencies | | |
| bits per byte: gzip | | |
| sentiment: probe on your model | | |
| sentiment: probe on word counts | | |
| training reviews continued exactly | | |

## Investigate

1. Convert your validation loss from project 4 (nats per token) into bits per byte. Does
   it match what `bits_per_byte` gives on the test reviews? Should it, exactly?
2. Print the surprise at each token of one test review. Which words surprise the model
   most? Names, plot details, the verdict?
3. Which beats which in sentiment: your model's features or word counts? Did the bigger
   model help? What does that say about what pretraining on reviews teaches?
4. Look at the reviews your model continued exactly, if any. What are they? Short? Did
   project 1's deduplication miss something?
5. A higher temperature makes writing more varied. What does it do to bits per byte?
   (Trick question: check what `bits_per_byte` measures.)

## Stretch

- **Zero-shot sentiment.** Without a probe: append " Overall, this movie was great." and
  " Overall, this movie was terrible." to a review, and compare the model's loss on each.
  How accurate is that?
- **Calibration.** When your model gives a token 30%, is it right 30% of the time? Bin
  the predicted probabilities of the chosen tokens and compare.
