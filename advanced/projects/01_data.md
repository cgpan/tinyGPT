# Project 1 · The data

Most of the work on a real language model is the data, and most of a model's problems
start there. Duplicates teach it to repeat; test text that leaks into training makes every
score a lie; personal details in the data can come back out in what it writes. In this
project you turn 100,000 raw IMDB reviews into training, validation and test documents
you can trust.

**You write**, in `src/scratchgpt/data.py`: `clean`, `scrub`, `fingerprint`, `shingles`,
`minhash`, `estimated_jaccard`, `candidate_pairs`, `near_duplicates`, `build_splits`.
**Given:** `imdb.download` and `imdb.read_reviews`, which fetch and read the files, and
`hash_functions`.
**Done when** `python -m pytest tests/test_01_data.py` passes, you have run
`scripts/prepare_data.py`, and you have written your data card.

## Predict

Write your guesses down before you look at any data.

1. Of the 100,000 reviews, how many are exact copies of another one?
2. Will any of the 25,000 test reviews also turn up among the 75,000 training ones?
3. How many email addresses do you expect? Phone numbers?
4. To find near-duplicates by comparing every pair of reviews, how many comparisons would
   you make? At a million comparisons a second, how long would that take?

## Build

Work through the functions in the order above. Each docstring is the full specification,
and each has its own tests: run them as you go, for example
`python -m pytest tests/test_01_data.py -k scrub`.

- **`clean`.** The reviews are scraped HTML: line breaks are `<br />`, quotes can be
  `&quot;`. Look at a few raw reviews first (`next(imdb.read_reviews(root))["text"]`).
- **`scrub`.** Regular expressions. The tests list what must be caught and what must be
  left alone: a rating like 7/10 is not a phone number. A pattern that catches too much
  damages the text; one that catches too little leaks someone's details.
- **`fingerprint`** catches copies that differ only in case or punctuation. Two reviews
  are exact duplicates when their fingerprints match.
- **`shingles`, `minhash`, `estimated_jaccard`.** Near-duplicates are reviews posted twice
  with a small edit. Their similarity is the Jaccard index of their sets of 5-word
  shingles: shared shingles divided by all shingles. MinHash estimates it from 128
  numbers per review, without comparing the texts.
- **`candidate_pairs`.** Comparing all pairs of signatures is still too slow. Cut each
  signature into 16 bands of 8 numbers, and only compare reviews that match in a whole
  band. A pair with similarity *s* becomes a candidate with probability
  1 - (1 - *s*^8)^16: 95% at *s* = 0.8, 6% at *s* = 0.5.
- **`build_splits`** puts it together. Read its docstring twice: the order of the steps
  matters, and so does which copy of a duplicate you keep.

## Run

```bash
python scripts/prepare_data.py
```

It downloads the reviews (84 MB, once), runs your pipeline, writes `data/imdb/` and prints
what it did. Compare the counts with your predictions.

## Investigate

1. Print five near-duplicate pairs (`data.near_duplicates` on a few thousand training
   reviews). What kind of edit made them different?
2. Decontamination removed some training reviews because they repeat a test review. If
   you had left them in, which number in project 7 would have been wrong, and in which
   direction? The entry track met the same mistake in chapter 4, as a missing mask.
3. Search the cleaned training text for `@` and for long runs of digits. Did `scrub` miss
   anything? Is there anything it should not have replaced?
4. How long did the whole pipeline take? Compare it with your answer to Predict 4.

## Your data card

Write `DATA_CARD.md` in the `advanced` folder, in your own words, with your numbers:

- **Source:** who collected the reviews, when, and under what terms (see the README's
  License and data section).
- **Contents:** how many reviews and bytes in each split; what language; who wrote them.
- **Processing:** every step of `build_splits`, with how many reviews or details each
  one removed or replaced.
- **Gaps:** what this data does not cover. Who doesn't write IMDB reviews? Which films
  are missing? What does a model trained only on this think a "review" is?

## Stretch

- Find the reviews that aren't in English, with a simple rule you can explain. How many
  are there? Should they stay?
- Make `near_duplicates` faster: compute all the signatures as one NumPy array.
- The `movie` field says which film a review is about. Do the unlabelled reviews share
  films with the test reviews? Does it matter for what we measure?
