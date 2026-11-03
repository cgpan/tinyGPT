# Project 1 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/data.py`](scratchgpt/data.py).

## Predict 4: comparing every pair

100,000 reviews make 100,000 × 99,999 / 2 ≈ 5 billion pairs. At a million comparisons
a second that is about 83 minutes, and comparing two real texts is far slower than a
millionth of a second. With MinHash and banding, each review is hashed once and only pairs
that share a band are compared: on all of IMDB the whole pipeline took 40 to 57 seconds.

## What the reference found in IMDB

| | count |
|---|---|
| exact duplicates removed | 1,151: 949 among the 75,000 training reviews, 202 among the 25,000 test reviews |
| near-duplicates removed from training | 43 |
| training reviews removed because they repeated a test review | 389 |
| email addresses, web addresses, phone numbers replaced | 95, 425, 6 |
| documents left: training, validation, test | 73,619 (96.4 MB), 2,480 (3.2 MB), 22,318 (28.4 MB) |

So the answer to Predict 2 is yes: 389 training reviews repeated, or nearly repeated, a
test review. Print a few of them: where do they come from, given that IMDB's labelled
training and test reviews are about different films?

## What's easy to get wrong

- **`clean` in the wrong order.** Collapse spaces *after* decoding entities: `&nbsp;`
  becomes a non-breaking space only when it's decoded, and the space rule must see it.
- **`scrub` catching too much.** `\d{3}-\d{4}` alone matches parts of longer numbers. The
  look-behind `(?<![\w-])` and look-ahead `(?![\w-])` make sure a phone number isn't cut out
  of the middle of something else.
- **`minhash` overflow.** `a * x + b` must fit in 64 bits: `a` and `x` are below 2³², so
  their product is below 2⁶⁴. That's why `hash_functions` draws `a` below 2³² − 1 and why
  `x` is a CRC-32.
- **`candidate_pairs` per band.** A pair needs *one* band to match entirely, not some
  numbers in every band. The `.tobytes()` of a band slice is a convenient dictionary key.
- **`build_splits` keeping the wrong copy.** Near-duplicates within training drop the later
  document of each pair (`j`), whatever happened to `i`. A training document near a
  held-out one is always dropped: the held-out set is never changed to suit training.

## Investigate 2: why decontaminate

A test review that is also in the training data gets a low loss because the model has
seen it, not because the model is good. Every test number in project 7 would come out
better than the truth: lower bits per byte, and a memorization rate that measures nothing.
It's the same lesson as the entry track's missing mask: a number that is too good is a bug
until you have shown it isn't.
