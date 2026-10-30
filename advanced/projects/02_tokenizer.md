# Project 2 · The tokenizer

The entry track read Shakespeare one character at a time. Real models read word pieces:
fewer tokens per text means more text per training step and more text in the model's
context. In this project you write the byte-level BPE (byte-pair encoding) tokenizer that
GPT-2 introduced, and train it on the reviews.

**You write**, in `src/scratchgpt/tokenizer.py`: `merge`, `train_bpe`, and the
`Tokenizer` methods `encode_piece`, `encode` and `decode`.
**Given:** `pretokenize` (it cuts text into words, so merges never cross a word
boundary), `count_words`, the vocabulary built in `Tokenizer.__init__`, `save`, `load`,
and `encode_documents`, which writes token files.
**Done when** `python -m pytest tests/test_02_tokenizer.py` passes and `data/imdb/` holds
your tokenizer and the three token files.

## Predict

1. The vocabulary starts as the 256 possible bytes. What will the first merge be? The
   tenth?
2. With 8,192 tokens, how many bytes of review text will one token stand for, on average?
   (On the entry track's Shakespeare, 8,192 tokens give 3.5 bytes per token.)
3. Will ` movie` be one token? `Movie`? `MOVIE`? A misspelling like `moive`?
4. What happens to an emoji, or a word in Japanese, that the tokenizer never saw in
   training?

## Build

- **`merge`** first: the tests pin down overlapping pairs exactly.
- **`train_bpe`, the simple way.** Count every neighbouring pair over all pieces, merge
  the most frequent, repeat. `test_train_bpe_agrees_with_the_simple_definition` holds a
  slow version written as plainly as possible. Yours must give exactly the same merges,
  ties included.
- **`encode_piece`** applies the merges in the order they were learned, which isn't the
  same as left to right: see `test_encode_applies_merges_earliest_first`.
- **`encode`** splits off special tokens, pretokenizes, and encodes each piece. Cache the
  pieces: a few thousand words make up most of any text.
- **`decode`** joins the bytes. A model can write bytes that aren't valid UTF-8, so
  decoding must not crash.

## Run

The simple training loop recounts everything after every merge. That is fine for the
tests, but on all 100 MB of reviews it can take hours. Learn the merges from a sample
first:

```bash
python scripts/train_tokenizer.py --sample-mb 5
```

The script prints the token counts, the longest tokens, and how one sentence was cut.
After the stretch goal below, run it on everything: `python scripts/train_tokenizer.py`.

## Investigate

1. Print the first 20 merges as text (`tok.decode([256 + i])`). Were your predictions
   right? Why do so many tokens start with a space?
2. Train with 1,024, 4,096, 8,192 and 16,384 tokens (`--vocab-size`, on a 10 MB sample)
   and fill in a table of bytes per token. What do you gain from each doubling, and what
   does it cost? (Look ahead: the embedding table has vocab_size × d_model parameters.)
3. Encode ` movie`, `Movie`, `MOVIE` and `moive`. What does that mean for how much the
   model must learn?
4. **The units trap.** A model's loss is in nats per *token*. Your tokenizer and a
   classmate's cut the same text into different numbers of tokens, so your losses can't be
   compared. Bits per *byte* can. Convert: bits per byte = loss / (ln 2 × bytes per
   token). You'll use this in projects 4 to 7.

## Stretch

- **Fast training.** After a merge, only the pieces that contained the merged pair
  change. Keep a map from each pair to the pieces that contain it, and update only those
  counts. Then find the most frequent pair without scanning them all (a heap). Your
  merges must still match the simple definition exactly.
- Encode the training text in parallel with `multiprocessing`. How much faster is it on
  your machine?
- Compare with GPT-2's tokenizer (`pip install tiktoken`, outside your package): bytes per
  token on the same reviews, and how it cuts the same sentences. GPT-2 has 50,257 tokens
  and learned them from web pages.
