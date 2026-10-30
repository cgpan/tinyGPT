# Project 2 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/tokenizer.py`](scratchgpt/tokenizer.py).

## The fast `train_bpe`

The simple version recounts every pair of every piece after each merge. The solution keeps
three things up to date instead:

- `pair_counts`: how often each pair occurs, weighted by how often its piece occurs;
- `where`: for each pair, the pieces that contain it;
- a heap of `(-count, -left, -right)`, so the most frequent pair (ties: the largest pair)
  comes out first.

After merging a pair, only the pieces in `where[pair]` change. For each, subtract the
counts of all its old pairs, merge, add the counts of all its new pairs. Counts that
changed are pushed onto the heap again; the old entries stay in it but are skipped when
popped, because their count no longer matches (a "lazy" heap).

On the entry track's Shakespeare (15,057 different pieces), learning 7,935 merges took
15.7 seconds with the count map alone, scanning all pairs with `max` for each merge, and
1.1 seconds with the heap. `test_train_bpe_agrees_with_the_simple_definition` checks that
both give exactly the same merges.

## What's easy to get wrong

- **Ties.** `max` over a dictionary returns the first maximum it meets, which depends on
  insertion order. Comparing `(count, pair)` makes the choice the same everywhere.
- **Overlapping pairs.** In `[a, a, a]` the pair `(a, a)` is counted twice, but merging
  gives `[X, a]`, not `[X, X]`.
- **Encoding by frequency, or left to right.** A piece must be encoded with the merge
  learned *earliest* among its pairs, again and again: that reproduces how training built
  the token.
- **Decoding token by token.** One character can be split over several tokens (an emoji
  is four bytes). Join all the bytes first, then decode once.

## Investigate 4: the units trap

Loss is measured per token. A tokenizer with longer tokens has fewer, harder predictions
per text, so its per-token loss is higher even if the model is just as good. Divide by
bytes per token (and ln 2, for bits) to get bits per byte, which is about the text, not the
tokenizer. The entry track's chapter 8 met the same trap with GPT-2.
