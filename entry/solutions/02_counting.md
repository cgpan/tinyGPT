# Chapter 2 · Solutions to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Part 1: counting pairs

```python
def count_pairs(ids, vocab_size):
    counts = np.zeros((vocab_size, vocab_size))
    for a, b in zip(ids[:-1], ids[1:]):
        counts[a, b] += 1
    return counts

tinygpt.check.count_pairs(count_pairs)
```

`zip(ids[:-1], ids[1:])` walks two copies of the list side by side, one shifted by a
place, so each step hands you a character `a` and the character `b` right after it.
For `[5, 7, 9]` the pairs are `(5, 7)` and `(7, 9)`: a text of *n* characters has
*n* − 1 pairs. This is exactly the loop inside `CountBigram.fit`.

Run the cell after the challenge again and your counts give the same validation loss as
tinyGPT's: 2.4819.

## Part 2: counting triples

```python
def count_triples(ids, vocab_size):
    counts = np.zeros((vocab_size, vocab_size, vocab_size))
    for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
        counts[a, b, c] += 1
    return counts

tinygpt.check.count_triples(count_triples)
```

The same idea with three copies of the list, shifted by zero, one and two places.
Run the `triple_loss` cell again: with two characters of memory the validation loss is
**2.0458**.

The table now has 65 × 65 × 65 = 274,625 cells, and the training text fills only
11,228 of them. Each extra character of memory multiplies the table by 65 again, which
is why chapter 3 stops counting and starts learning.
