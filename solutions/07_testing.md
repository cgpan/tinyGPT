# Chapter 7 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## A test that catches `no_position`

```python
import torch

def test_position_matters():
    torch.manual_seed(0)
    cfg = tinygpt.Config(vocab_size=65, block_size=8, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    model = tinygpt.GPT(cfg).eval()
    same = torch.full((1, 8), 5)                  # one character, eight times
    logits, _ = model(same)
    assert not torch.allclose(logits[0, 0], logits[0, -1])

tinygpt.check.catches(test_position_matters, bug="no_position")
```

## Why it works

Without the position table, every copy of the character starts as the same vector. The
first position can only attend to itself; the last attends to eight identical copies, and
a weighted average of identical vectors is that same vector again. So every position ends
up with exactly the same scores, and `torch.allclose` says they are equal: the test fails.

With the position table, each copy gets a different position vector added, so the scores
differ and the test passes.

This test checks a **property** (position changes the output) rather than an exact number,
so it doesn't break when the model's starting weights change. Compare it with
`test_pos_emb_switch_removes_the_position_table`, which only checks that the table exists,
not that it is used.

## A good test from an AI, for comparison

For Modify B, a test that does catch `no_shift` checks the relation between `x` and `y`,
not only their shapes. For example:

```python
def test_targets_are_inputs_moved_by_one():
    ids = torch.arange(100)
    x, y = tinygpt.get_batch(ids, block_size=8, batch_size=4)
    assert torch.equal(x[:, 1:], y[:, :-1])

tinygpt.check.catches(test_targets_are_inputs_moved_by_one, bug="no_shift")
```
