# Chapter 3 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Your own training step

```python
def my_train_step(model, x, y, optimizer):
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

tinygpt.check.train_step(my_train_step)
```

## Why the order matters

1. `model(x, y)` runs the model forward and returns the loss for this batch.
2. `optimizer.zero_grad()` throws away the gradients left over from the last step.
   Without it, PyTorch *adds* the new gradients to the old ones (Modify B showed what
   that does).
3. `loss.backward()` computes the new gradient of every number in the model.
4. `optimizer.step()` moves every number a little, against its gradient.
5. `loss.item()` turns the one-number tensor into an ordinary Python number.

Clearing *after* `step()` instead of before `backward()` also works; the only rule is
that old gradients must be gone before new ones are computed.

tinyGPT's version has one more line, `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`,
between `backward` and `step`. It shrinks any unusually large step. It doesn't matter for
this small model, which is why the check passes without it, but it protects bigger models
from the occasional wild step.
