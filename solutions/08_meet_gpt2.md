# Chapter 8 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## The fair average

```python
def my_per_character(tokens, surprises):
    return sum(surprises) / sum(len(token) for token in tokens)

tinygpt.check.per_character(my_per_character)
```

## Why per character

Both models are surprised by the same text. The **total** surprise is a fair thing to
compare: it is how many nats each model needs, in all, to account for every character.
What differs is the number you divide by:

- divide by the number of **tokens**, and a model with long tokens looks worse, because
  each of its tokens carries several characters' worth of surprise;
- divide by the number of **characters**, and both models are measured against the same
  thing: the text.

`len(token)` counts the characters in one token: 1 for every tinyGPT token, usually
several for a GPT-2 token. For tinyGPT the two averages are therefore the same number.

The same trap comes up whenever language models with different tokenizers are compared.
Papers report *bits per character* or *bits per byte* for exactly this reason. To turn
nats into bits, divide by ln 2 (about 0.693).
