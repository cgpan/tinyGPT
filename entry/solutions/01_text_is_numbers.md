# Chapter 1 · Solution to the Make challenge

Try the challenge yourself first. A solution you have read is much less use to you than
one you have struggled with, even if you did not finish it.

## Your own tokenizer

```python
chars = sorted(set(text))
stoi = {}
itos = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

def my_encode(s):
    ids = []
    for ch in s:
        ids.append(stoi[ch])
    return ids

def my_decode(ids):
    return "".join([itos[i] for i in ids])

tinygpt.check.tokenizer(my_encode, my_decode)
```

## Why it works

- `enumerate(chars)` hands out each character's position in the sorted list, and that
  position *is* its number. Because the list is sorted, everyone who builds a tokenizer
  from this text gets the same numbers.
- `stoi` and `itos` are mirror images: one goes from character to number, the other back.
- `my_encode` looks up one character at a time. `my_decode` looks up one number at a
  time and glues the characters together with `"".join(...)`.

Now compare with tinyGPT's own version, `tinygpt.show(tinygpt.CharTokenizer)`. It does
the same thing in fewer lines, with a *dictionary comprehension*:
`{ch: i for i, ch in enumerate(chars)}` builds `stoi` in one line. It also keeps `chars`
itself instead of `itos`, because a list can already be looked up by number.
