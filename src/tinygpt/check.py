"""Checks for the Make exercises: does your function agree with tinyGPT's own?

Each check runs your function on a few small inputs, compares the answers with
tinyGPT's, and prints what it found. A check never stops your notebook with an
error.
"""

import numpy as np

from tinygpt.bigram import CountBigram
from tinygpt.data import load_text
from tinygpt.tokenizer import CharTokenizer


def _short(value, limit=40):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _times(n):
    return f"{n} time" if n == 1 else f"{n} times"


def _call(fn, *args):
    """Run your function; if it crashes, return a readable message instead."""
    try:
        return fn(*args), None
    except Exception as error:
        shown = ", ".join(_short(a) for a in args)
        return None, f"{fn.__name__}({shown}) raised {type(error).__name__}: {error}"


def _report(problem, success):
    print(f"✗ Not yet. {problem}" if problem else f"✓ {success}")


def tokenizer(encode, decode):
    """encode(text) -> list of numbers, and decode(numbers) -> text.

    Both should use tinyGPT's vocabulary: every distinct character of the corpus, sorted.
    """
    _report(_tokenizer_problem(encode, decode),
            "Your encode and decode agree with tinyGPT's on every test. You built a tokenizer.")


def _tokenizer_problem(encode, decode):
    reference = CharTokenizer.from_text(load_text())
    samples = ["To be, or not to be", "ROMEO:\nBut, soft!", "".join(reference.chars), "\n"]
    for text in samples:
        want = reference.encode(text)
        got, error = _call(encode, text)
        if error:
            return error
        if not isinstance(got, list):
            return f"encode({_short(text)}) returned {_short(got)}; it should return a list of numbers."
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return (f"in encode({_short(text)}), the character {text[i]!r} should become {w}, "
                        f"but yours gives {_short(g)}. Is your vocabulary sorted(set(text))?")
        if len(got) != len(want):
            return (f"encode({_short(text)}) should give {len(want)} numbers, one per character, "
                    f"but yours gives {len(got)}.")
        back, error = _call(decode, want)
        if error:
            return error
        if back != text:
            return f"decode({_short(want)}) should give back {_short(text)}, but yours gives {_short(back)}."
    return None


def _cases():
    rng = np.random.default_rng(0)
    return [([0, 1, 0, 1, 1, 2, 0], 3), ([2, 2, 2, 2], 3), ([1], 2), (rng.integers(0, 5, 300).tolist(), 5)]


def _table_problem(fn, reference, what):
    for ids, vocab_size in _cases():
        want = reference(ids, vocab_size)
        got, error = _call(fn, list(ids), vocab_size)
        if error:
            return error
        got = np.asarray(got)
        if got.shape != want.shape:
            return (f"for vocab_size={vocab_size} the table should have shape {want.shape}, "
                    f"but yours has shape {got.shape}. Did you return the table?")
        wrong = np.argwhere(got != want)
        if len(wrong):
            cell = tuple(int(i) for i in wrong[0])
            arrow = " → ".join(str(i) for i in cell)
            return (f"for ids={_short(ids)}, the {what} ({arrow}) appears {_times(int(want[cell]))}, "
                    f"but your table says {got[cell]:g}.")
    return None


def count_pairs(fn):
    """fn(ids, vocab_size) -> table where table[a, b] counts how often b comes right after a."""
    _report(_table_problem(fn, lambda ids, v: CountBigram(v).fit(ids).counts, "pair"),
            "Your table matches tinyGPT's on every test. You built the counting part of a language model.")


def _triples(ids, vocab_size):
    counts = np.zeros((vocab_size, vocab_size, vocab_size))
    for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
        counts[a, b, c] += 1
    return counts


def count_triples(fn):
    """fn(ids, vocab_size) -> table where table[a, b, c] counts how often c comes right after a, b."""
    _report(_table_problem(fn, _triples, "triple"),
            "Your table matches on every test: a model with two characters of memory.")
