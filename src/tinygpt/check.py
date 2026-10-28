"""Checks for the Make exercises: does your function agree with tinyGPT's own?

Each check runs your function on a few small inputs, compares the answers with
tinyGPT's, and prints what it found. A check never stops your notebook with an
error.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from tinygpt.bigram import CountBigram
from tinygpt.config import Config
from tinygpt.data import load_text
from tinygpt.model import BigramLM
from tinygpt.tokenizer import CharTokenizer
from tinygpt.training import train_step as reference_train_step


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


def train_step(fn):
    """fn(model, x, y, optimizer) should do one step of learning and return the loss as a number."""
    _report(_train_step_problem(fn),
            "Your train_step changes the weights exactly as tinyGPT's does. You wrote the heart of training.")


def _train_step_problem(fn):
    torch.manual_seed(0)
    x, y = torch.randint(0, 65, (8, 16)), torch.randint(0, 65, (8, 16))
    runs = []
    for step_fn in (reference_train_step, fn):
        model = BigramLM(Config(vocab_size=65))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        tables, losses = [], []
        for _ in range(3):
            loss, error = _call(step_fn, model, x, y, optimizer)
            if error:
                return error
            losses.append(loss)
            tables.append(model.table.weight.detach().clone())
        runs.append((tables, losses))
    (want_tables, want_losses), (got_tables, got_losses) = runs
    if not isinstance(got_losses[0], float):
        return f"your function returned {_short(got_losses[0])}; return the loss as a plain number with loss.item()."
    if torch.equal(got_tables[0], torch.zeros_like(got_tables[0])):
        return "the weights did not change at all. Did you call optimizer.step()?"
    if not torch.allclose(got_tables[0], want_tables[0], atol=1e-6):
        return "the first step moved the weights the wrong way. Call loss.backward() before optimizer.step()."
    if not all(torch.allclose(g, w, atol=1e-6) for g, w in zip(got_tables, want_tables)):
        return ("the first step is right, but later steps are not. Old gradients are piling up: "
                "clear them with optimizer.zero_grad() before loss.backward().")
    if not all(math.isclose(g, w, rel_tol=1e-5) for g, w in zip(got_losses, want_losses)):
        return "the weights are right, but the loss you return is not. Return the loss of this batch."
    return None


def _reference_attention(q, k, v, mask=True, scale=True):
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(k.shape[-1])
    if mask:
        T = q.shape[1]
        scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def attention(fn):
    """fn(q, k, v) -> causal self-attention for one head. q, k and v have shape (B, T, head_size)."""
    _report(_attention_problem(fn),
            "Your attention matches tinyGPT's on every test. You wrote the core of a transformer.")


def _attention_problem(fn):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    got, error = _call(fn, q, k, v)
    if error:
        return error
    if not isinstance(got, torch.Tensor):
        return f"your function returned {_short(got)}; it should return a tensor."
    if got.shape != v.shape:
        return f"the output should have shape {tuple(v.shape)}, the same as v, but yours has {tuple(got.shape)}."

    def matches(**options):
        return torch.allclose(got, _reference_attention(q, k, v, **options), atol=1e-5)

    if matches():
        return None
    if matches(mask=False):
        return ("every position can see the future. Before the softmax, set the scores above the "
                "diagonal to -inf, for example with torch.tril and masked_fill.")
    if matches(scale=False):
        return "divide the scores by math.sqrt(head_size) before the softmax."
    if matches(mask=False, scale=False):
        return "two things are missing: divide the scores by math.sqrt(head_size), and hide the future."
    return ("the output doesn't match yet. Check the order: scores = q @ k.transpose(-2, -1), scale, "
            "mask, softmax over the last dimension (dim=-1), then multiply by v.")


def param_count(fn):
    """fn(vocab_size, block_size, n_layer, n_embd) -> how many parameters a GPT with those settings has."""
    _report(_param_count_problem(fn),
            "Your formula gives the exact parameter count every time. You know where every number lives.")


def _param_count_problem(fn):
    from tinygpt.model import GPT, count_params

    for vocab_size, block_size, n_layer, n_embd in [(65, 8, 1, 16), (65, 16, 2, 32), (10, 8, 3, 8), (65, 128, 6, 384)]:
        want = count_params(GPT(Config(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                                       n_head=4 if n_embd % 4 == 0 else 1, n_embd=n_embd)))
        got, error = _call(fn, vocab_size, block_size, n_layer, n_embd)
        if error:
            return error
        if got != want:
            off = f", off by {got - want:+,}" if isinstance(got, (int, float)) else ""
            return (f"for vocab_size={vocab_size}, block_size={block_size}, n_layer={n_layer}, n_embd={n_embd} "
                    f"the model has {want:,} parameters, but your formula gives {_short(got)}{off}.")
    return None
