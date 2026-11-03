"""Project 8: the watermark detector."""

import random

import pytest
import torch
from conftest import tiny_config

from scratchgpt import watermark
from scratchgpt.model import GPT
from scratchgpt.sampling import generate


def test_green_lists_are_fixed_by_key_and_previous_token():
    a = watermark.green_ids(7, 1000)
    assert len(a) == 250 and len(set(a.tolist())) == 250
    assert torch.equal(a, watermark.green_ids(7, 1000))
    assert not torch.equal(a, watermark.green_ids(8, 1000))
    assert not torch.equal(a, watermark.green_ids(7, 1000, key="another key"))


def test_mark_raises_only_green_scores():
    logits = torch.zeros(2, 100)
    marked = watermark.mark(logits, torch.tensor([3, 4]))
    assert set(marked[0].unique().tolist()) == {0.0, watermark.DELTA}
    assert torch.equal(marked[0] > 0, torch.isin(torch.arange(100), watermark.green_ids(3, 100)))


def all_green(start, length, vocab_size):
    ids = [start]
    for _ in range(length):
        ids.append(watermark.green_ids(ids[-1], vocab_size)[0].item())
    return ids


def test_detect_scores_text_that_is_all_green():
    ids = all_green(5, 100, 1000)
    # T = 100 checked, all green: z = (100 - 25) / sqrt(100 * 0.25 * 0.75)
    assert watermark.detect(ids, 1000) == pytest.approx((100 - 25) / (100 * 0.25 * 0.75) ** 0.5)


def test_detect_finds_nothing_in_random_text():
    rng = random.Random(0)
    scores = [watermark.detect([rng.randrange(1000) for _ in range(200)], 1000) for _ in range(20)]
    assert abs(sum(scores) / len(scores)) < 1.0
    assert max(scores) < 4


def test_detect_needs_the_right_key():
    ids = all_green(5, 100, 1000)
    assert watermark.detect(ids, 1000, key="a different key") < 4


def test_detect_handles_short_texts():
    assert watermark.detect([], 1000) == 0.0 and watermark.detect([3], 1000) == 0.0


def test_generated_text_carries_the_watermark():
    torch.manual_seed(0)
    model = GPT(tiny_config(vocab_size=200, context=64))
    g = torch.Generator().manual_seed(0)
    marked = generate(model, torch.tensor([[1]]), 60, generator=g)[0].tolist()
    plain = generate(model, torch.tensor([[1]]), 60, watermark=False, generator=g)[0].tolist()
    assert watermark.detect(marked, 200) > 4
    assert watermark.detect(plain, 200) < 3
