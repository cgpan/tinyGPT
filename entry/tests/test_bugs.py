"""The deliberate bugs of chapter 7: each one must really break something, and go away cleanly.

tinygpt.run_tests(bug=...) leaves this file out, because a bug planted for the whole run
would make these tests fail too.
"""

import math

import torch

import tinygpt
from tinygpt import check
from tinygpt.bugs import BUGS, planted


def tiny_gpt():
    torch.manual_seed(0)
    return tinygpt.GPT(tinygpt.Config(vocab_size=65, block_size=8, n_layer=1, n_head=2, n_embd=32, dropout=0.0)).eval()


def shifted(get_batch):
    x, y = get_batch(torch.arange(100), block_size=8, batch_size=2)
    return torch.equal(y, x + 1)


def test_no_shift_breaks_every_route_to_get_batch_and_is_removed_cleanly():
    from tinygpt.data import get_batch                     # a name bound before the bug is planted
    with planted("no_shift"):
        assert not shifted(get_batch) and not shifted(tinygpt.get_batch) and not shifted(tinygpt.training.get_batch)
    assert shifted(get_batch) and shifted(tinygpt.get_batch)


def test_big_init_raises_the_starting_loss_and_is_removed_cleanly():
    x, y = torch.randint(0, 65, (4, 8)), torch.randint(0, 65, (4, 8))
    with planted("big_init"):
        assert tiny_gpt()(x, y)[1].item() > 10
    assert math.isclose(tiny_gpt()(x, y)[1].item(), math.log(65), abs_tol=0.1)


def test_every_bug_changes_what_the_model_computes():
    x = torch.randint(0, 65, (1, 8))
    clean = tiny_gpt()(x)[0]
    for bug in ["no_mask", "softmax_wrong_way", "no_position", "big_init"]:
        with planted(bug):
            assert not torch.allclose(tiny_gpt()(x)[0], clean), bug
        assert torch.equal(tiny_gpt()(x)[0], clean), bug


def test_every_bug_has_a_description():
    assert set(BUGS) == {"no_mask", "softmax_wrong_way", "no_shift", "big_init", "no_position"}


def test_catches_check(capsys):
    def position_test():
        logits, _ = tiny_gpt()(torch.full((1, 8), 5))       # one character, eight times
        assert not torch.allclose(logits[0, 0], logits[0, -1], atol=1e-3)

    check.catches(position_test, bug="no_position")
    assert capsys.readouterr().out.startswith("✓")

    def weak_test():
        assert tiny_gpt()(torch.full((1, 8), 5))[0].shape == (1, 8, 65)
    check.catches(weak_test, bug="no_position")
    assert "doesn't catch it" in capsys.readouterr().out

    def wrong_test():
        assert False, "always fails"
    check.catches(wrong_test, bug="no_position")
    assert "would blame correct code" in capsys.readouterr().out
