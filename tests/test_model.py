import math

import pytest
import torch

from tinygpt import GPT, BigramLM, Config, count_params


def tiny(**changes):
    values = dict(vocab_size=65, block_size=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    values.update(changes)
    return Config(**values)


def test_output_shape():
    cfg = tiny()
    logits, loss = GPT(cfg)(torch.randint(0, 65, (4, 8)))
    assert logits.shape == (4, 8, 65)
    assert loss is None


def test_untrained_gpt_loss_is_close_to_uniform_guessing():
    torch.manual_seed(0)
    x, y = torch.randint(0, 65, (16, 8)), torch.randint(0, 65, (16, 8))
    _, loss = GPT(tiny())(x, y)
    assert math.isclose(loss.item(), math.log(65), abs_tol=0.05)


def test_untrained_bigram_loss_is_exactly_uniform_guessing():
    x, y = torch.randint(0, 65, (16, 8)), torch.randint(0, 65, (16, 8))
    _, loss = BigramLM(tiny())(x, y)
    assert math.isclose(loss.item(), math.log(65), rel_tol=1e-6)


def changing_the_last_character_changes_earlier_outputs(cfg):
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    a = torch.randint(0, 65, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 65                        # edit only the future
    with torch.no_grad():
        return not torch.equal(model(a)[0][:, :-1], model(b)[0][:, :-1])


def test_attention_cannot_see_the_future():
    assert not changing_the_last_character_changes_earlier_outputs(tiny())


def test_without_the_mask_attention_does_see_the_future():
    assert changing_the_last_character_changes_earlier_outputs(tiny(causal=False))


def test_pos_emb_switch_removes_the_position_table():
    with_pos, without = GPT(tiny()), GPT(tiny(pos_emb=False))
    assert count_params(with_pos) - count_params(without) == 8 * 32
    assert without(torch.randint(0, 65, (2, 8)))[0].shape == (2, 8, 65)


def test_too_long_input_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        GPT(tiny())(torch.zeros((1, 9), dtype=torch.long))


def test_n_embd_must_divide_into_heads():
    with pytest.raises(ValueError, match="divisible"):
        tiny(n_embd=30, n_head=4)
