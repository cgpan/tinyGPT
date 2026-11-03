"""Project 3: the transformer's parts, checked against PyTorch's own versions."""

import math

import pytest
import torch
from conftest import tiny_config
from torch.nn import functional as F     # tests may use it as the answer key; your code may not

from scratchgpt import layers
from scratchgpt.model import GPT, count_parameters, parameter_formula


def test_linear_matches_x_times_w_transposed():
    torch.manual_seed(0)
    lin = layers.Linear(8, 5)
    assert lin.weight.shape == (5, 8)
    x = torch.randn(3, 4, 8)
    assert torch.allclose(lin(x), F.linear(x, lin.weight), atol=1e-6)
    assert not hasattr(lin, "bias") or lin.bias is None


def test_linear_starts_small_and_cut_off():
    torch.manual_seed(0)
    w = layers.Linear(512, 512, std=0.02).weight
    assert abs(w.std().item() - 0.02 * 0.986) < 0.001       # cutting a normal at 3 sd shrinks its sd a little
    assert w.abs().max().item() <= 0.06 + 1e-6


def test_embedding_looks_up_rows():
    emb = layers.Embedding(10, 4)
    ids = torch.tensor([[1, 2], [9, 1]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    assert torch.equal(out[1, 1], emb.weight[1])


def test_rmsnorm_matches_the_formula_and_keeps_the_dtype():
    norm = layers.RMSNorm(6)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(1.0, 7.0))
    x = torch.randn(2, 3, 6)
    expected = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)
    assert norm(x.to(torch.bfloat16)).dtype == torch.bfloat16
    assert torch.equal(layers.RMSNorm(6).weight, torch.ones(6))


def test_softmax_matches_torch_even_for_huge_scores():
    x = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0], [-1e4, 0.0, 1e4]])
    out = layers.softmax(x, dim=-1)
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)
    assert not torch.isnan(out).any()
    y = torch.randn(4, 5, 6)
    assert torch.allclose(layers.softmax(y, dim=1), torch.softmax(y, dim=1), atol=1e-6)


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(10, 7) * 5
    targets = torch.randint(7, (10,))
    assert torch.allclose(layers.cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)
    huge = torch.tensor([[1000.0, 0.0]])
    assert layers.cross_entropy(huge, torch.tensor([1])).item() == pytest.approx(1000.0)


def test_cross_entropy_has_gradients():
    logits = torch.randn(4, 3, requires_grad=True)
    layers.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_dropout_zeroes_and_rescales_only_in_training():
    torch.manual_seed(0)
    x = torch.ones(1000, 100)
    out = layers.dropout(x, 0.25, training=True)
    assert abs((out == 0).float().mean().item() - 0.25) < 0.01
    assert torch.allclose(out[out != 0], torch.full_like(out[out != 0], 1 / 0.75))
    assert torch.equal(layers.dropout(x, 0.25, training=False), x)
    assert torch.equal(layers.dropout(x, 0.0, training=True), x)


def slow_rope(x, theta=10000.0):
    """Rotary embeddings one position and one pair at a time, with 2 x 2 rotation matrices."""
    out = torch.empty_like(x)
    T, D = x.shape[-2:]
    for t in range(T):
        for i in range(D // 2):
            angle = t * theta ** (-2 * i / D)
            rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            out[..., t, 2 * i : 2 * i + 2] = x[..., t, 2 * i : 2 * i + 2] @ rotation.T
    return out


def test_rope_tables_have_the_right_angles():
    cos, sin = layers.rope_frequencies(8, 5)
    assert cos.shape == sin.shape == (5, 4) and cos.dtype == torch.float32
    assert cos[3, 1].item() == pytest.approx(math.cos(3 * 10000 ** (-2 / 8)), abs=1e-6)
    assert torch.allclose(cos[0], torch.ones(4)) and torch.allclose(sin[0], torch.zeros(4))


def test_rope_rotates_pairs_by_position():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, 8)
    cos, sin = layers.rope_frequencies(8, 6)
    assert torch.allclose(layers.apply_rope(x, cos, sin), slow_rope(x), atol=1e-5)
    assert layers.apply_rope(x.to(torch.bfloat16), cos, sin).dtype == torch.bfloat16


def test_rope_makes_scores_depend_only_on_distance():
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    cos, sin = layers.rope_frequencies(8, 20)
    def score(m, n):
        return (layers.apply_rope(q.view(1, 8), cos[m : m + 1], sin[m : m + 1]) *
                layers.apply_rope(k.view(1, 8), cos[n : n + 1], sin[n : n + 1])).sum().item()
    assert score(5, 2) == pytest.approx(score(15, 12), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)


def test_causal_attention_matches_torch():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 7, 4) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(layers.causal_attention(q, k, v), expected, atol=1e-5)


def test_causal_attention_with_fewer_queries_than_keys():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)
    allowed = torch.ones(3, 7, dtype=torch.bool).tril(4)      # query i sits at position 4 + i
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    assert torch.allclose(layers.causal_attention(q, k, v), expected, atol=1e-5)


def test_attention_cannot_see_the_future():
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = layers.Attention(cfg, layer=0).eval()
    cos, sin = layers.rope_frequencies(cfg.head_dim, cfg.context)
    x = torch.randn(1, 10, cfg.d_model)
    changed = x.clone()
    changed[0, 6:] = torch.randn(4, cfg.d_model)
    a, b = attn(x, cos, sin), attn(changed, cos, sin)
    assert torch.allclose(a[0, :6], b[0, :6], atol=1e-6)
    assert not torch.allclose(a[0, 6:], b[0, 6:], atol=1e-3)


def test_attention_output_layer_starts_smaller():
    torch.manual_seed(0)
    cfg = tiny_config(n_layer=8, d_model=256, n_head=4)
    attn, ff = layers.Attention(cfg, layer=0), layers.FeedForward(cfg)
    params = dict(attn.named_parameters())
    assert set(params) == {"qkv.weight", "out.weight"}
    assert params["qkv.weight"].shape == (768, 256)
    assert params["out.weight"].std().item() < 0.5 * params["qkv.weight"].std().item()
    assert ff.w2.weight.std().item() < 0.5 * ff.w1.weight.std().item()


def test_feed_forward_is_swiglu():
    torch.manual_seed(0)
    cfg = tiny_config(d_model=32)
    ff = layers.FeedForward(cfg).eval()
    assert ff.w1.weight.shape == (cfg.ff_dim, 32) and ff.w2.weight.shape == (32, cfg.ff_dim)
    x = torch.randn(2, 5, 32)
    expected = F.linear(F.silu(F.linear(x, ff.w1.weight)) * F.linear(x, ff.w3.weight), ff.w2.weight)
    assert torch.allclose(ff(x), expected, atol=1e-6)


# --- the whole model -------------------------------------------------------------------

def test_gpt_gives_logits_and_loss():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = GPT(cfg)
    ids = torch.randint(cfg.vocab_size, (3, cfg.context))
    logits, loss = model(ids, ids)
    assert logits.shape == (3, cfg.context, cfg.vocab_size)
    assert loss.shape == () and model(ids)[1] is None
    assert model(ids, hidden=True).shape == (3, cfg.context, cfg.d_model)


def test_untrained_gpt_is_close_to_uniform_guessing():
    torch.manual_seed(0)
    cfg = tiny_config(vocab_size=500, d_model=64, n_head=4)
    model = GPT(cfg).eval()
    ids, targets = torch.randint(cfg.vocab_size, (2, 4, cfg.context))
    _, loss = model(ids, targets)
    assert abs(loss.item() - math.log(500)) < 0.1


def test_gpt_is_causal():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    ids = torch.randint(64, (1, 12))
    changed = ids.clone()
    changed[0, 8] = (changed[0, 8] + 1) % 64
    assert torch.allclose(model(ids)[0][0, :8], model(changed)[0][0, :8], atol=1e-6)


def test_gpt_refuses_more_than_its_context():
    model = GPT(tiny_config(context=8))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_tied_embeddings_share_one_table():
    tied = GPT(tiny_config(tie_embeddings=True))
    untied = GPT(tiny_config(tie_embeddings=False))
    assert count_parameters(untied) - count_parameters(tied) == 64 * 32
    assert "cos" not in tied.state_dict()                  # rebuilt, not saved


@pytest.mark.parametrize("changes", [{}, {"tie_embeddings": False}, {"n_layer": 3, "d_model": 48, "n_head": 3},
                                     {"d_ff": 100}, {"vocab_size": 1000}])
def test_parameter_formula_matches_the_model(changes):
    cfg = tiny_config(**changes)
    assert parameter_formula(cfg) == count_parameters(GPT(cfg))


def test_gpt_can_learn_a_batch_by_heart():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = GPT(cfg)
    ids = torch.randint(cfg.vocab_size, (2, cfg.context + 1))
    x, y = ids[:, :-1], ids[:, 1:]
    params = list(model.parameters())
    for _ in range(150):
        _, loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= 0.5 * p.grad                      # plain gradient descent: AdamW is project 4
                p.grad = None
    assert loss.item() < 0.5
