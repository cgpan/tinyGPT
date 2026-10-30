"""Project 5: compute, memory, speed, and the KV cache."""

import pytest
import torch
from conftest import tiny_config

from scratchgpt import layers, perf, sampling
from scratchgpt.config import Config
from scratchgpt.model import GPT


def test_matmul_parameters_leave_out_norms_and_the_lookup():
    cfg = Config(vocab_size=1000, context=64, n_layer=2, n_head=2, d_model=64, d_ff=128)
    assert perf.matmul_parameters(cfg) == 2 * (4 * 64 * 64 + 3 * 64 * 128) + 1000 * 64


def test_flops_per_token_counts_matmuls_and_attention():
    cfg = Config(vocab_size=1000, context=64, n_layer=2, n_head=2, d_model=64, d_ff=128)
    assert perf.flops_per_token(cfg) == 6 * perf.matmul_parameters(cfg) + 12 * 2 * 64 * 64


def test_mfu_is_the_fraction_of_peak():
    cfg = Config()
    tokens_per_second = 1e12 / perf.flops_per_token(cfg)      # exactly one TFLOP per second
    assert perf.mfu(tokens_per_second, cfg, peak=100) == pytest.approx(0.01)


def test_training_memory_counts_16_bytes_per_parameter_before_activations():
    from scratchgpt.model import parameter_formula
    cfg = Config()
    memory = perf.training_memory(cfg, batch_size=16)
    n = parameter_formula(cfg)
    assert (memory["params"], memory["grads"], memory["optimizer"]) == (4 * n, 4 * n, 8 * n)
    assert memory["total"] == sum(v for k, v in memory.items() if k != "total")
    double_batch = perf.training_memory(cfg, batch_size=32)
    assert double_batch["activations"] == 2 * memory["activations"]
    longer = perf.training_memory(Config(context=1024), batch_size=16)
    assert longer["activations"] > 2 * memory["activations"]    # attention grows with context squared


def test_peak_tflops_finds_the_card():
    assert perf.peak_tflops("NVIDIA GeForce RTX 4070 SUPER") == 71
    assert perf.peak_tflops("NVIDIA GeForce RTX 4070") == 58
    assert perf.peak_tflops("NVIDIA A100-SXM4-40GB") == 312
    assert perf.peak_tflops("Some other card") is None


@pytest.mark.parametrize("t_q", [7, 3, 1])
def test_fused_attention_means_the_same_as_yours(t_q):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 2, t_q, 8), torch.randn(2, 2, 7, 8), torch.randn(2, 2, 7, 8)
    assert torch.allclose(perf.fused_causal_attention(q, k, v), layers.causal_attention(q, k, v), atol=1e-5)


def test_the_model_can_use_fused_attention():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    fused = GPT(tiny_config(fused_attention=True)).eval()
    fused.load_state_dict(model.state_dict())
    ids = torch.randint(64, (2, 16))
    assert torch.allclose(model(ids)[0], fused(ids)[0], atol=1e-5)


# --- the KV cache ----------------------------------------------------------------------

def test_kv_cache_grows_per_layer():
    cache = sampling.KVCache(n_layer=2)
    assert cache.length(0) == 0 and cache.length(1) == 0
    k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    all_k, all_v = cache.update(0, k, v)
    assert torch.equal(all_k, k) and cache.length(0) == 3 and cache.length(1) == 0
    k2 = torch.randn(1, 2, 1, 4)
    all_k, all_v = cache.update(0, k2, k2)
    assert all_k.shape == (1, 2, 4, 4) and torch.equal(all_k[:, :, 3:], k2)


def test_cached_model_gives_the_same_logits():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    ids = torch.randint(64, (2, 10))
    full = model(ids)[0]
    cache = sampling.KVCache(model.cfg.n_layer)
    first = model(ids[:, :6], cache=cache)[0]
    rest = [model(ids[:, t : t + 1], cache=cache)[0] for t in range(6, 10)]
    assert torch.allclose(torch.cat([first] + rest, dim=1), full, atol=1e-5)


def test_cache_refuses_to_go_past_the_context():
    model = GPT(tiny_config(context=8)).eval()
    cache = sampling.KVCache(model.cfg.n_layer)
    model(torch.zeros(1, 8, dtype=torch.long), cache=cache)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, dtype=torch.long), cache=cache)


def test_generate_with_the_cache_writes_the_same_greedy_text():
    torch.manual_seed(0)
    model = GPT(tiny_config(context=16))
    ids = torch.randint(64, (2, 5))
    plain = sampling.generate(model, ids, 9, temperature=0, use_cache=False)
    cached = sampling.generate(model, ids, 9, temperature=0, use_cache=True)
    assert torch.equal(plain, cached)


def test_generate_with_the_cache_runs_past_the_context():
    torch.manual_seed(0)
    model = GPT(tiny_config(context=8))
    out = sampling.generate(model, torch.zeros(1, 3, dtype=torch.long), 20, temperature=0, use_cache=True)
    assert out.shape == (1, 23)
