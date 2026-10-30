"""Project 4: AdamW, the schedule, clipping, batches, the training loop, checkpoints, sampling."""

import json
import math

import numpy as np
import pytest
import torch
from conftest import tiny_config

from scratchgpt import optim, sampling
from scratchgpt import train as training
from scratchgpt.model import GPT


# --- the optimizer ---------------------------------------------------------------------

def two_copies():
    torch.manual_seed(0)
    a = GPT(tiny_config())
    b = GPT(tiny_config())
    b.load_state_dict(a.state_dict())
    return a, b


def test_adamw_matches_torch():
    ours_model, torch_model = two_copies()
    ours = optim.AdamW(ours_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    theirs = torch.optim.AdamW(torch_model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ids = torch.randint(64, (4, 17))
    for _ in range(5):
        for model, opt in ((ours_model, ours), (torch_model, theirs)):
            _, loss = model(ids[:, :-1], ids[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad()
    for p, q in zip(ours_model.parameters(), torch_model.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_adamw_skips_parameters_without_gradients():
    p = torch.nn.Parameter(torch.ones(3))
    opt = optim.AdamW([p], lr=0.1, weight_decay=0.5)
    opt.step()
    assert torch.equal(p, torch.ones(3))


def test_adamw_state_can_be_saved_and_restored():
    model, copy = two_copies()
    opt = optim.AdamW(model.parameters(), lr=1e-2)
    ids = torch.randint(64, (2, 17))
    model(ids[:, :-1], ids[:, 1:])[1].backward()
    opt.step()
    opt2 = optim.AdamW(copy.parameters(), lr=1e-2)
    copy.load_state_dict(model.state_dict())
    opt2.load_state_dict(opt.state_dict())
    for m, o in ((model, opt), (copy, opt2)):
        o.zero_grad()
        m(ids[:, :-1], ids[:, 1:])[1].backward()
        o.step()
    for p, q in zip(model.parameters(), copy.parameters()):
        assert torch.equal(p, q)


def test_parameter_groups_decay_matrices_only():
    model = GPT(tiny_config())
    decay, no_decay = optim.parameter_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() == 1 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_learning_rate_warms_up_then_follows_a_cosine():
    lr = lambda step: optim.learning_rate(step, max_lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr(0) == pytest.approx(0.1)
    assert lr(4) == pytest.approx(0.5)
    assert lr(9) == pytest.approx(1.0)
    assert lr(10) == pytest.approx(1.0)
    assert lr(60) == pytest.approx(0.55)                    # halfway down
    assert lr(110) == pytest.approx(0.1) and lr(500) == pytest.approx(0.1)
    assert all(lr(s) >= lr(s + 1) for s in range(10, 120))


def test_clip_gradients_matches_torch():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for p, q in zip(a, b):
        p.grad = torch.randn(5, 5) * 3
        q.grad = p.grad.clone()
    norm = optim.clip_gradients(a, 1.0)
    expected = torch.nn.utils.clip_grad_norm_(b, 1.0)
    assert norm == pytest.approx(expected.item(), rel=1e-5)
    for p, q in zip(a, b):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


def test_clip_gradients_leaves_small_gradients_alone():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), 0.1)
    assert optim.clip_gradients([p], 1.0) == pytest.approx(0.2)
    assert torch.allclose(p.grad, torch.full((4,), 0.1))


# --- batches ---------------------------------------------------------------------------

def test_get_batch_targets_are_the_next_tokens():
    tokens = np.arange(1000, dtype=np.uint16)
    x, y = training.get_batch(tokens, 6, 10, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (6, 10) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)


def test_get_batch_follows_the_generator():
    tokens = np.arange(1000, dtype=np.uint16)
    g = torch.Generator().manual_seed(5)
    first = training.get_batch(tokens, 4, 10, g)[0]
    state = g.get_state()
    second = training.get_batch(tokens, 4, 10, g)[0]
    g.set_state(state)
    assert torch.equal(training.get_batch(tokens, 4, 10, g)[0], second)
    assert not torch.equal(first, second)
    starts = torch.randint(1000 - 10, (4,), generator=torch.Generator().manual_seed(5))
    assert torch.equal(first[:, 0], starts)


def test_get_batch_reaches_the_last_token():
    tokens = np.arange(12, dtype=np.uint16)
    seen = set()
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        seen |= set(training.get_batch(tokens, 4, 10, g)[1].flatten().tolist())
    assert 11 in seen


# --- one step, and the loop ------------------------------------------------------------

class Recorder:
    """Stands in for an optimizer: keeps a copy of the gradients it is asked to apply."""

    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]

    def step(self):
        self.grads = [p.grad.clone() for p in self.param_groups[0]["params"]]

    def zero_grad(self):
        for p in self.param_groups[0]["params"]:
            p.grad = None


def test_gradient_accumulation_matches_one_big_batch():
    big_model, small_model = two_copies()
    ids = torch.randint(64, (8, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    off = torch.amp.GradScaler("cpu", enabled=False)
    big, small = Recorder(big_model.parameters()), Recorder(small_model.parameters())
    loss_big, norm_big = training.train_step(big_model, big, [(x, y)], off, torch.device("cpu"), grad_clip=1e9)
    loss_small, norm_small = training.train_step(small_model, small, [(x[:4], y[:4]), (x[4:], y[4:])],
                                                 off, torch.device("cpu"), grad_clip=1e9)
    assert loss_small == pytest.approx(loss_big, abs=1e-5)
    assert norm_small == pytest.approx(norm_big, rel=1e-4)
    for g, h in zip(big.grads, small.grads):
        assert torch.allclose(g, h, atol=1e-6)
    assert all(p.grad is None for p in big_model.parameters())


def test_train_step_clips_before_stepping():
    model, _ = two_copies()
    ids = torch.randint(64, (4, 17))
    rec = Recorder(model.parameters())
    _, norm = training.train_step(model, rec, [(ids[:, :-1], ids[:, 1:])], torch.amp.GradScaler("cpu", enabled=False),
                                  torch.device("cpu"), grad_clip=1e-3)
    assert norm > 1e-3
    assert torch.sqrt(sum((g**2).sum() for g in rec.grads)).item() == pytest.approx(1e-3, rel=1e-3)


def test_train_step_unscales_float16_gradients_before_clipping():
    torch.manual_seed(0)
    model = GPT(tiny_config())
    opt = optim.AdamW(optim.parameter_groups(model, 0.1), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=True)          # what a T4 uses, here on the CPU
    ids = torch.randint(64, (4, 17))
    try:
        losses = [training.train_step(model, opt, [(ids[:, :-1], ids[:, 1:])], scaler, torch.device("cpu"),
                                      torch.float16, grad_clip=1.0) for _ in range(20)]
    except RuntimeError as error:                               # an older PyTorch without float16 on CPUs
        pytest.skip(f"float16 autocast not available on this CPU: {error}")
    assert losses[0][1] < 100, "the norm is 65,536 times too big: unscale before clipping"
    assert losses[-1][0] < losses[0][0]
    assert next(model.parameters()).dtype == torch.float32


def test_estimate_loss_is_repeatable_and_leaves_the_mode(token_dir):
    model = GPT(tiny_config()).train()
    tokens = training.load_tokens(token_dir / "val.bin")
    a = training.estimate_loss(model, tokens, tiny_config(), 3, torch.device("cpu"))
    b = training.estimate_loss(model, tokens, tiny_config(), 3, torch.device("cpu"))
    assert a == b and model.training


def test_train_learns_the_pattern(token_dir, tmp_path):
    cfg = tiny_config(data_dir=str(token_dir), out_dir=str(tmp_path / "run"), max_steps=60, eval_interval=20)
    history = training.train(cfg, log=lambda *_: None)
    assert [row["step"] for row in history] == [20, 40, 60]
    assert history[-1]["val_loss"] < 0.3 and history[-1]["val_loss"] < history[0]["val_loss"]
    assert set(history[0]) == {"step", "train_loss", "val_loss", "lr", "grad_norm", "tokens", "seconds", "tokens_per_second"}
    logged = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged == history
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "last.pt").exists()
    model, saved_cfg = training.load_model(tmp_path / "run" / "best.pt", torch.device("cpu"))
    assert saved_cfg.vocab_size == 64 and not model.training


def test_train_resumes_exactly_where_it_stopped(token_dir, tmp_path):
    settings = dict(data_dir=str(token_dir), max_steps=24, eval_interval=8, checkpoint_interval=8, dropout=0.1)
    straight = training.train(tiny_config(out_dir=str(tmp_path / "a"), **settings), log=lambda *_: None)
    first = training.train(tiny_config(out_dir=str(tmp_path / "b"), **settings), stop_after=11, log=lambda *_: None)
    assert [row["step"] for row in first] == [8]
    rest = training.train(tiny_config(out_dir=str(tmp_path / "b"), **settings), log=lambda *_: None)
    assert [row["step"] for row in rest] == [16, 24]
    for row_a, row_b in zip(straight[1:], rest):
        assert row_a["val_loss"] == row_b["val_loss"] and row_a["train_loss"] == row_b["train_loss"]
    a = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoints_are_written_safely(token_dir, tmp_path):
    cfg = tiny_config(data_dir=str(token_dir), out_dir=str(tmp_path / "run"), max_steps=10)
    training.train(cfg, log=lambda *_: None)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["best.pt", "last.pt", "log.jsonl"]
    saved = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert saved["step"] == 10 and saved["config"]["max_steps"] == 10
    assert {"model", "optimizer", "scaler", "generator", "rng", "best_val"} <= set(saved)


def test_train_stops_when_time_is_up(token_dir, tmp_path):
    cfg = tiny_config(data_dir=str(token_dir), out_dir=str(tmp_path / "run"), max_steps=1000, max_hours=0.0)
    training.train(cfg, log=lambda *_: None)
    assert torch.load(tmp_path / "run" / "last.pt", weights_only=False)["step"] == 1


# --- writing ---------------------------------------------------------------------------

def test_sample_next_greedy_and_top_k():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    assert sampling.sample_next(logits, temperature=0).tolist() == [[1]]
    g = torch.Generator().manual_seed(0)
    draws = {sampling.sample_next(logits, top_k=2, generator=g).item() for _ in range(200)}
    assert draws == {1, 3}


def test_sample_next_top_p_keeps_the_smallest_likely_set():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    g = torch.Generator().manual_seed(0)
    draws = {sampling.sample_next(logits, top_p=0.75, generator=g).item() for _ in range(300)}
    assert draws == {0, 1}                     # 0.5 + 0.3 >= 0.75; 0.15 isn't needed
    assert {sampling.sample_next(logits, top_p=0.01, generator=g).item() for _ in range(50)} == {0}


def test_sample_next_temperature_flattens_or_sharpens():
    logits = torch.tensor([[0.0, 1.0]])
    g = torch.Generator().manual_seed(0)
    hot = sum(sampling.sample_next(logits, temperature=5.0, generator=g).item() for _ in range(2000)) / 2000
    cold = sum(sampling.sample_next(logits, temperature=0.2, generator=g).item() for _ in range(2000)) / 2000
    assert 0.5 < hot < 0.6 and cold > 0.98


def test_generate_continues_and_stops_at_the_end_marker():
    torch.manual_seed(0)
    model = GPT(tiny_config(context=8))
    ids = torch.tensor([[1, 2, 3]])
    out = sampling.generate(model, ids, 20, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 23) and torch.equal(out[:, :3], ids)
    first = out[0, 3].item()
    stopped = sampling.generate(model, ids, 20, eos_id=first, generator=torch.Generator().manual_seed(0))
    assert stopped.shape == (1, 4)


def test_generate_is_watermarked_unless_told_not_to():
    torch.manual_seed(0)
    model = GPT(tiny_config())
    ids = torch.tensor([[5]])
    marked = sampling.generate(model, ids, 30, temperature=0)
    plain = sampling.generate(model, ids, 30, temperature=0, watermark=False)
    assert not torch.equal(marked, plain)
