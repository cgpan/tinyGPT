from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tinygpt
from tinygpt import check


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_show_prints_the_real_source(capsys):
    tinygpt.show(tinygpt.CharTokenizer)
    out = capsys.readouterr().out
    assert out.startswith("# from tinygpt/tokenizer.py")
    assert "class CharTokenizer" in out


def test_show_names_the_right_file_for_a_decorated_function(capsys):
    tinygpt.show(tinygpt.generate)                  # wrapped by @torch.no_grad()
    out = capsys.readouterr().out
    assert out.startswith("# from tinygpt/sampling.py")
    assert "@torch.no_grad()" in out


def test_print_setup_names_the_hardware(capsys):
    tinygpt.print_setup()
    assert "tinyGPT" in capsys.readouterr().out


# --- the Make checks ---------------------------------------------------------

def reference_tokenizer():
    tok = tinygpt.CharTokenizer.from_text(tinygpt.load_text())
    return tok.encode, tok.decode


def test_tokenizer_check_accepts_a_correct_tokenizer(capsys):
    check.tokenizer(*reference_tokenizer())
    assert capsys.readouterr().out.startswith("✓")


def test_tokenizer_check_explains_an_empty_answer(capsys):
    check.tokenizer(lambda s: [], lambda ids: "")
    assert "should give 19 numbers" in capsys.readouterr().out


def test_tokenizer_check_spots_an_unsorted_vocabulary(capsys):
    chars = list(reversed(sorted(set(tinygpt.load_text()))))
    stoi = {ch: i for i, ch in enumerate(chars)}
    check.tokenizer(lambda s: [stoi[ch] for ch in s], lambda ids: "".join(chars[i] for i in ids))
    assert "sorted(set(text))" in capsys.readouterr().out


def test_tokenizer_check_reports_a_crash_instead_of_raising(capsys):
    def my_encode(s):
        return {}["x"]
    check.tokenizer(my_encode, lambda ids: "")
    assert "raised KeyError" in capsys.readouterr().out


def correct_count_pairs(ids, vocab_size):
    counts = np.zeros((vocab_size, vocab_size))
    for a, b in zip(ids[:-1], ids[1:]):
        counts[a, b] += 1
    return counts


def test_count_pairs_check(capsys):
    check.count_pairs(correct_count_pairs)
    assert capsys.readouterr().out.startswith("✓")
    check.count_pairs(lambda ids, v: np.zeros((v, v)))
    assert "the pair (0 → 1) appears 2 times, but your table says 0" in capsys.readouterr().out
    check.count_pairs(lambda ids, v: None)
    assert "Did you return the table?" in capsys.readouterr().out


def test_count_triples_check(capsys):
    def count_triples(ids, v):
        counts = np.zeros((v, v, v))
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            counts[a, b, c] += 1
        return counts
    check.count_triples(count_triples)
    assert capsys.readouterr().out.startswith("✓")
    check.count_triples(lambda ids, v: np.zeros((v, v, v)))
    assert "appears 1 time," in capsys.readouterr().out


# --- plots ---------------------------------------------------------------------

def test_scoreboard_also_prints_its_numbers(capsys):
    tinygpt.scoreboard({"knowing frequencies": 3.35})
    out = capsys.readouterr().out
    assert "3.35  knowing frequencies" in out
    assert "4.17  knowing nothing" in out


def test_char_counts_and_heatmap_draw():
    tinygpt.char_counts("hello, world\n")
    tok = tinygpt.CharTokenizer.from_text("abc")
    tinygpt.bigram_heatmap(tinygpt.CountBigram(3).fit([0, 1, 2, 0]).counts, tok)
    assert plt.get_fignums()


def test_plot_history_prints_its_numbers(capsys):
    history = [{"step": 0, "train_loss": 4.2, "val_loss": 4.2}, {"step": 10, "train_loss": 2.0, "val_loss": 2.5}]
    tinygpt.plot_history(history)
    assert "validation" in capsys.readouterr().out
    tinygpt.plot_history({"lr 0.01": history, "lr 0.1": history})
    out = capsys.readouterr().out
    assert "lr 0.01" in out and "2.500" in out


def test_attention_map_reports_where_the_head_looks(capsys):
    cfg = tinygpt.Config(vocab_size=65, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    tok = tinygpt.CharTokenizer.from_text(tinygpt.load_text())
    model = tinygpt.GPT(cfg).train()
    tinygpt.attention_map(model, tok, "To be, or not", head=1)
    assert "Reading the last character, 't'" in capsys.readouterr().out
    assert model.training


# --- the chapter 3 and 4 checks ---------------------------------------------------

def test_train_step_check_accepts_the_real_one(capsys):
    check.train_step(tinygpt.train_step)
    assert capsys.readouterr().out.startswith("✓")


def test_train_step_check_spots_missing_zero_grad(capsys):
    def forgetful(model, x, y, optimizer):
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
        return loss.item()
    check.train_step(forgetful)
    assert "optimizer.zero_grad()" in capsys.readouterr().out


def test_train_step_check_spots_a_missing_step(capsys):
    def lazy(model, x, y, optimizer):
        return model(x, y)[1].item()
    check.train_step(lazy)
    assert "optimizer.step()" in capsys.readouterr().out


def test_attention_check(capsys):
    import math
    import torch
    from torch.nn import functional as F

    def attention(q, k, v, mask=True):
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        if mask:
            T = q.shape[1]
            scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v

    check.attention(attention)
    assert capsys.readouterr().out.startswith("✓")
    check.attention(lambda q, k, v: attention(q, k, v, mask=False))
    assert "can see the future" in capsys.readouterr().out
    check.attention(lambda q, k, v: v)
    assert "doesn't match" in capsys.readouterr().out


def test_param_count_check(capsys):
    def formula(vocab_size, block_size, n_layer, n_embd):
        d = n_embd
        per_block = 12 * d * d + 10 * d
        return vocab_size * d + block_size * d + n_layer * per_block + 2 * d + d * vocab_size + vocab_size
    check.param_count(formula)
    assert capsys.readouterr().out.startswith("✓")
    check.param_count(lambda vocab_size, block_size, n_layer, n_embd: 12 * n_layer * n_embd ** 2)
    assert "off by" in capsys.readouterr().out


def test_longest_copy():
    assert tinygpt.longest_copy("xx my lord, yy", "O my lord, sir") == " my lord, "
    assert tinygpt.longest_copy("zzz", "abc") == ""


def test_greedy_check(capsys):
    import torch

    def greedy(model, idx, n):
        for _ in range(n):
            logits, _ = model(idx[:, -model.cfg.block_size:])
            idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        return idx

    def uncropped(model, idx, n):
        for _ in range(n):
            logits, _ = model(idx)
            idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        return idx

    check.greedy(greedy)
    assert capsys.readouterr().out.startswith("✓")
    check.greedy(uncropped)
    assert "last block_size characters" in capsys.readouterr().out
    check.greedy(lambda model, idx, n: idx)
    assert "should have shape (2, 15)" in capsys.readouterr().out


# --- keeping the model between chapters ---------------------------------------------

def save_tiny_model(path):
    cfg = tinygpt.Config(vocab_size=65, block_size=8, n_layer=1, n_head=2, n_embd=16)
    tok = tinygpt.CharTokenizer.from_text(tinygpt.load_text())
    history = [{"step": 5, "train_loss": 3.0, "val_loss": 3.1, "minutes": 0.1}]
    tinygpt.save_checkpoint(path, tinygpt.GPT(cfg), tok, cfg, history)


def test_on_your_own_computer_the_model_stays_in_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    save_tiny_model(tmp_path / "runs/base/model.pt")
    tinygpt.keep_model()
    assert str(Path("runs") / "base" / "model.pt") in capsys.readouterr().out
    model, tok, history = tinygpt.load_kept_model()
    assert history[-1]["step"] == 5 and tok.vocab_size == 65


def test_on_colab_the_model_goes_to_google_drive(tmp_path, monkeypatch, capsys):
    import sys
    import types

    from tinygpt import notebook

    drive_root = tmp_path / "drive"
    mounted = []
    fake_drive = types.SimpleNamespace(mount=lambda where: (mounted.append(where), (drive_root / "MyDrive").mkdir(parents=True)))
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(drive=fake_drive))
    monkeypatch.setattr(notebook, "DRIVE", drive_root)
    monkeypatch.chdir(tmp_path)
    save_tiny_model(tmp_path / "runs/base/model.pt")

    tinygpt.keep_model()
    assert mounted == [str(drive_root)]
    assert (drive_root / "MyDrive/tinyGPT/model.pt").exists()
    (tmp_path / "runs/base/model.pt").unlink()             # a new Colab session: the local copy is gone
    model, tok, history = tinygpt.load_kept_model()
    assert history[-1]["val_loss"] == 3.1
    assert mounted == [str(drive_root)]                     # already mounted: not asked twice
