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
