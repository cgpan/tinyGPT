import pytest
import torch

from tinygpt import CharTokenizer

TEXT = "To be, or not to be.\n"


def test_roundtrip_is_lossless():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocabulary_is_the_sorted_distinct_characters():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.chars == sorted(set(TEXT))
    assert tok.vocab_size == len(set(TEXT))


def test_ids_are_in_range():
    tok = CharTokenizer.from_text(TEXT)
    assert all(0 <= i < tok.vocab_size for i in tok.encode(TEXT))


def test_docstring_example():
    tok = CharTokenizer.from_text("hello")
    assert tok.encode("hell") == [1, 0, 2, 2]


def test_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tok.encode("abz")


def test_duplicate_characters_are_rejected():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "b", "a"])


def test_decode_accepts_a_tensor():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(torch.tensor(tok.encode(TEXT))) == TEXT
