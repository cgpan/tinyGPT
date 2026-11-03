"""Project 2: byte-level BPE."""

import random
import time
from collections import Counter

import numpy as np
import pytest

from scratchgpt import tokenizer as bpe
from scratchgpt.tokenizer import EOT, Tokenizer


def made_up_text(n_words=20000, seed=0):
    """Text with a few thousand different made-up words, common ones much more common."""
    rng = random.Random(seed)
    vocabulary = ["".join(rng.choice("etaoinshrdlucmfwyp") for _ in range(rng.randint(2, 9)))
                  for _ in range(3000)]
    weights = [1 / (rank + 1) for rank in range(len(vocabulary))]
    return " ".join(rng.choices(vocabulary, weights, k=n_words))


def naive_bpe(word_counts, num_merges):
    """The definition of training, written as simply as possible (and slowly)."""
    words = {w: list(w.encode("utf-8")) for w in word_counts}
    merges = []
    for step in range(num_merges):
        counts = Counter()
        for w, ids in words.items():
            for pair in zip(ids, ids[1:]):
                counts[pair] += word_counts[w]
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        merges.append(best)
        words = {w: bpe.merge(ids, best, 256 + step) for w, ids in words.items()}
    return merges


# --- pieces ----------------------------------------------------------------------------

def test_pretokenize_cuts_text_into_pieces_that_join_back():
    text = "Hello, world! It's 2024... café  über_cool 🎬\n\nThe end."
    pieces = bpe.pretokenize(text)
    assert "".join(pieces) == text
    assert pieces[:4] == ["Hello", ",", " world", "!"]
    assert "'s" in pieces and " 2024" in pieces and " café" in pieces


def test_merge_replaces_pairs_left_to_right_without_overlap():
    assert bpe.merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert bpe.merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert bpe.merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert bpe.merge([5], (1, 2), 9) == [5]


# --- training --------------------------------------------------------------------------

def test_train_bpe_merges_the_most_frequent_pair_first():
    # 'ab' occurs 2 + 3 = 5 times, 'aa' 2 * 2 = 4 times
    merges = bpe.train_bpe({"aaab": 2, "ab": 3}, vocab_size=300)
    assert merges[0] == (97, 98)


def test_train_bpe_breaks_ties_by_the_larger_pair():
    # after (a, b) -> 256: 'aaab' is [a, a, 256]; (a, a) and (a, 256) both occur twice
    merges = bpe.train_bpe({"aaab": 2, "ab": 3}, vocab_size=300)
    assert merges == [(97, 98), (97, 256), (97, 257)]      # and then nothing is left to merge


def test_train_bpe_fills_the_vocabulary_exactly():
    words = bpe.count_words([made_up_text(2000)])
    merges = bpe.train_bpe(words, vocab_size=256 + 1 + 50)
    assert len(merges) == 50
    assert Tokenizer(merges).vocab_size == 256 + 50 + 1


def test_train_bpe_agrees_with_the_simple_definition():
    words = bpe.count_words([made_up_text(3000, seed=1)])
    assert bpe.train_bpe(words, vocab_size=256 + 1 + 120) == naive_bpe(words, 120)


def test_train_bpe_handles_any_unicode():
    merges = bpe.train_bpe(bpe.count_words(["日本語 日本語 日本 🎬🎬 🎬🎬"]), vocab_size=270)
    tok = Tokenizer(merges)
    assert tok.decode(tok.encode("日本語 🎬🎬")) == "日本語 🎬🎬"


def test_train_bpe_is_fast_enough():
    words = bpe.count_words([made_up_text(50000, seed=2)])
    start = time.perf_counter()
    bpe.train_bpe(words, vocab_size=256 + 1 + 500)
    assert time.perf_counter() - start < 60, "500 merges took over a minute: see the stretch goal"


# --- encoding and decoding -------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    return Tokenizer(bpe.train_bpe(bpe.count_words([made_up_text(5000, seed=3)]), vocab_size=256 + 1 + 200))


def test_vocabulary_is_bytes_then_merges_then_special_tokens(tok):
    assert tok.vocab[65] == b"A"
    assert tok.vocab[256] == tok.vocab[tok.merges[0][0]] + tok.vocab[tok.merges[0][1]]
    assert tok.eot_id == tok.vocab_size - 1 and tok.vocab[tok.eot_id] == EOT.encode()


@pytest.mark.parametrize("text", ["", "plain words here", "Ünïcödé, 中文, emoji 🎬🍿!",
                                  "tabs\tand\nnewlines\n\n  spaces", "it's 100% <b>html</b>"])
def test_decode_undoes_encode(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_encode_applies_merges_earliest_first():
    tok = Tokenizer([(98, 99), (97, 98)])          # first 'bc' -> 256, then 'ab' -> 257
    assert tok.encode_piece("abc") == [97, 256]    # not [257, 99]: 'bc' was learned first
    assert tok.encode_piece("ab") == [257]


def test_encode_works_piece_by_piece():
    tok = Tokenizer([(97, 98), (97, 256), (97, 257)])
    assert tok.encode("aaab") == [258]
    assert tok.encode("ab ab") == [256, 32, 256]   # ' ab' is a piece of its own: no merge for ' a'


def test_encode_compresses_text(tok):
    text = made_up_text(500, seed=3)
    assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_special_tokens_become_one_id_only_when_allowed(tok):
    ids = tok.encode(f"one{EOT}two")
    assert ids.count(tok.eot_id) == 1
    plain = tok.encode(f"one{EOT}two", allow_special=False)
    assert tok.eot_id not in plain
    assert tok.decode(plain) == f"one{EOT}two"


def test_decode_replaces_invalid_bytes():
    assert Tokenizer([]).decode([0xE6, 0x97]) == "�"   # half of a three-byte character


def test_save_and_load_give_the_same_tokenizer(tok, tmp_path):
    tok.save(tmp_path / "tokenizer.json")
    again = Tokenizer.load(tmp_path / "tokenizer.json")
    text = made_up_text(200, seed=9)
    assert again.encode(text) == tok.encode(text)


def test_encode_documents_writes_16_bit_ids_with_end_markers(tok, tmp_path):
    stats = bpe.encode_documents(tok, ["first doc", "second"], tmp_path / "x.bin")
    ids = np.fromfile(tmp_path / "x.bin", dtype=np.uint16)
    assert stats == {"documents": 2, "tokens": len(ids), "bytes": len("first doc") + len("second")}
    assert ids[-1] == tok.eot_id and list(ids).count(tok.eot_id) == 2
    assert tok.decode([i for i in ids if i != tok.eot_id]) == "first docsecond"
