import math

import numpy as np

from tinygpt import CountBigram

IDS = [0, 1, 0, 1, 1, 2, 0]


def test_counts_neighbouring_pairs():
    counts = CountBigram(3).fit(IDS).counts
    assert counts[0, 1] == 2
    assert counts[1, 0] == counts[1, 1] == counts[1, 2] == 1
    assert counts[2, 0] == 1
    assert counts.sum() == len(IDS) - 1


def test_each_row_of_probabilities_sums_to_one():
    probs = CountBigram(3).fit(IDS).probs()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_with_no_counts_every_character_is_equally_likely():
    assert math.isclose(CountBigram(65).loss(IDS), math.log(65))


def test_loss_matches_a_hand_calculation():
    # with alpha=0: a 0 is always followed by a 1, so p(1 | 0) = 1 and costs -log(1) = 0;
    # a 1 is followed by 0, 1 or 2 once each, so p(0 | 1) = 1/3 and costs log(3)
    model = CountBigram(3, alpha=0.0).fit(IDS)
    assert math.isclose(model.loss([0, 1, 0]), (0 + math.log(3)) / 2)


def test_without_smoothing_an_unseen_pair_is_infinitely_surprising():
    model = CountBigram(3, alpha=0.0).fit(IDS)
    with np.errstate(divide="ignore"):
        assert model.loss([0, 0]) == math.inf


def test_generate_gives_valid_ids_and_is_repeatable():
    model = CountBigram(3).fit(IDS)
    out = model.generate(50, seed=1)
    assert len(out) == 51
    assert all(0 <= i < 3 for i in out)
    assert out == model.generate(50, seed=1)
