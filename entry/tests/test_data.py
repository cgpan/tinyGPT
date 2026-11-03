import torch

from tinygpt import get_batch, load_data, load_text


def test_corpus_is_complete():
    text = load_text()
    assert len(text) == 1_115_394
    assert len(set(text)) == 65
    assert text.startswith("First Citizen:\n")


def test_split_sizes():
    tok, train_ids, val_ids = load_data()
    assert tok.vocab_size == 65
    assert len(train_ids) == 1_003_854
    assert len(val_ids) == 111_540
    assert train_ids.dtype == torch.long


def test_targets_are_inputs_shifted_by_one():
    ids = torch.arange(1000)                  # each id is one more than the id before it
    x, y = get_batch(ids, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_batches_reach_the_end_without_overrunning():
    ids = torch.arange(10)                    # only one window of 9 fits: x = 0..8, y = 1..9
    for _ in range(20):
        x, y = get_batch(ids, block_size=9, batch_size=3)
        assert torch.equal(x[0], torch.arange(9))
        assert torch.equal(y[0], torch.arange(1, 10))
