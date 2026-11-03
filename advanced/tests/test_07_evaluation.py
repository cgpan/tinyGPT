"""Project 7: bits per byte, baselines, the sentiment probe, memorization."""

import math

import numpy as np
import pytest
import torch
from conftest import tiny_config

from scratchgpt import evaluate
from scratchgpt.model import GPT
from scratchgpt.tokenizer import Tokenizer, count_words, train_bpe


@pytest.fixture(scope="module")
def tok():
    text = "the film was good. the film was bad. a good film, a bad film. " * 20
    return Tokenizer(train_bpe(count_words([text]), vocab_size=256 + 1 + 40))


class Uniform(torch.nn.Module):
    """A 'model' that gives every token the same score."""

    def __init__(self, vocab_size, context):
        super().__init__()
        self.cfg = tiny_config(vocab_size=vocab_size, context=context)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, targets=None, cache=None, hidden=False):
        return torch.zeros(*ids.shape, self.cfg.vocab_size) + self.dummy, None


def test_token_byte_lengths(tok):
    lengths = evaluate.token_byte_lengths(tok)
    assert lengths.shape == (tok.vocab_size,) and lengths.dtype == torch.long
    assert lengths[65] == 1 and lengths[256] == len(tok.vocab[256])
    assert lengths[tok.eot_id] == 0


def test_bits_per_byte_of_uniform_guessing(tok):
    ids = np.array(tok.encode("the film was good, a bad film was the film. " * 5), dtype=np.uint16)
    lengths = evaluate.token_byte_lengths(tok)
    bpb = evaluate.bits_per_byte(Uniform(tok.vocab_size, 16), ids, lengths, torch.device("cpu"), batch_size=3)
    targets = torch.as_tensor(ids[1:].astype("int64"))
    expected = len(targets) * math.log2(tok.vocab_size) / lengths[targets].sum().item()
    assert bpb == pytest.approx(expected, rel=1e-5)


def test_bits_per_byte_skips_end_markers(tok):
    lengths = evaluate.token_byte_lengths(tok)
    text_ids = tok.encode("a good film. " * 6)
    with_markers = np.array(sum(([i, tok.eot_id] for i in text_ids), []), dtype=np.uint16)
    model = Uniform(tok.vocab_size, 8)
    bpb = evaluate.bits_per_byte(model, with_markers, lengths, torch.device("cpu"))
    targets = text_ids[1:]                                    # the first token is never a target
    expected = len(targets) * math.log2(tok.vocab_size) / lengths[torch.tensor(targets)].sum().item()
    assert bpb == pytest.approx(expected, rel=1e-5)


def test_bits_per_byte_respects_max_tokens(tok):
    ids = np.array(tok.encode("the film was good. " * 30), dtype=np.uint16)
    lengths = evaluate.token_byte_lengths(tok)
    model = Uniform(tok.vocab_size, 16)
    assert evaluate.bits_per_byte(model, ids, lengths, torch.device("cpu"), max_tokens=40) > 0


def test_unigram_bits_per_byte_matches_counting_by_hand():
    lengths = torch.tensor([1, 2, 0])                          # token 2 is an end marker
    train = np.array([0, 0, 0, 1, 2], dtype=np.uint16)
    val = np.array([2, 0, 1, 2], dtype=np.uint16)
    p = np.array([4, 2, 2]) / 8                                # add-one counts
    expected = -(math.log(p[0]) + math.log(p[1])) / (math.log(2) * 3)
    assert evaluate.unigram_bits_per_byte(train, val, lengths) == pytest.approx(expected)


def test_gzip_baseline_is_below_eight_bits():
    assert evaluate.gzip_bits_per_byte("the film was good. " * 100) < 1.0


def test_review_features_average_over_real_tokens_only(tok):
    torch.manual_seed(0)
    model = GPT(tiny_config(vocab_size=tok.vocab_size, context=16)).eval()
    texts = ["a good film", "the film was bad, the film was bad"]
    together = evaluate.review_features(model, tok, texts, torch.device("cpu"))
    alone = evaluate.review_features(model, tok, texts[:1], torch.device("cpu"))
    assert together.shape == (2, model.cfg.d_model) and together.dtype == torch.float32
    assert torch.allclose(together[0], alone[0], atol=1e-5)    # the padding changed nothing


def test_bag_of_words_counts_tokens(tok):
    features = evaluate.bag_of_words(tok, ["good good", "bad"])
    assert features.shape == (2, tok.vocab_size)
    assert torch.allclose(features.sum(dim=1), torch.ones(2))


def test_probe_separates_separable_data():
    torch.manual_seed(0)
    x = torch.randn(400, 10)
    y = (x[:, 0] + 0.5 * x[:, 3] > 0).long()
    predict = evaluate.train_probe(x[:300], y[:300])
    accuracy = (predict(x[300:]) == y[300:]).float().mean().item()
    assert accuracy > 0.9


def test_probe_loss_does_not_overflow():
    x = torch.tensor([[1e4], [-1e4]])
    predict = evaluate.train_probe(x, torch.tensor([1, 0]), steps=20)
    assert predict(torch.tensor([[5.0], [-5.0]])).tolist() == [1, 0]


def test_extraction_rate_finds_a_memorized_text():
    torch.manual_seed(0)
    cfg = tiny_config(context=16, d_model=64, n_head=4)
    model = GPT(cfg)
    doc = torch.randint(64, (12,)).tolist()
    ids = torch.tensor([doc])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)   # tests may use torch.optim
    for _ in range(200):
        _, loss = model(ids[:, :-1], ids[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    other = torch.randint(64, (12,)).tolist()
    assert evaluate.extraction_rate(model, [doc], torch.device("cpu"), prefix=4, suffix=8) == 1.0
    assert evaluate.extraction_rate(model, [doc, other, [1, 2]], torch.device("cpu"), prefix=4, suffix=8) == 0.5
