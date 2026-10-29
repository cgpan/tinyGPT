"""Chapter 8's fair comparison, tested without downloading GPT-2."""

import pytest
import torch

import tinygpt
from tinygpt import check, compare


class PairTokenizer:
    """A stand-in for GPT-2's tokenizer: every token is two characters long."""

    def __init__(self, text):
        self.pairs = sorted({text[i : i + 2] for i in range(0, len(text) - 1, 2)})

    def encode(self, text):
        return [self.pairs.index(text[i : i + 2]) for i in range(0, len(text) - 1, 2)]

    def decode(self, ids):
        return "".join(self.pairs[i] for i in ids)


def tiny_gpt(vocab_size):
    torch.manual_seed(0)
    cfg = tinygpt.Config(vocab_size=vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    return tinygpt.GPT(cfg)


TEXT = "To be, or not to be, that is the question: "


def test_token_surprises_cover_the_text_after_the_first_token():
    tok = tinygpt.CharTokenizer.from_text(TEXT)
    tokens, surprises = compare.token_surprises(TEXT, tiny_gpt(tok.vocab_size), tok)
    assert "".join(tokens) == TEXT[1:]                     # longer than block_size: read in windows
    assert len(surprises) == len(tokens) and all(s > 0 for s in surprises)


def test_per_character_divides_by_characters_not_tokens():
    tok = PairTokenizer(TEXT)
    per_token, per_char = compare.per_token_and_per_character(TEXT, tiny_gpt(len(tok.pairs)), tok)
    assert per_char == pytest.approx(per_token / 2)       # every token is two characters

    tok = tinygpt.CharTokenizer.from_text(TEXT)
    per_token, per_char = compare.per_token_and_per_character(TEXT, tiny_gpt(tok.vocab_size), tok)
    assert per_char == pytest.approx(per_token)            # one character per token


def test_per_character_check(capsys):
    check.per_character(lambda tokens, surprises: sum(surprises) / sum(len(t) for t in tokens))
    assert capsys.readouterr().out.startswith("✓")
    check.per_character(lambda tokens, surprises: sum(surprises) / len(tokens))
    assert "per *token*" in capsys.readouterr().out


def test_hugging_face_models_work_too():
    """The same code with a small, untrained GPT-2 from the transformers library."""
    pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([TEXT * 20], vocab_size=300, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=8, n_layer=1, n_head=2, n_embd=16,
                                       bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id))
    tokens, _ = compare.token_surprises(TEXT, model, tok)
    assert "".join(tokens) == TEXT[len(tok.decode(tok.encode(TEXT)[:1])):]
    assert compare.gpt2_sample(model, tok, prompt="To be", length=5, seed=0).startswith("To be")
