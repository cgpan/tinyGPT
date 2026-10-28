import torch

from tinygpt import GPT, CharTokenizer, Config, generate, sample


def tiny_model():
    torch.manual_seed(0)
    return GPT(Config(vocab_size=65, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.1))


def test_generate_runs_past_block_size():
    out = generate(tiny_model(), torch.zeros((2, 1), dtype=torch.long), 30)
    assert out.shape == (2, 31)


def test_generate_leaves_training_mode_as_it_found_it():
    model = tiny_model().train()
    generate(model, torch.zeros((1, 1), dtype=torch.long), 5)
    assert model.training


def test_top_k_of_one_always_picks_the_likeliest_character():
    model, start = tiny_model(), torch.zeros((1, 1), dtype=torch.long)
    assert torch.equal(generate(model, start, 20, top_k=1), generate(model, start, 20, top_k=1))


def test_sample_continues_the_prompt():
    tok = CharTokenizer([chr(i) for i in range(32, 97)])     # any 65 characters
    text = sample(tiny_model(), tok, prompt="ROMEO:", length=40, seed=0)
    assert text.startswith("ROMEO:")
    assert len(text) == 46
    assert text == sample(tiny_model(), tok, prompt="ROMEO:", length=40, seed=0)
