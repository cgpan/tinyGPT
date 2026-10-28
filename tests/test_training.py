import torch

from tinygpt import GPT, BigramLM, Config, load_checkpoint, train, train_step


def tiny(**changes):
    values = dict(vocab_size=65, block_size=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
                  batch_size=4, max_iters=20, eval_interval=10, eval_iters=2)
    values.update(changes)
    return Config(**values)


def test_model_can_memorise_one_batch():
    """If it cannot memorise 4 short sequences, the training wiring is broken."""
    torch.manual_seed(0)
    model = GPT(tiny())
    x, y = torch.randint(0, 65, (4, 8)), torch.randint(0, 65, (4, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = train_step(model, x, y, optimizer)
    for _ in range(100):
        last = train_step(model, x, y, optimizer)
    assert last < first / 4


def test_train_runs_end_to_end_and_records_history():
    model, history = train(tiny(), device="cpu")
    assert [row["step"] for row in history] == [0, 10, 20]
    assert history[-1]["val_loss"] < history[0]["val_loss"]
    assert model.training                                     # left ready for more training


def test_train_stops_at_the_time_limit():
    _, history = train(tiny(max_minutes=0.0), device="cpu")
    assert [row["step"] for row in history] == [0]            # out of time before the first step


def test_train_accepts_another_model():
    cfg = tiny()
    model, history = train(cfg, model=BigramLM(cfg), device="cpu")
    assert isinstance(model, BigramLM)


def test_checkpoint_round_trip(tmp_path):
    cfg = tiny()
    model, history = train(cfg, device="cpu", save_path=tmp_path / "model.pt")
    loaded, tok, loaded_history = load_checkpoint(tmp_path / "model.pt", device="cpu")
    x = torch.randint(0, 65, (2, 8))
    model.eval()
    assert torch.equal(model(x)[0], loaded(x)[0])
    assert tok.vocab_size == 65
    assert loaded_history == history


def test_train_uses_the_step_function_it_is_given():
    calls = []

    def my_step(model, x, y, optimizer):
        calls.append(x.shape)
        return train_step(model, x, y, optimizer)

    train(tiny(), device="cpu", step_fn=my_step)
    assert calls == [(4, 8)] * 20
