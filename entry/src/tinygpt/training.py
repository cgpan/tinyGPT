"""Training: show the model batch after batch of text, and nudge its weights each time."""

import time

import torch

from tinygpt.checkpoint import save_checkpoint
from tinygpt.config import get_device
from tinygpt.data import get_batch, load_data
from tinygpt.model import GPT, count_params


def train_step(model, x, y, optimizer):
    """One step of learning. Returns the loss on this batch."""
    logits, loss = model(x, y)                                   # 1. guess every next character; score the guesses
    optimizer.zero_grad()                                        # 2. clear the previous step's gradients
    loss.backward()                                              # 3. find which way each weight should move
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # 4. never take one enormous step
    optimizer.step()                                             # 5. move every weight a little that way
    return loss.item()


@torch.no_grad()
def estimate_loss(model, ids, cfg, device):
    """Average the loss over `cfg.eval_iters` random batches. One batch is too noisy to trust."""
    model.eval()                                                 # dropout off while measuring
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(ids, cfg.block_size, cfg.batch_size, device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg, model=None, data=None, device=None, save_path=None, step_fn=train_step):
    """Train `model` (a new GPT unless you pass one) and return (model, history).

    `history` has one row per measurement: {"step", "train_loss", "val_loss", "minutes"}.
    Training stops after cfg.max_iters steps, or sooner if it has been running
    for cfg.max_minutes. If `save_path` is given, the trained model is saved there.
    `step_fn` does each step of learning; pass your own to try it in the real loop.
    """
    device = device or get_device()
    if device == "cuda":
        torch.set_float32_matmul_precision("high")               # faster matrix maths on recent NVIDIA GPUs
    torch.manual_seed(cfg.seed)

    if data is None:
        data = load_data()
    tok, train_ids, val_ids = data
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError(f"config says vocab_size={cfg.vocab_size}, but the tokenizer has {tok.vocab_size}")
    if model is None:
        model = GPT(cfg)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    n = count_params(model)
    print(f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n:,}", f"parameters, training on {device}")
    if device == "cpu" and n > 1_000_000:
        print("warning: no GPU found, and a model this size trains slowly on a CPU. See the README.")

    history = []
    start = time.perf_counter()
    for step in range(cfg.max_iters + 1):
        minutes = (time.perf_counter() - start) / 60
        last = step == cfg.max_iters or minutes > cfg.max_minutes
        if step % cfg.eval_interval == 0 or last:
            row = {"step": step,
                   "train_loss": estimate_loss(model, train_ids, cfg, device),
                   "val_loss": estimate_loss(model, val_ids, cfg, device),
                   "minutes": round(minutes, 2)}
            history.append(row)
            print(f"step {step:5d} | train loss {row['train_loss']:.4f} | "
                  f"val loss {row['val_loss']:.4f} | {minutes:5.1f} min")
        if last:
            break
        x, y = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
        step_fn(model, x, y, optimizer)

    if step < cfg.max_iters:
        print(f"stopped early at step {step}: training reached max_minutes = {cfg.max_minutes}")
    if save_path is not None:
        save_checkpoint(save_path, model, tok, cfg, history)
        print(f"saved to {save_path}")
    return model, history
