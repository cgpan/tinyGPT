"""Project 4: AdamW, the learning-rate schedule and gradient clipping, without torch.optim."""

import math

import torch


def parameter_groups(model, weight_decay):
    """Two groups for AdamW: weight decay for parameters with 2 or more dimensions (the
    matrices and the embedding table), none for 1-dimensional ones (the norms' weights).
    Returns [{"params": [...], "weight_decay": weight_decay}, {"params": [...], "weight_decay": 0.0}]."""
    # BEGIN SOLUTION 4
    params = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    # END SOLUTION


class AdamW:
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019).

    For each parameter p with gradient g, at step t (counting from 1):
        p = p - lr * weight_decay * p
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        p = p - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
    m and v start at zero. Parameters without a gradient are skipped.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        # Given: param_groups works like torch.optim's, so the training loop can set lr.
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{"params": params}]
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        self.param_groups = [{**defaults, **g, "params": list(g["params"])} for g in groups]
        self.state = {}                               # parameter -> {"step", "m", "v"}

    def zero_grad(self):
        """Given."""
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self):
        # BEGIN SOLUTION 4
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.get(p)
                if state is None:
                    state = self.state[p] = {"step": 0, "m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                state["step"] += 1
                t, m, v, g = state["step"], state["m"], state["v"], p.grad
                p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (v / (1 - beta2**t)).sqrt_().add_(eps)
                p.addcdiv_(m, denom, value=-lr / (1 - beta1**t))
        # END SOLUTION

    def state_dict(self):
        """Given: everything needed to resume, with parameters named by their position."""
        index = {p: i for i, p in enumerate(p for g in self.param_groups for p in g["params"])}
        return {
            "state": {index[p]: {"step": s["step"], "m": s["m"].clone(), "v": s["v"].clone()}
                      for p, s in self.state.items()},
            "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
        }

    def load_state_dict(self, saved):
        """Given."""
        params = [p for g in self.param_groups for p in g["params"]]
        for group, saved_group in zip(self.param_groups, saved["param_groups"]):
            group.update(saved_group)
        self.state = {}
        for i, s in saved["state"].items():
            p = params[int(i)]
            self.state[p] = {"step": s["step"], "m": s["m"].to(p.device, copy=True),
                             "v": s["v"].to(p.device, copy=True)}


def learning_rate(step, max_lr, min_lr, warmup_steps, max_steps):
    """The learning rate for optimizer step `step` (counting from 0).

    Warmup: rises in a straight line, max_lr * (step + 1) / warmup_steps, for the first
    warmup_steps steps. Then it falls along half a cosine from max_lr to min_lr at max_steps:
        min_lr + (max_lr - min_lr) * (1 + cos(pi * progress)) / 2,
        progress = (step - warmup_steps) / (max_steps - warmup_steps).
    From max_steps on it stays at min_lr.
    """
    # BEGIN SOLUTION 4
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
    # END SOLUTION


@torch.no_grad()
def clip_gradients(params, max_norm):
    """If the gradients' total norm (the square root of the sum of all their squared entries)
    is above max_norm, scale every gradient by max_norm / (norm + 1e-6). Returns the total
    norm before clipping, as a float. Parameters without a gradient are skipped."""
    # BEGIN SOLUTION 4
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    scale = max_norm / (norm + 1e-6)
    if scale < 1:
        for g in grads:
            g.mul_(scale.to(g.dtype))
    return norm.item()
    # END SOLUTION
