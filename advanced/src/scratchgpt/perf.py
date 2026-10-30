"""Project 5: how much compute and memory training takes, and how much of the GPU you use."""

import torch
from torch.nn import functional as F   # allowed in this file only: fused attention, for comparison

# Approximate peak dense throughput in 16-bit (float32 accumulate), in TFLOPS, from the
# makers' spec sheets. Given. Your real speed is always well below these.
PEAK_TFLOPS = {
    "Tesla T4": 65,
    "NVIDIA L4": 121,
    "NVIDIA A100": 312,
    "NVIDIA H100": 989,
    "NVIDIA GeForce RTX 3060": 51,
    "NVIDIA GeForce RTX 4070": 58,
    "NVIDIA GeForce RTX 4070 SUPER": 71,
    "NVIDIA GeForce RTX 4080": 97,
    "NVIDIA GeForce RTX 4090": 165,
}


def peak_tflops(device_name):
    """The table's entry whose name starts the device's name, longest match first. Given."""
    for name in sorted(PEAK_TFLOPS, key=len, reverse=True):
        if device_name.startswith(name):
            return PEAK_TFLOPS[name]
    return None


def fused_causal_attention(q, k, v):
    """PyTorch's fused attention (FlashAttention on most GPUs), with the same meaning as
    layers.causal_attention, including Tq < Tk with a cache. Given."""
    t_q, t_k = q.shape[-2], k.shape[-2]
    if t_q == t_k:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if t_q == 1:
        return F.scaled_dot_product_attention(q, k, v)
    allowed = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(t_k - t_q)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)


def matmul_parameters(cfg):
    """The parameters used in matrix multiplications for every token: all the blocks' Linear
    weights and the output layer (vocab_size * d_model, tied or not). Not the norms (they are
    elementwise), and not the embedding lookup (it reads one row, without multiplying)."""
    raise NotImplementedError("write this in project 5: see projects/05_performance.md")


def flops_per_token(cfg):
    """Training FLOPs per token: 6 per matmul parameter (2 for the forward pass, 4 for the
    backward), plus attention's own matrix products: 12 * n_layer * d_model * context
    (q k^T and weights @ v cost 2 * context * d_model each per token and layer, forward;
    times 3 for forward and backward). The causal mask is ignored, as is usual."""
    raise NotImplementedError("write this in project 5: see projects/05_performance.md")


def mfu(tokens_per_second, cfg, peak):
    """Model FLOPs utilisation: the fraction of the GPU's peak (in TFLOPS) the training
    actually uses."""
    raise NotImplementedError("write this in project 5: see projects/05_performance.md")


def training_memory(cfg, batch_size, bytes_per_activation=2):
    """A rough estimate of the GPU memory training needs, in bytes, as a dict:

        params       4 bytes per parameter (float32)
        grads        4 bytes per parameter
        optimizer    8 bytes per parameter (AdamW's m and v)
        activations  what the backward pass keeps, for one forward pass of batch_size
        total        the sum

    Activations, per layer and per token: about 34 * d_model numbers (the inputs of each
    Linear, the norms, SwiGLU's pieces) and n_head * context numbers for each of the
    attention scores, the weights and the mask (our attention keeps all three), each
    bytes_per_activation bytes; plus the logits, 4 * vocab_size bytes per token. Use
    parameter_formula for the parameter count.
    """
    raise NotImplementedError("write this in project 5: see projects/05_performance.md")
