"""Which device to train on, and in which number format. Given: nothing to write here."""

import contextlib

import torch


def get_device(setting="auto"):
    """The fastest device here: an NVIDIA GPU (cuda), an Apple GPU (mps), else the CPU."""
    if setting != "auto":
        return torch.device(setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device, setting="auto"):
    """The 16-bit format for mixed precision, or None for plain float32.

    auto: bfloat16 on NVIDIA GPUs that have it (RTX 30 series and later, A100, L4), float16 on
    older ones (Colab's T4), and float32 on Apple GPUs and CPUs.
    """
    if setting == "float32":
        return None
    if setting in ("bfloat16", "float16"):
        return getattr(torch, setting)
    if setting != "auto":
        raise ValueError(f"dtype must be auto, float32, bfloat16 or float16, not {setting!r}")
    if device.type == "cuda":
        # bfloat16 runs natively from compute capability 8 (RTX 30 series, A100, L4). Older
        # GPUs such as the T4 only emulate it, slowly, though is_bf16_supported() says yes.
        native = torch.cuda.get_device_capability(device)[0] >= 8
        return torch.bfloat16 if native else torch.float16
    return None


def autocast(device, dtype):
    """Run the code inside in mixed precision (or do nothing when dtype is None)."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def describe(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple GPU (mps)"
    return "CPU"


def synchronize(device):
    """Wait for the GPU to finish, so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
