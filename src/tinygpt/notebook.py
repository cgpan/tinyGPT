"""Small helpers for the chapter notebooks."""

import inspect
import textwrap
from importlib import metadata
from pathlib import Path

import torch

from tinygpt.config import get_device


def show(obj):
    """Display the source code of a tinygpt function or class: the chapter's open box.

    This is the code that actually runs when you call it, not a copy.
    """
    source = textwrap.dedent(inspect.getsource(obj))
    path = Path(inspect.getsourcefile(obj))
    header = f"# from tinygpt/{path.name}\n\n"
    try:
        from IPython import get_ipython
        from IPython.display import Code, display
    except ImportError:
        get_ipython = None
    if get_ipython is not None and get_ipython() is not None:
        display(Code(header + source, language="python"))
    else:
        print(header + source)


def print_setup():
    """Say which version is installed and which hardware it will run on."""
    device = get_device()
    if device == "cuda":
        hardware = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    elif device == "mps":
        hardware = "Apple Silicon GPU"
    else:
        hardware = "CPU only"
    print(f"tinyGPT {metadata.version('tinygpt')} is ready. PyTorch {torch.__version__}, running on: {hardware}.")
    if device == "cpu":
        print("No GPU found. That's fine for chapters 1 and 2; from chapter 3 on you need one.\n"
              "In Colab: Runtime > Change runtime type > T4 GPU, then run this cell again.")
