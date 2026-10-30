"""Small helpers for the chapter notebooks."""

import inspect
import os
import shutil
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

import torch

from tinygpt.checkpoint import load_checkpoint
from tinygpt.config import get_device, load_config
from tinygpt.training import train

MODEL_PATH = Path("runs/base/model.pt")     # where chapter 5 saves the full model
DRIVE = Path("/content/drive")              # where Colab mounts Google Drive


def show(obj):
    """Display the source code of a tinygpt function or class, or of a file: the chapter's open box.

    This is the code that actually runs when you call it, not a copy.
    """
    if isinstance(obj, (str, Path)):
        source, header = Path(obj).read_text(encoding="utf-8"), f"# {obj}\n\n"
    else:
        obj = inspect.unwrap(obj)                # see through decorators such as @torch.no_grad()
        source = textwrap.dedent(inspect.getsource(obj))
        header = f"# from tinygpt/{Path(inspect.getsourcefile(obj)).name}\n\n"
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
        print("No GPU found. Chapters 1 to 4 work without one (3 and 4 just run more slowly);\n"
              "chapter 5 needs one. In Colab: Runtime > Change runtime type > T4 GPU, then run this cell again.")


def _kept_path():
    """Google Drive on Colab, where files outlive the session; the runs/ folder anywhere else."""
    if "google.colab" not in sys.modules:
        return MODEL_PATH
    if not (DRIVE / "MyDrive").exists():
        from google.colab import drive
        drive.mount(str(DRIVE))                 # Colab asks for permission the first time
    return DRIVE / "MyDrive" / "tinyGPT" / "model.pt"


def keep_model(path=MODEL_PATH):
    """Keep a trained model where the next chapter can find it.

    On Colab, everything on the virtual machine disappears when the session ends, so the
    model is copied to your Google Drive. On your own computer it is already safe on disk.
    """
    target = _kept_path()
    if Path(path).resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(f"Your model is kept in {target}")


def load_kept_model():
    """Load the model chapter 5 kept: returns (model, tokenizer, history).

    If there isn't one, train it first with configs/base.yaml (6 to 8 minutes on a T4).
    """
    path = _kept_path()
    if not path.exists():
        print(f"No model found at {path}, so training one now with configs/base.yaml.")
        train(load_config("configs/base.yaml"), save_path=MODEL_PATH)
        keep_model(MODEL_PATH)
    model, tok, history = load_checkpoint(path)
    last = history[-1]
    print(f"Loaded {path}: trained for {last['step']:,} steps, validation loss {last['val_loss']:.2f}.")
    return model, tok, history


def run_tests(bug=None):
    """Run tinyGPT's tests with pytest, optionally with one deliberate bug planted.

    The tests run in a separate Python process, so a planted bug never reaches the
    notebook. The bugs are listed in tinygpt.bugs.BUGS.
    """
    from tinygpt.bugs import BUGS

    if bug is not None and bug not in BUGS:
        raise ValueError(f"no bug called {bug!r}; choose from: {', '.join(BUGS)}")
    if not Path("tests").is_dir():
        raise FileNotFoundError("No tests/ folder here. Run the setup cell first: it moves into the tinyGPT folder.")
    command = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "--color=no", "-p", "no:cacheprovider", "tests"]
    if bug:
        # these check the bug-planting and the exercise solutions, which need tinyGPT without a bug
        command += ["--ignore=tests/test_bugs.py", "--ignore=tests/test_solutions.py"]
    env = dict(os.environ, TINYGPT_BUG=bug or "", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print(f"--- tests with the {bug} bug planted (leaving out the tests of the bugs and of the solutions) ---"
          if bug else "--- tests on tinyGPT as it is ---")
    print((done.stdout or done.stderr).strip())
