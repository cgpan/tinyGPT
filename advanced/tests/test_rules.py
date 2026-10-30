"""The from-scratch rules: your package may not use PyTorch's ready-made layers, losses or
optimizers, or anyone else's tokenizer or model. This test reads your code and says where
a rule is broken.

Allowed: tensor operations and autograd, and from torch.nn only Module, Parameter,
ModuleList, ModuleDict and init. One exception: perf.py may use torch.nn.functional for
the fused-attention comparison in project 5.
"""

import ast
from pathlib import Path

import pytest

import scratchgpt

PACKAGE = Path(scratchgpt.__file__).parent
ALLOWED_NN = {"Module", "Parameter", "ModuleList", "ModuleDict", "init", "Buffer"}
BANNED_MODULES = ("torch.nn.functional", "torch.optim", "tiktoken", "tokenizers", "transformers",
                  "sentencepiece", "datasets", "sklearn", "minbpe")
BANNED_NAMES = {"softmax", "log_softmax", "logsumexp", "cross_entropy", "scaled_dot_product_attention",
                "clip_grad_norm_", "layer_norm", "rms_norm", "silu", "gelu", "embedding",
                "multi_head_attention_forward"}
EXCEPTIONS = {"perf.py": {"torch.nn.functional", "scaled_dot_product_attention"}}


def dotted(node):
    """'torch.nn.Linear' for the expression torch.nn.Linear, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def problems(path):
    allowed = EXCEPTIONS.get(path.name, set())
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nn_aliases = {a.asname or "nn" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                  and node.module == "torch" for a in node.names if a.name == "nn"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [f"{node.module}.{alias.name}" for alias in node.names]
            if node.module == "torch.nn":
                names += [f"torch.nn.{a.name}" for a in node.names if a.name not in ALLOWED_NN]
        else:
            names = []
        for name in names:
            if name not in allowed and (name.startswith(BANNED_MODULES) or
                                        (name.startswith("torch.nn.") and name.split(".")[2] not in ALLOWED_NN)):
                found.append(f"line {node.lineno}: import of {name}")
        if not isinstance(node, ast.Attribute):
            continue
        name = dotted(node)
        parts = name.split(".") if name else [None, node.attr]
        if node.attr in BANNED_NAMES and node.attr not in allowed and parts[0] not in ("self", "layers"):
            found.append(f"line {node.lineno}: {name or '.' + node.attr}")
        if name and len(parts) >= 3 and parts[:2] == ["torch", "nn"] and parts[2] not in ALLOWED_NN:
            if ".".join(parts[:3]) not in allowed:
                found.append(f"line {node.lineno}: {name}")
        if name and len(parts) >= 2 and parts[0] in nn_aliases and parts[1] not in ALLOWED_NN:
            found.append(f"line {node.lineno}: {name}")
        if name and name.startswith("torch.optim"):
            found.append(f"line {node.lineno}: {name}")
    return sorted(set(found))


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_code_is_written_from_scratch(path):
    broken = problems(path)
    assert not broken, f"{path.name} uses what you should write yourself:\n  " + "\n  ".join(broken)


def test_the_rules_catch_ready_made_parts(tmp_path):
    cheat = tmp_path / "cheat.py"
    cheat.write_text(
        "import torch\nfrom torch import nn\nimport torch.nn.functional as F\nimport tiktoken\n"
        "a = nn.Linear(2, 2)\nb = torch.nn.LayerNorm(2)\nc = torch.softmax(x, -1)\nd = x.softmax(-1)\n"
        "e = torch.optim.AdamW(p)\nf = F.cross_entropy(x, y)\ng = nn.Parameter(x)\n", encoding="utf-8")
    found = problems(cheat)
    for line in (3, 4, 5, 6, 7, 8, 9, 10):
        assert any(p.startswith(f"line {line}:") for p in found), (line, found)
    assert not any(p.startswith("line 11:") for p in found)
