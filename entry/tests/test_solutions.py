"""Every solution in solutions/ must pass its own check."""

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import tinygpt

SOLUTIONS = sorted((Path(__file__).parents[1] / "solutions").glob("*.md"))


@pytest.fixture(scope="module")
def notebook_names():
    """The names a solution may use because the chapter's notebook defined them earlier."""
    tok, train_ids, val_ids = tinygpt.load_data()
    return {"tinygpt": tinygpt, "np": np, "torch": torch, "text": tinygpt.load_text(),
            "tok": tok, "train_ids": train_ids, "val_ids": val_ids}


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_passes_its_checks(path, notebook_names):
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    names = dict(notebook_names)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for block in blocks:
            exec(block, names)
    assert "✓" in out.getvalue()
    assert "✗" not in out.getvalue()
