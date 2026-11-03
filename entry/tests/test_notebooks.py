"""The chapter notebooks follow one template. These tests keep it that way."""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))
REPO = Path(__file__).resolve().parents[2]          # the repository: entry/ sits inside it
STEPS = ["## 1 · Predict", "## 2 · Run", "## 3 · Investigate", "## 4 · Modify", "## 5 · Make"]


def cells(path, kind):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind]


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 2


def test_every_notebook_starts_with_the_same_setup_cell():
    first_cells = {cells(path, "code")[0] for path in NOTEBOOKS}
    assert len(first_cells) == 1
    assert "tinygpt.print_setup()" in first_cells.pop()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_outputs_are_cleared_so_predictions_are_not_spoiled(path):
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_the_five_steps_appear_in_order(path):
    text = "\n".join(cells(path, "markdown"))
    positions = [text.find(step) for step in STEPS]
    assert -1 not in positions and positions == sorted(positions)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_exactly_one_open_box(path):
    assert sum(source.count("tinygpt.show(") for source in cells(path, "code")) == 1


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_solutions_live_in_their_own_file(path):
    solution = path.resolve().parents[1] / "solutions" / f"{path.stem}.md"
    assert solution.exists()
    # Colab can't follow relative links, so the notebook links to the file on GitHub
    url = f"https://github.com/cgpan/tinyGPT/blob/main/{solution.relative_to(REPO).as_posix()}"
    assert url in "\n".join(cells(path, "markdown")), f"the link to the solution should be {url}"
    assert not any("Solution" in source for source in cells(path, "code"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_model_code_is_defined_in_the_notebook(path):
    for source in cells(path, "code"):
        assert not re.search(r"^\s*class ", source, re.MULTILINE)
        assert "nn.Module" not in source
