"""Every project has a brief, a test file and solution notes, and the Colab workspace is
saved without outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ["01_data", "02_tokenizer", "03_model", "04_training", "05_performance", "06_scaling",
            "07_evaluation", "08_release"]


def test_every_project_has_a_brief_tests_and_notes():
    for name in PROJECTS:
        assert (ROOT / "projects" / f"{name}.md").exists(), name
        assert (ROOT / "solutions" / f"{name}.md").exists(), name
        assert (ROOT / "tests" / f"test_{name}.py").exists(), name


def test_briefs_have_the_same_sections():
    for name in PROJECTS:
        text = (ROOT / "projects" / f"{name}.md").read_text(encoding="utf-8")
        for heading in ["## Predict", "## Build", "## Run", "## Investigate", "## Stretch"]:
            assert heading in text, (name, heading)
        assert "**Done when**" in text, name


def test_colab_workspace_has_no_outputs():
    notebook = json.loads((ROOT / "colab.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_colab_workspace_runs_only_scripts_that_exist():
    notebook = json.loads((ROOT / "colab.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        for word in source.split():
            if word.startswith("scripts/") or word.startswith("tests/") or word.startswith("configs/"):
                if word != "configs/final.yaml":                # the student writes this one
                    assert (ROOT / word).exists(), word


def test_colab_workspace_keeps_clear_of_the_entry_tracks_drive_folder():
    # The entry track keeps its trained model in MyDrive/tinyGPT (entry/src/tinygpt/notebook.py).
    # If the workspace cloned into that folder too, it would find it, skip the clone, and fail.
    notebook = json.loads((ROOT / "colab.ipynb").read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][1]["source"])
    folders = [line.split('Path("')[1].split('")')[0] for line in setup.splitlines() if line.startswith("REPO = ")]
    assert folders and folders[0] != "/content/drive/MyDrive/tinyGPT"
    entry_helper = (ROOT.parent / "entry" / "src" / "tinygpt" / "notebook.py").read_text(encoding="utf-8")
    assert '"MyDrive" / "tinyGPT" / "model.pt"' in entry_helper, "the entry track's Drive folder moved: update this test"
