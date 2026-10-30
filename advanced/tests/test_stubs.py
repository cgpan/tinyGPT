"""For maintainers: the students' package must be the reference solution with the answers
taken out. Runs only against the solution (PYTHONPATH=solutions); skipped for students."""

import importlib.util
from pathlib import Path

import pytest

import scratchgpt

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(Path(scratchgpt.__file__).parent != ROOT / "solutions" / "scratchgpt",
                                reason="for maintainers: run with PYTHONPATH=solutions")


def make_stubs():
    spec = importlib.util.spec_from_file_location("make_stubs", ROOT / "scripts" / "make_stubs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_student_package_is_the_solution_without_answers():
    for name, text in make_stubs().stubs().items():
        path = ROOT / "src" / "scratchgpt" / name
        assert path.exists() and path.read_text(encoding="utf-8") == text, (
            f"src/scratchgpt/{name} is out of date: run python scripts/make_stubs.py")


def test_every_answer_belongs_to_a_project():
    for path in (ROOT / "solutions" / "scratchgpt").glob("*.py"):
        stubbed = make_stubs().stub(path.read_text(encoding="utf-8"))
        assert "BEGIN SOLUTION" not in stubbed and "END SOLUTION" not in stubbed
