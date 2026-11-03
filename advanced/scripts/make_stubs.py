"""For maintainers: write the students' package (src/scratchgpt) from the reference solution
(solutions/scratchgpt). Every block between '# BEGIN SOLUTION n' and '# END SOLUTION' becomes
a NotImplementedError that names project n; everything else is copied as it is.

    python scripts/make_stubs.py           # write src/scratchgpt
    python scripts/make_stubs.py --check   # only say whether src/scratchgpt is up to date
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solutions" / "scratchgpt"
STUBS = ROOT / "src" / "scratchgpt"
BRIEFS = {1: "01_data", 2: "02_tokenizer", 3: "03_model", 4: "04_training", 5: "05_performance",
          6: "06_scaling", 7: "07_evaluation", 8: "08_release"}
BEGIN = re.compile(r"^(\s*)# BEGIN SOLUTION (\d)\s*$")
END = re.compile(r"^\s*# END SOLUTION\s*$")


def stub(source):
    out, inside = [], False
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        begin = BEGIN.match(line)
        if begin:
            if inside:
                raise ValueError(f"line {number}: BEGIN SOLUTION inside another")
            indent, project = begin.group(1), int(begin.group(2))
            out.append(f'{indent}raise NotImplementedError("write this in project {project}: '
                       f'see projects/{BRIEFS[project]}.md")\n')
            inside = True
        elif END.match(line):
            if not inside:
                raise ValueError(f"line {number}: END SOLUTION without BEGIN")
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("a BEGIN SOLUTION is never ended")
    return "".join(out)


def stubs():
    """{file name: stub text} for every module of the solution."""
    return {path.name: stub(path.read_text(encoding="utf-8")) for path in sorted(SOLUTION.glob("*.py"))}


def main():
    wanted = stubs()
    if "--check" in sys.argv:
        stale = [name for name, text in wanted.items()
                 if not (STUBS / name).exists() or (STUBS / name).read_text(encoding="utf-8") != text]
        print("up to date" if not stale else "out of date: " + ", ".join(stale))
        sys.exit(1 if stale else 0)
    STUBS.mkdir(parents=True, exist_ok=True)
    for name, text in wanted.items():
        (STUBS / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(wanted)} files to {STUBS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
