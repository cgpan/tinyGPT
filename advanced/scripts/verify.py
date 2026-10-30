"""For instructors: run the whole advanced track on this machine with the reference
solution, and write a report.

    python scripts/verify.py              # everything: about 1 to 2 hours with a GPU
    python scripts/verify.py --quick      # the pipeline on a 2 MB slice, tiny model: minutes
    python scripts/verify.py --sweep      # also the scaling sweep (adds roughly 10 to 30 minutes)

Downloads IMDB (84 MB) into data/, trains into runs/verify-*, and writes
runs/report-advanced-<computer name>.md. Nothing in the repository is changed.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "solutions"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(*command, timeout=None):
    """Run a command in the advanced folder with the reference solution; returns (ok, output, minutes)."""
    start = time.perf_counter()
    print("$", " ".join(command), flush=True)
    done = subprocess.run([sys.executable, *command], cwd=ROOT, env=ENV, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    output = (done.stdout + done.stderr).strip()
    print("\n".join(output.splitlines()[-6:]), flush=True)
    return done.returncode == 0, output, (time.perf_counter() - start) / 60


def environment():
    code = ("import torch, platform; from scratchgpt.device import get_device, describe, amp_dtype;"
            "d = get_device(); print(f'Python {platform.python_version()}, PyTorch {torch.__version__}, '"
            "f'{describe(d)}, mixed precision: {amp_dtype(d)}')")
    ok, out, _ = run("-c", code)
    return out if ok else f"could not check: {out}"


def quick_slice(source, dest, megabytes=2):
    """A small copy of the prepared data, for --quick."""
    dest.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        lines, size = [], 0
        with open(source / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                lines.append(line)
                size += len(line)
                if size > megabytes * 1e6 * (1 if split == "train" else 0.2):
                    break
        (dest / f"{split}.jsonl").write_text("".join(lines), encoding="utf-8")
    shutil.copy(source / "stats.json", dest / "stats.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    sections, summary = [], []

    def record(name, ok, output, minutes, lines=25):
        summary.append((name, "OK" if ok else "FAILED", minutes))
        tail = "\n".join(output.splitlines()[-lines:])
        sections.append(f"### {name}: {'OK' if ok else 'FAILED'}, {minutes:.1f} min\n\n```\n{tail}\n```\n")
        return ok

    env = environment()
    print(env)
    record("tests (reference solution)", *run("-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no"), lines=8)
    record("student stubs up to date", *run("scripts/make_stubs.py", "--check"), lines=3)

    if not record("prepare data", *run("scripts/prepare_data.py"), lines=20):
        return write_report(env, summary, sections, args)
    data = "data/imdb"
    if args.quick:
        quick_slice(ROOT / "data" / "imdb", ROOT / "data" / "quick")
        data = "data/quick"
    vocab = "2048" if args.quick else "8192"
    if not record("tokenizer", *run("scripts/train_tokenizer.py", "--data", data, "--vocab-size", vocab), lines=20):
        return write_report(env, summary, sections, args)

    configs = ["configs/tiny.yaml"] if args.quick else ["configs/tiny.yaml", "configs/small.yaml", "configs/base.yaml"]
    record("benchmark", *run("scripts/benchmark.py", *configs, f"data_dir={data}"), lines=10)

    runs = [("tiny", "configs/tiny.yaml")] if args.quick else [("small", "configs/small.yaml"), ("base", "configs/base.yaml")]
    for name, config in runs:
        out_dir = f"runs/verify-{name}"
        shutil.rmtree(ROOT / out_dir, ignore_errors=True)
        if not record(f"train {name}", *run("scripts/train.py", config, f"data_dir={data}", f"out_dir={out_dir}"), lines=30):
            continue
        record(f"samples from {name}", *run("scripts/sample.py", out_dir, "--n", "5", "--seed", "0", "--tokens", "200"), lines=80)
        record(f"samples from {name}, prompted", *run("scripts/sample.py", out_dir, "--n", "3", "--seed", "10",
                                                     "--prompt", "I watched this movie last night and"), lines=50)
    final = f"runs/verify-{runs[-1][0]}"
    if (ROOT / final / "best.pt").exists():
        record("evaluate", *run("scripts/evaluate.py", final), lines=12)
        record("watermark", *run("scripts/watermark_check.py", final), lines=10)
        record("model card", *run("scripts/model_card.py", final, "--force"), lines=3)
        card = ROOT / final / "MODEL_CARD.md"
        if card.exists():
            sections.append("### the model card it wrote\n\n" + card.read_text(encoding="utf-8") + "\n")
    if args.sweep and not args.quick:
        shutil.rmtree(ROOT / "runs" / "verify-sweep", ignore_errors=True)
        record("scaling sweep", *run("scripts/sweep.py", "configs/small.yaml", f"data_dir={data}",
                                     "--out", "runs/verify-sweep"), lines=14)
    write_report(env, summary, sections, args)


def write_report(env, summary, sections, args):
    name = platform.node().split(".")[0] or "this-computer"
    path = ROOT / "runs" / f"report-advanced-{name}.md"
    path.parent.mkdir(exist_ok=True)
    rows = "\n".join(f"| {step} | {result} | {minutes:.1f} |" for step, result, minutes in summary)
    mode = "quick" if args.quick else "full"
    path.write_text(f"# Advanced track: verification on {name} ({mode})\n\n{env}\n\n"
                    f"| step | result | minutes |\n|---|---|---|\n{rows}\n\n" + "\n".join(sections), encoding="utf-8")
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
