"""Check that tinyGPT works on this machine, and write a report.

    python scripts/verify.py            # everything: about 20 to 40 minutes with a GPU
    python scripts/verify.py --quick    # tests, benchmark, the tiny run and chapter 1 only

Runs the test suite, the benchmark, the three training configs, and every chapter
notebook from top to bottom (chapter 8 downloads GPT-2, about 500 MB). The report goes
to runs/report-<computer name>.md. Nothing in the repository is changed.
"""

import argparse
import contextlib
import io
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

import tinygpt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def run(*command):
    """Run a command in the repository; return its output (stdout and stderr together)."""
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (done.stdout + done.stderr).strip()


def say(message):
    print(message.encode("ascii", "replace").decode(), flush=True)     # safe on any console


def environment():
    device = tinygpt.get_device()
    hardware = {"cuda": lambda: torch.cuda.get_device_name(0), "mps": lambda: "Apple Silicon GPU (MPS)"}
    return [
        f"- date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- computer: {platform.node()} ({platform.platform()}, {platform.machine()})",
        f"- processor: {platform.processor() or 'unknown'}",
        f"- Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}",
        f"- device: {device}, {hardware.get(device, lambda: 'CPU only')()}",
        f"- tinyGPT commit: {run('git', 'rev-parse', '--short', 'HEAD')}",
    ]


def training(names):
    lines = ["| config | parameters | steps | minutes | train loss | val loss | stopped early |",
             "|---|---|---|---|---|---|---|"]
    for name in names:
        say(f"training configs/{name}.yaml ...")
        cfg = tinygpt.load_config(ROOT / "configs" / f"{name}.yaml")
        out = io.StringIO()
        start = time.time()
        with contextlib.redirect_stdout(out):
            model, history = tinygpt.train(cfg, save_path=RUNS / name / "model.pt")
        minutes = (time.time() - start) / 60
        last = history[-1]
        lines.append(f"| {name} | {tinygpt.count_params(model):,} | {last['step']:,} | {minutes:.1f} | "
                     f"{last['train_loss']:.4f} | {last['val_loss']:.4f} | {last['step'] < cfg.max_iters} |")
        say(f"  {name}: val loss {last['val_loss']:.4f} after {last['step']} steps, {minutes:.1f} min")
    return lines


def notebooks(names):
    import nbformat
    from ipykernel.kernelspec import install
    from nbclient import NotebookClient

    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # a kernel that runs this very Python, so the notebooks see the same PyTorch
    prefix = RUNS / "jupyter"
    install(prefix=str(prefix), kernel_name="tinygpt-verify")
    os.environ["JUPYTER_PATH"] = str(prefix / "share" / "jupyter")

    summary = ["| notebook | result | minutes |", "|---|---|---|"]
    details = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if not any(path.name.startswith(n) for n in names):
            continue
        say(f"running {path.name} ...")
        nb = nbformat.read(path, as_version=4)
        start = time.time()
        client = NotebookClient(nb, timeout=1800, kernel_name="tinygpt-verify", allow_errors=True,
                                resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute()
        except Exception as error:                       # the kernel died, or a cell timed out
            details.append(f"**Stopped:** {type(error).__name__}: {error}")
        minutes = (time.time() - start) / 60
        (RUNS / "executed").mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, RUNS / "executed" / path.name)
        errors, cells = [], []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            text = ""
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    text += output.text
                elif output.output_type == "error":
                    errors.append(f"cell {i}: {output.ename}: {output.evalue}")
                    text += f"[{output.ename}: {output.evalue}]\n"
                elif "text/plain" in output.get("data", {}) and "image/png" not in output.get("data", {}):
                    text += output["data"]["text/plain"] + "\n"
            first_line = cell.source.splitlines()[0] if cell.source else ""
            if text.strip():
                cells.append(f"[{i}] {first_line}\n{text.strip()[:700]}")
        result = "OK" if not errors else f"{len(errors)} error(s)"
        summary.append(f"| {path.name} | {result} | {minutes:.1f} |")
        say(f"  {path.name}: {result} in {minutes:.1f} min")
        details += [f"### {path.name}: {result}, {minutes:.1f} min", ""]
        details += [f"- {e}" for e in errors]
        details += ["", "```", "\n\n".join(cells), "```", ""]
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tests, benchmark, tiny run and chapter 1 only")
    args = parser.parse_args()
    os.chdir(ROOT)
    RUNS.mkdir(exist_ok=True)

    report = ["# tinyGPT verification report", ""] + environment()
    say("running the tests ...")
    report += ["", "## Tests", "", "```", run(sys.executable, "-m", "pytest", "-q", "--tb=short", "--color=no",
                                               "-p", "no:cacheprovider"), "```"]
    say("running the benchmark ...")
    report += ["", "## Benchmark", "", "```", run(sys.executable, "scripts/benchmark.py"), "```"]
    report += ["", "## Training runs", ""] + training(["tiny"] if args.quick else ["tiny", "small", "base"])
    if not args.quick:
        model, tok, _ = tinygpt.load_checkpoint(RUNS / "base" / "model.pt")
        report += ["", "## A sample from base (temperature 1.0)", "", "```",
                   tinygpt.sample(model, tok, prompt="ROMEO:", length=400, seed=0), "```"]
    summary, details = notebooks(["01"] if args.quick else ["0"])
    report += ["", "## Notebooks", ""] + summary + [""] + details

    path = RUNS / f"report-{platform.node() or 'this-computer'}.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    say(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
