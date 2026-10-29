"""Charts for the chapter notebooks. Every chart also prints its numbers as text."""

import math

import numpy as np

# colours
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# the three numbers every chapter is measured against
LANDMARKS = {"knowing nothing": 4.17, "counting pairs": 2.48, "tinyGPT": 1.49}


def char_label(ch):
    """Make invisible characters visible on an axis."""
    return {"\n": "⏎", " ": "␣"}.get(ch, ch)


def _axes(figsize):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    return plt, fig, ax


def scoreboard(results=None):
    """Show your results on the line from 4.17 (knowing nothing) to 1.49 (tinyGPT).

    results: a dict like {"knowing frequencies": 3.35}.
    """
    results = dict(results or {})
    rows = [(v, k, "landmark") for k, v in LANDMARKS.items()] + [(v, k, "yours") for k, v in results.items()]
    print("validation loss (lower is better)")
    for value, name, kind in sorted(rows, reverse=True):
        print(f"  {value:5.2f}  {name:<32} {kind}")

    plt, fig, ax = _axes((9, 2.8))
    low = min([1.0, *(v - 0.3 for v in results.values())])
    high = max([4.5, *(v + 0.3 for v in results.values())])
    ax.set_xlim(high, low)                      # worst on the left: progress moves right
    ax.set_ylim(-1.2, 1.4)
    ax.set_yticks([])
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_xticks(range(math.floor(high), math.ceil(low) - 1, -1))
    for name, value in LANDMARKS.items():
        ax.plot([value, value], [-0.12, 0.12], color=MUTED, lw=1)
        ax.text(value, -0.3, f"{name}\n{value:.2f}", ha="center", va="top", fontsize=9, color=INK_2)
    for i, (name, value) in enumerate(sorted(results.items(), key=lambda kv: -kv[1])):
        ax.plot(value, 0, "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(value, 0.3 + 0.45 * (i % 2), f"{name}\n{value:.2f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_title("The number: validation loss, lower is better  →", loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    plt.show()


def char_counts(text):
    """Bar chart: how often each character appears, most common first."""
    from collections import Counter

    counts = Counter(text).most_common()
    print(f"most common: {counts[:5]}\nrarest:      {counts[-5:]}")

    plt, fig, ax = _axes((11, 3.2))
    xs = range(len(counts))
    ax.bar(xs, [n for _, n in counts], width=0.6, color=BLUE)
    ax.set_xticks(list(xs), [char_label(ch) for ch, _ in counts], fontsize=7)
    ax.set_xlim(-1, len(counts))
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.8)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    ax.set_axisbelow(True)
    ax.set_ylabel("times it appears", color=INK_2)
    ax.set_title("How often each character appears  (⏎ = line break, ␣ = space)", loc="left",
                 fontsize=10, color=INK_2)
    fig.tight_layout()
    plt.show()


def bigram_heatmap(counts, tok):
    """Heatmap of a counting bigram: row = a character, column = the character that came next."""
    from matplotlib.colors import LinearSegmentedColormap

    plt, fig, ax = _axes((8.5, 8))
    ax.spines["bottom"].set_visible(False)
    labels = [char_label(ch) for ch in tok.chars]
    image = ax.imshow(np.log10(1 + counts), cmap=LinearSegmentedColormap.from_list("blues", BLUE_RAMP))
    ax.set_xticks(range(len(labels)), labels, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.tick_params(length=0, labeltop=True, labelbottom=False)
    ax.set_xlabel("next character", color=INK_2)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("this character", color=INK_2)
    marks = [0] + [10 ** t for t in range(1, math.ceil(math.log10(1 + counts.max())))]
    bar = fig.colorbar(image, ax=ax, shrink=0.6, ticks=[math.log10(1 + m) for m in marks])
    bar.ax.set_yticklabels([f"{m:,}" for m in marks], color=INK_2, fontsize=8)
    bar.outline.set_visible(False)
    bar.set_label("times this pair appears", color=INK_2)
    fig.tight_layout()
    plt.show()


def plot_history(runs):
    """Line chart of the loss as training goes on.

    runs: the history from one train() call, which draws its training and validation
    loss; or a dict like {"lr 0.01": history, "lr 0.1": history} to compare the
    validation loss of several runs.
    """
    if isinstance(runs, dict):
        lines = {name: [(r["step"], r["val_loss"]) for r in history] for name, history in runs.items()}
        title = "Validation loss while training"
    else:
        lines = {"training": [(r["step"], r["train_loss"]) for r in runs],
                 "validation": [(r["step"], r["val_loss"]) for r in runs]}
        title = "Loss while training"
    print(f"{'':<26}{'start':>8}{'end':>8}{'lowest':>8}")
    for name, points in lines.items():
        values = [v for _, v in points]
        print(f"{name:<26}{values[0]:>8.3f}{values[-1]:>8.3f}{min(values):>8.3f}")

    plt, fig, ax = _axes((9, 4.2))
    values = [v for points in lines.values() for _, v in points]
    ax.set_ylim(min(1.3, min(values) - 0.1), max(4.4, max(values) + 0.1))
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for name, value in LANDMARKS.items():
        ax.axhline(value, color=AXIS, lw=1, zorder=1)
        ax.text(1.01, value, f"{name} {value:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=MUTED)
    for (name, points), color in zip(lines.items(), SERIES):
        ax.plot(*zip(*points), color=color, lw=2, label=name, solid_capstyle="round", zorder=2)
    ax.set_xlabel("training step", color=INK_2)
    ax.set_ylabel("loss", color=INK_2)
    ax.set_title(title, loc="left", fontsize=10, color=INK_2)
    if len(lines) > 1:
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    plt.show()


def attention_map(model, tok, text, layer=0, head=0):
    """Heatmap of where one attention head looks while it reads `text`.

    Row t shows how the character at position t spreads its attention over the
    characters up to and including itself. Each row sums to 1.
    """
    import torch
    from matplotlib.colors import LinearSegmentedColormap

    attention_head = model.blocks[layer].attn.heads[head]
    seen = {}
    hook = attention_head.register_forward_pre_hook(lambda module, args: seen.update(x=args[0]))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            idx = torch.tensor([tok.encode(text)], device=next(model.parameters()).device)
            model(idx)
            weights = attention_head.attention_weights(seen["x"])[0].cpu().numpy()
    finally:
        hook.remove()
        model.train(was_training)

    top = np.argsort(weights[-1])[::-1][:3]
    looks = ", ".join(f"{text[i]!r} at position {i} ({weights[-1][i]:.2f})" for i in top)
    print(f"Reading the last character, {text[-1]!r}, this head looks most at: {looks}")

    plt, fig, ax = _axes((7.5, 7))
    ax.spines["bottom"].set_visible(False)
    labels = [char_label(ch) for ch in text]
    image = ax.imshow(weights, cmap=LinearSegmentedColormap.from_list("blues", BLUE_RAMP), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(length=0, labeltop=True, labelbottom=False)
    ax.set_xlabel("looks at this character", color=INK_2)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("while reading this character", color=INK_2)
    bar = fig.colorbar(image, ax=ax, shrink=0.6)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelcolor=INK_2, labelsize=8)
    bar.set_label("attention weight", color=INK_2)
    ax.set_title(f"Layer {layer}, head {head}", loc="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    plt.show()
