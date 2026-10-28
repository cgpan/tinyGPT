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
    ax.set_xlim(4.5, 1.0)                       # worst on the left: progress moves right
    ax.set_ylim(-1.2, 1.4)
    ax.set_yticks([])
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_xticks([4, 3, 2, 1])
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
