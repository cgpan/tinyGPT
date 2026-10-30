"""Project 6: scaling laws, and planning the final run from them."""

import math

import numpy as np


def fit_power_law(x, y, grid=400):
    """Fit y = a * x**(-b) + c to the points (x, y): c is the loss no amount of x removes.

    For each c in np.linspace(0, 0.999 * min(y), grid), fit a straight line to
    (log x, log(y - c)) by least squares (np.polyfit); its slope is -b and its intercept
    log a. Keep the c with the smallest squared error in y itself. Returns (a, b, c).
    """
    # BEGIN SOLUTION 6
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best = None
    for c in np.linspace(0, 0.999 * y.min(), grid):
        slope, intercept = np.polyfit(np.log(x), np.log(y - c), 1)
        a, b = math.exp(intercept), -slope
        error = float(np.sum((a * x ** (-b) + c - y) ** 2))
        if best is None or error < best[0]:
            best = (error, a, b, c)
    return best[1], best[2], float(best[3])
    # END SOLUTION


def power_law(x, a, b, c):
    """Given."""
    return a * np.asarray(x, dtype=float) ** (-b) + c


def plan_run(compute, unique_tokens, max_epochs=4, tokens_per_parameter=20):
    """How big a model, trained on how many tokens, for `compute` FLOPs (C = 6 N D).

    With plenty of data the compute-optimal choice (Hoffmann et al., 2022) is about
    tokens_per_parameter tokens per parameter: D = tokens_per_parameter * N, so
    N = sqrt(C / (6 * tokens_per_parameter)). But data repeated more than about max_epochs
    times is worth much less (Muennighoff et al., 2023), so if that D is more than
    max_epochs * unique_tokens, set D to that and spend the rest of the compute on a bigger
    model: N = C / (6 D). Returns (parameters, tokens), as ints.
    """
    # BEGIN SOLUTION 6
    n = math.sqrt(compute / (6 * tokens_per_parameter))
    d = tokens_per_parameter * n
    if d > max_epochs * unique_tokens:
        d = max_epochs * unique_tokens
        n = compute / (6 * d)
    return int(n), int(d)
    # END SOLUTION


def sweep_configs(base, widths=(128, 192, 256, 384, 512)):
    """Configs for a scaling sweep: the base config at several widths, with depth growing
    alongside (n_layer = d_model // 64) and heads of 64. Given."""
    import dataclasses
    return [dataclasses.replace(base, d_model=w, n_layer=max(2, w // 64), n_head=max(1, w // 64),
                                out_dir=f"{base.out_dir}/d{w}") for w in widths]
