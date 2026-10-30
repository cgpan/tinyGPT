"""Project 6: fitting a scaling law, and planning a run with it."""

import math

import numpy as np
import pytest

from scratchgpt import scaling
from scratchgpt.config import Config


def test_fit_power_law_recovers_a_known_law():
    x = np.array([1e15, 3e15, 1e16, 3e16, 1e17])
    y = scaling.power_law(x, a=2000.0, b=0.2, c=1.5)
    a, b, c = scaling.fit_power_law(x, y)
    assert b == pytest.approx(0.2, abs=0.01)
    assert c == pytest.approx(1.5, abs=0.02)
    assert scaling.power_law(1e18, a, b, c) == pytest.approx(scaling.power_law(1e18, 2000.0, 0.2, 1.5), rel=0.01)


def test_fit_power_law_copes_with_noise():
    rng = np.random.default_rng(0)
    x = np.logspace(14, 17, 6)
    y = scaling.power_law(x, a=500.0, b=0.15, c=2.0) * (1 + rng.normal(0, 0.003, 6))
    a, b, c = scaling.fit_power_law(x, y)
    assert abs(scaling.power_law(3e17, a, b, c) - scaling.power_law(3e17, 500.0, 0.15, 2.0)) < 0.05


def test_plan_run_is_compute_optimal_when_data_is_plentiful():
    n, d = scaling.plan_run(compute=6 * 20 * 1e8**2, unique_tokens=1e12)
    assert n == pytest.approx(1e8, rel=1e-6) and d == pytest.approx(20 * 1e8, rel=1e-6)


def test_plan_run_spends_extra_compute_on_parameters_when_data_runs_out():
    compute = 6 * 20 * 1e8**2                                 # wants 2e9 tokens
    n, d = scaling.plan_run(compute, unique_tokens=1e8, max_epochs=4)
    assert d == 4e8
    assert n == pytest.approx(compute / (6 * 4e8), rel=1e-6)
    assert 6 * n * d == pytest.approx(compute, rel=1e-6)


def test_sweep_configs_grow_width_and_depth_together():
    configs = scaling.sweep_configs(Config(out_dir="runs/sweep"), widths=(128, 256))
    assert [(c.d_model, c.n_layer, c.n_head) for c in configs] == [(128, 2, 2), (256, 4, 4)]
    assert configs[1].out_dir == "runs/sweep/d256"
    assert math.isclose(configs[0].lr, Config().lr)
