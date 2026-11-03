"""The given code: settings, and the choice of device and number format."""

import pytest
import torch

from scratchgpt.config import Config, load_config, parse_overrides
from scratchgpt.device import amp_dtype


def test_settings_get_their_types_from_the_command_line():
    cfg = load_config(None, **parse_overrides(["lr=3e-3", "max_steps=1e3", "compile=true", "dropout=0"]))
    assert cfg.lr == 0.003 and isinstance(cfg.lr, float)
    assert cfg.max_steps == 1000 and isinstance(cfg.max_steps, int)
    assert cfg.compile is True and cfg.dropout == 0.0


def test_settings_in_yaml_without_a_decimal_point_are_numbers(tmp_path):
    path = tmp_path / "final.yaml"
    path.write_text("lr: 3e-4\nmin_lr: 3e-5\nn_layer: 6\n", encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.lr, cfg.min_lr, cfg.n_layer) == (0.0003, 0.00003, 6)


def test_unknown_settings_are_refused():
    with pytest.raises(ValueError, match="unknown setting"):
        load_config(None, learning_rate=0.1)


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError):
        Config(d_model=100, n_head=3)


def test_number_formats():
    cpu = torch.device("cpu")
    assert amp_dtype(cpu) is None and amp_dtype(cpu, "float32") is None
    assert amp_dtype(cpu, "bfloat16") == torch.bfloat16
    with pytest.raises(ValueError):
        amp_dtype(cpu, "float8")
