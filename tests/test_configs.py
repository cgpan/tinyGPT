from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from tinygpt import GPT, Config, count_params, load_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


def test_the_three_configs_exist():
    assert [p.stem for p in CONFIGS] == ["base", "small", "tiny"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_spells_out_every_setting_with_the_right_type(path):
    values = yaml.safe_load(path.read_text())
    expected = {f.name: type(f.default) for f in fields(Config)}
    assert set(values) == set(expected)
    for key, value in values.items():
        assert type(value) is expected[key], f"{key}: {value!r} should be {expected[key].__name__}"


def test_base_config_is_the_default_config():
    assert load_config(CONFIGS[0]) == Config()


def test_base_model_is_between_5_and_15_million_parameters():
    assert 5e6 < count_params(GPT(Config())) < 15e6


def test_overrides_replace_file_values():
    cfg = load_config(CONFIGS[1], n_layer=1, causal=False)
    assert cfg.n_layer == 1 and cfg.causal is False
