import math

import pytest

from fl.privacy.zkp import ZKPSampledMode


def make_state(n):
    return {f"layer{i}": None for i in range(n)}


def test_default_sampling(monkeypatch):
    monkeypatch.delenv("FL_ZKP_NUM_LAYERS", raising=False)
    monkeypatch.delenv("FL_ZKP_SAMPLE_PCT", raising=False)
    monkeypatch.delenv("FL_ZKP_SAMPLE_SEED", raising=False)

    mode = ZKPSampledMode()
    state = make_state(10)
    sampled = mode._select_layers(state)

    assert 1 <= len(sampled) <= len(state)
    assert all(s in state for s in sampled)


def test_num_layers_env(monkeypatch):
    monkeypatch.setenv("FL_ZKP_NUM_LAYERS", "3")
    mode = ZKPSampledMode()
    state = make_state(10)
    sampled = mode._select_layers(state)
    assert len(sampled) == 3


def test_pct_sampling_env(monkeypatch):
    monkeypatch.delenv("FL_ZKP_NUM_LAYERS", raising=False)
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.5")
    mode = ZKPSampledMode()
    state = make_state(7)
    sampled = mode._select_layers(state)
    assert len(sampled) == math.ceil(7 * 0.5)


def test_seed_determinism(monkeypatch):
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.3")
    monkeypatch.setenv("FL_ZKP_SAMPLE_SEED", "42")
    mode = ZKPSampledMode()
    state = make_state(20)
    a = mode._select_layers(state)
    b = mode._select_layers(state)
    assert a == b


def test_select_by_size(monkeypatch):
    # Create a fake state dict with different sizes
    import numpy as np

    s = {
        "small": np.zeros((2, 2)),
        "medium": np.zeros((10, 10)),
        "large": np.zeros((100, 100)),
    }
    monkeypatch.setenv("FL_ZKP_SELECT_BY", "size")
    monkeypatch.setenv("FL_ZKP_NUM_LAYERS", "1")
    mode = ZKPSampledMode()
    selected = mode._select_layers(s)
    assert selected == ["large"]
