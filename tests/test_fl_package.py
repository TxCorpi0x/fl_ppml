"""
Smoke tests for the fl package.

Run with::

    cd fl_ppml
    pip install -e .
    pytest tests/
"""

from __future__ import annotations

import sys
import os

# Ensure fl_ppml/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ── Registry tests ────────────────────────────────────────────────────────────


def test_dataset_registry():
    from fl.datasets import list_datasets, get_dataset_loader

    datasets = list_datasets()
    assert "creditcard" in datasets
    assert "healthcare" in datasets
    assert "stock" in datasets
    Loader = get_dataset_loader("creditcard")
    assert Loader is not None


def test_model_registry():
    from fl.models import list_models, get_model

    models = list_models()
    assert "tabular" in models
    assert "cnn" in models
    TabularNet = get_model("tabular")
    net = TabularNet(input_dim=30, num_classes=2)
    assert net is not None


def test_privacy_registry():
    from fl.privacy import list_modes, get_privacy_mode

    modes = list_modes()
    expected = {
        "baseline",
        "he_tenseal",
        "he_concrete_tfhe",
        "zkp",
        "dp",
    }
    assert expected.issubset(set(modes)), f"Missing modes: {expected - set(modes)}"
    for name in modes:
        mode = get_privacy_mode(name)
        assert mode.name == name


# ── Config tests ──────────────────────────────────────────────────────────────


def test_config_defaults():
    from fl.config import FLConfig

    cfg = FLConfig()
    assert cfg.dataset == "creditcard"
    assert cfg.privacy_mode == "baseline"
    assert cfg.num_classes == 2
    assert cfg.is_he is False
    assert cfg.is_zkp is False
    assert cfg.is_dp is False


def test_config_he_property():
    from fl.config import FLConfig

    for mode in ["he_tenseal", "he_concrete_tfhe"]:
        cfg = FLConfig(privacy_mode=mode)
        assert cfg.is_he is True, f"is_he should be True for {mode}"


def test_config_from_dict():
    from fl.config import FLConfig

    cfg = FLConfig.from_dict(
        {"dataset": "healthcare", "num_rounds": 5, "unknown_key": "ignored"}
    )
    assert cfg.dataset == "healthcare"
    assert cfg.num_rounds == 5


# ── Model factory tests ───────────────────────────────────────────────────────


def test_model_for_batch_tabular():
    import torch
    from fl.models import get_model_for_batch

    batch = (torch.randn(32, 30), torch.zeros(32, dtype=torch.long))
    net = get_model_for_batch(batch, num_classes=2)
    out = net(batch[0])
    assert out.shape == (32, 2)


def test_model_for_batch_image():
    import torch
    from fl.models import get_model_for_batch

    batch = (torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))
    net = get_model_for_batch(batch, num_classes=10)
    out = net(batch[0])
    assert out.shape == (4, 10)


# ── Privacy mode context setup (no external deps) ─────────────────────────────


def test_baseline_contexts():
    from fl.config import FLConfig
    from fl.privacy import get_privacy_mode

    cfg = FLConfig()
    mode = get_privacy_mode("baseline")
    assert mode.setup_client_context(cfg) is None
    assert mode.setup_server_context(cfg) is None


def test_dp_context_without_params_file_requires_explicit_epsilon():
    import math

    import pytest

    from fl.config import FLConfig
    from fl.privacy import get_privacy_mode

    mode = get_privacy_mode("dp")
    # The default dp_epsilon is the "load from file" sentinel; silently using it
    # weakened the privacy budget.
    with pytest.raises(FileNotFoundError):
        mode.setup_client_context(FLConfig(dp_params_path="/nonexistent/dp_params.json"))

    cfg = FLConfig(dp_params_path="/nonexistent/dp_params.json", dp_epsilon=1.0)
    ctx = mode.setup_client_context(cfg)
    assert ctx.epsilon == 1.0
    assert math.isclose(ctx.noise_multiplier, math.sqrt(2 * math.log(1.25 / cfg.dp_delta)))


# ── Custom mode registration test ─────────────────────────────────────────────


def test_register_custom_mode():
    from fl.privacy.registry import register_mode, get_privacy_mode, _REGISTRY
    from fl.privacy.base import PrivacyMode

    @register_mode("_test_custom")
    class CustomMode(PrivacyMode):
        @property
        def name(self):
            return "_test_custom"

        def setup_client_context(self, config):
            return None

        def setup_server_context(self, config):
            return None

    mode = get_privacy_mode("_test_custom")
    assert mode.name == "_test_custom"

    # Clean up
    del _REGISTRY["_test_custom"]


def test_register_custom_dataset():
    from fl.datasets.registry import register_dataset, get_dataset_loader, _REGISTRY
    from fl.datasets.base import DatasetLoader, DatasetSpec

    @register_dataset("_test_ds")
    class TestDS(DatasetLoader):
        spec = DatasetSpec(name="_test_ds", input_dim=10, num_classes=2)

        def load(self, config):
            return [], [], None

    Cls = get_dataset_loader("_test_ds")
    assert Cls.get_spec().name == "_test_ds"

    del _REGISTRY["_test_ds"]
