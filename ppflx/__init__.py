"""
ppflx — privacy-preserving federated learning on Flower.

Public API:
    FLConfig            – single configuration dataclass
    get_privacy_mode    – privacy-mode registry lookup
    get_model           – model registry lookup

Quick example::

    from ppflx import FLConfig, get_privacy_mode

    config = FLConfig(dataset="creditcard", privacy_mode="zkp", num_rounds=3)
    mode = get_privacy_mode(config.privacy_mode)

The benchmark harness (dataset loaders, the Flower App, the comparison runner)
lives in the separate ``ppflx_bench`` package.
"""

from ppflx.config import FLConfig
from ppflx.privacy import get_privacy_mode, list_modes
from ppflx.models import get_model, list_models

__all__ = ["FLConfig", "get_privacy_mode", "list_modes", "get_model", "list_models"]
