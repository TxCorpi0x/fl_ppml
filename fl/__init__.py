"""
fl — privacy-preserving federated learning library.

Public API:
    FLConfig            – single configuration dataclass
    get_dataset_loader  – dataset registry lookup
    get_privacy_mode    – privacy-mode registry lookup
    get_model           – model registry lookup
    run_comparison      – top-level orchestration entrypoint

Quick example::

    from fl import FLConfig, run_comparison

    cfg = FLConfig(dataset="creditcard", privacy_mode="baseline", num_rounds=3)
    results = run_comparison(cfg)
"""

from fl.config import FLConfig
from fl.datasets import get_dataset_loader, list_datasets
from fl.privacy import get_privacy_mode, list_modes
from fl.models import get_model, list_models

__all__ = [
    "FLConfig",
    "get_dataset_loader",
    "list_datasets",
    "get_privacy_mode",
    "list_modes",
    "get_model",
    "list_models",
]
