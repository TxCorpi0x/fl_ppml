"""
fl.compare — dataset/mode registry and experiment runner.

Public API::

    from fl.compare import DATASETS, MODES, run_comparison, CompareConfig

    # Run with dataset defaults
    run_comparison(dataset="healthcare", modes=["baseline", "zkp", "dp"])

    # Full control
    cfg = CompareConfig(
        dataset="creditcard",
        modes=["baseline", "he_tenseal", "dp"],
        num_clients=3,
        num_rounds=5,
        output_dir="./results/my_run",
    )
    run_comparison(cfg)

Adding a new dataset: add one entry to fl/compare/registry.py → DATASETS.
Adding a new mode:  add one entry to fl/compare/registry.py → MODES.
No other files need to change.
"""

from fl.compare.registry import DATASETS, MODES, DatasetConfig, ModeConfig
from fl.compare.runner import (
    CompareConfig,
    run_comparison,
    run_alpha_sweep,
    run_dp_epsilon_sweep,
)

__all__ = [
    "DATASETS",
    "MODES",
    "DatasetConfig",
    "ModeConfig",
    "CompareConfig",
    "run_comparison",
    "run_alpha_sweep",
    "run_dp_epsilon_sweep",
]
