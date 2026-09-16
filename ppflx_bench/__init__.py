"""
ppflx_bench — the benchmark harness around the ppflx library.

Contents:
    app          – the Flower App components (ServerApp, ClientApp)
    launch       – SuperLink/SuperNode lifecycle for one run
    runner       – single-mode runs on the Simulation Runtime
    compare      – multi-mode orchestration, reporting and validation
    datasets     – dataset registry, loaders and partitioning
    legacy       – pre-refactor utilities kept for reference
"""

from ppflx_bench.datasets import get_dataset_loader, list_datasets

__all__ = ["get_dataset_loader", "list_datasets"]
