"""
Dataset registry and base class.

Usage::

    from ppflx_bench.datasets import get_dataset_loader, list_datasets

    Loader = get_dataset_loader("creditcard")
    trainloaders, valloaders, testloader = Loader().load(config)
"""

from ppflx_bench.datasets.base import DatasetSpec, DatasetLoader
from ppflx_bench.datasets.registry import register_dataset, get_dataset_loader, list_datasets

# Auto-import built-in loaders so their @register_dataset decorators run.
from ppflx_bench.datasets import creditcard, healthcare, stock, mnist, cifar10  # noqa: F401

__all__ = [
    "DatasetSpec",
    "DatasetLoader",
    "register_dataset",
    "get_dataset_loader",
    "list_datasets",
]
