"""
Dataset registry: @register_dataset('name') decorator + lookup helpers.
"""

from __future__ import annotations

from typing import Dict, Type

from fl.datasets.base import DatasetLoader

_REGISTRY: Dict[str, Type[DatasetLoader]] = {}


def register_dataset(name: str):
    """
    Class decorator that registers a DatasetLoader under *name*.

    Example::

        @register_dataset("creditcard")
        class CreditCardLoader(DatasetLoader):
            ...
    """

    def _decorator(cls: Type[DatasetLoader]) -> Type[DatasetLoader]:
        if name in _REGISTRY:
            import warnings

            warnings.warn(
                f"Dataset '{name}' is already registered; overwriting with {cls.__name__}.",
                stacklevel=2,
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_dataset_loader(name: str) -> Type[DatasetLoader]:
    """Return the loader class registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names."""
    return sorted(_REGISTRY.keys())
