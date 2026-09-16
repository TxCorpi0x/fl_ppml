"""
Model registry: @register_model('name') decorator + lookup helpers.
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """
    Class decorator that registers a nn.Module under *name*.

    Example::

        @register_model("tabular")
        class TabularNet(nn.Module):
            ...
    """

    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            import warnings

            warnings.warn(
                f"Model '{name}' is already registered; overwriting with {cls.__name__}.",
                stacklevel=2,
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_model(name: str) -> Type[nn.Module]:
    """Return the model class registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


def get_model_for_batch(batch, num_classes: int) -> nn.Module:
    """
    Auto-select model based on the shape of a sample batch.

    * 2-D batch → tabular data → TabularNet
    * 4-D batch → image data   → CNN

    Args:
        batch: A (features, labels) tuple from a DataLoader.
        num_classes: Number of output classes.

    Returns:
        An instantiated nn.Module.
    """
    x, _ = batch
    if x.dim() == 2:  # [batch, features]
        cls = get_model("tabular")
        input_dim = x.shape[1]
        return cls(input_dim=input_dim, num_classes=num_classes, hidden_dims=[64, 32])
    else:  # [batch, C, H, W]
        cls = get_model("cnn")
        in_channels = x.shape[1]  # 1 for grayscale (MNIST), 3 for RGB (CIFAR-10)
        input_size = x.shape[2]  # spatial height (assumed square)
        return cls(
            num_classes=num_classes, in_channels=in_channels, input_size=input_size
        )
