"""
Model registry and built-in model definitions.

Usage::

    from fl.models import get_model, get_model_for_batch

    # By name:
    TabularNet = get_model("tabular")
    net = TabularNet(input_dim=30, num_classes=2)

    # Auto-select by batch shape:
    net = get_model_for_batch(next(iter(trainloader)), num_classes=2)
"""

from fl.models.registry import (
    register_model,
    get_model,
    list_models,
    get_model_for_batch,
)

# Auto-import built-in models so their decorators run.
from fl.models import tabular, cnn  # noqa: F401

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "get_model_for_batch",
]
