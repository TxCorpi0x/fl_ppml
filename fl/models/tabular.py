"""
TabularNet — MLP for tabular data (healthcare, creditcard, stock).
"""

from __future__ import annotations

from fl.core.model_builder import TabularNet as _TabularNet

from fl.models.registry import register_model


@register_model("tabular")
class TabularNet(_TabularNet):
    """
    Simple MLP for tabular input.

    Architecture (default): input → 64 → 32 → num_classes
    Configurable via hidden_dims list.

    Args:
        input_dim:   Number of input features.
        num_classes: Number of output classes.
        hidden_dims: Hidden layer sizes (default [64, 32]).
    """

    # Inherits forward() and __init__() from the tested TabularNet.
    pass


__all__ = ["TabularNet"]
