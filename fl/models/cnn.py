"""
CNN — Convolutional network for image data (CIFAR-10, etc.).
"""

from __future__ import annotations

from fl.core.model_builder import Net as _Net

from fl.models.registry import register_model


@register_model("cnn")
class CNN(_Net):
    """
    Simple CNN for image input (3-channel, 32×32).

    Architecture: Conv5→Pool → Conv5→Pool → FC120 → FC84 → num_classes

    Args:
        num_classes: Number of output classes (default 10 for CIFAR).
    """

    pass


__all__ = ["CNN"]
