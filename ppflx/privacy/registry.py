"""
Privacy mode registry.
"""

from __future__ import annotations

from typing import Dict, Type

from ppflx.privacy.base import PrivacyMode

_REGISTRY: Dict[str, Type[PrivacyMode]] = {}


def register_mode(name: str):
    """
    Class decorator that registers a PrivacyMode under *name*.

    Example::

        @register_mode("my_mode")
        class MyMode(PrivacyMode):
            ...
    """

    def _decorator(cls: Type[PrivacyMode]) -> Type[PrivacyMode]:
        if name in _REGISTRY:
            import warnings

            warnings.warn(
                f"Privacy mode '{name}' is already registered; overwriting.",
                stacklevel=2,
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_privacy_mode(name: str) -> PrivacyMode:
    """Return an instance of the mode registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown privacy mode '{name}'. Available modes: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_modes() -> list[str]:
    """Return a sorted list of all registered mode names."""
    return sorted(_REGISTRY.keys())
