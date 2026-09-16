"""
ppflx.keys — Cryptographic key and parameter management.

Each sub-module owns one privacy mode's key lifecycle:
    generate(...) → creates and saves key material
    load(path)    → returns a ready-to-use context/params object

Unified dispatcher::

    from ppflx.keys import generate, load
    generate("he_tenseal", secret_path="keys/he_tenseal/secret_context.bin", public_path="keys/he_tenseal/public_context.bin")
    generate("dp",  output="keys/dp/dp_params.json", epsilon=1.0)
    generate("zkp", output="keys/zkp/zkp_params.json", bit_length=2048)

Or from the command line::

    python -m ppflx.keys generate he_tenseal
    python -m ppflx.keys generate dp --epsilon 1.0
    python -m ppflx.keys generate zkp --bit_length 2048
    python -m ppflx.keys list
"""

from __future__ import annotations

from typing import Any

# Per-mode modules — imported lazily to avoid heavy optional deps at import time
_GENERATORS = {
    "he_tenseal": "ppflx.keys.he_tenseal",
    "he_concrete_tfhe": "ppflx.keys.concrete_tfhe",
    "dp": "ppflx.keys.dp",
    "zkp": "ppflx.keys.zkp",
}


def list_modes() -> list[str]:
    """Return the supported modes for key generation."""
    return sorted(_GENERATORS)


def generate(mode: str, **kwargs: Any) -> Any:
    """
    Generate and save key material for *mode*.

    Keyword arguments are forwarded to the mode's ``generate()`` function.
    Returns whatever the mode's generator returns (config / context / None).

    Example::

        from ppflx.keys import generate
        generate("dp", output="dp_params.json", epsilon=1.0)
        generate("he_tenseal", secret_path="keys/he_tenseal/secret_context.bin", public_path="keys/he_tenseal/public_context.bin")
    """
    if mode not in _GENERATORS:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(_GENERATORS)}")
    import importlib

    mod = importlib.import_module(_GENERATORS[mode])
    return mod.generate(**kwargs)


def load(mode: str, **kwargs: Any) -> Any:
    """
    Load existing key material for *mode*.

    Keyword arguments are forwarded to the mode's ``load()`` function.
    Returns a ready-to-use context/params object.

    Example::

        from ppflx.keys import load
        ctx = load("he_tenseal", secret_path="keys/he_tenseal/secret_context.bin")
        dp  = load("dp",         path="keys/dp/dp_params.json")
    """
    if mode not in _GENERATORS:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(_GENERATORS)}")
    import importlib

    mod = importlib.import_module(_GENERATORS[mode])
    if not hasattr(mod, "load"):
        raise NotImplementedError(f"Mode '{mode}' does not expose a load() function.")
    return mod.load(**kwargs)


__all__ = ["generate", "load", "list_modes"]
