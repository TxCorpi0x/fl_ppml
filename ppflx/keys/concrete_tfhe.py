"""
ppflx.keys.concrete_tfhe — Concrete TFHE key management.

Pre-compiled key bundles live in ``ppflx/keys/prebuilt/`` (formerly
``ppflx/concrete_tfhe_keys/``).  Each bundle is named::

    <prefix>_secret_keys.bin
    <prefix>_eval_keys.bin
    <prefix>_client.zip
    <prefix>_server.zip

At runtime, keys are generated freshly via
:func:`ppflx.core.security.get_concrete_aggregation_context` → ``ctx.generate_keys()``.
This module provides helpers for that flow and for inspecting prebuilt bundles.

Usage::

    from ppflx.keys.concrete_tfhe import generate, list_prebuilt, prebuilt_dir

    # Generate runtime keys (not saved to disk — held in memory)
    ctx = generate(bit_width=14, num_clients=2, enable_fhe=True)

    # List pre-compiled key bundles
    for name in list_prebuilt():
        print(name)
"""

from __future__ import annotations

import os
from pathlib import Path


# Prebuilt key bundles directory (relative to this file)
prebuilt_dir: Path = Path(__file__).parent / "prebuilt"


# ── Prebuilt key helpers ──────────────────────────────────────────────────────


def list_prebuilt() -> list[str]:
    """
    Return a sorted list of prebuilt key bundle prefixes available in
    :data:`prebuilt_dir`.

    A bundle is counted if ``<prefix>_eval_keys.bin`` exists.
    """
    if not prebuilt_dir.exists():
        return []
    return sorted(
        p.stem.removesuffix("_eval_keys") for p in prebuilt_dir.glob("*_eval_keys.bin")
    )


def load_prebuilt(prefix: str) -> dict:
    """
    Load a prebuilt key bundle by prefix name.

    Returns a dict with keys ``secret_keys``, ``eval_keys``,
    ``client_zip``, ``server_zip`` (raw bytes).

    Raises FileNotFoundError if the bundle does not exist.
    """
    bundle: dict[str, bytes] = {}
    for suffix, key in [
        ("_secret_keys.bin", "secret_keys"),
        ("_eval_keys.bin", "eval_keys"),
        ("_client.zip", "client_zip"),
        ("_server.zip", "server_zip"),
    ]:
        path = prebuilt_dir / f"{prefix}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Prebuilt key file not found: {path}")
        bundle[key] = path.read_bytes()
    return bundle


# ── Runtime key generation ────────────────────────────────────────────────────


def generate(
    bit_width: int = 14,
    num_clients: int = 2,
    enable_fhe: bool = True,
    adaptive_quant: bool = False,
):
    """
    Instantiate a Concrete TFHE aggregation context and generate fresh keys.

    Keys are held in memory on the returned context object (``ctx.private_key``,
    ``ctx.public_key``) — they are **not** persisted to disk.  Re-generate per
    FL run (key generation takes seconds for 14-bit, not minutes).

    Args:
        bit_width:      Fixed-point quantisation width (14 recommended for FHE).
        num_clients:    Number of federated clients (used for pre-averaging divisor).
        enable_fhe:     If False, simulate FHE cost without real crypto (sim mode).
        adaptive_quant: Must be False for actual FHE; adaptive quant is incompatible.

    Returns:
        Aggregation context with ``.private_key`` and ``.public_key`` attributes.
    """
    from ppflx.core.security import get_concrete_aggregation_context

    ctx = get_concrete_aggregation_context(
        bit_width=bit_width,
        enable_fhe=enable_fhe,
        adaptive_quant=adaptive_quant,
        num_clients=num_clients,
    )
    private_key, public_key = ctx.generate_keys()
    ctx.private_key = private_key
    ctx.public_key = public_key

    mode = "FHE" if enable_fhe else "simulation"
    print(
        f"[concrete_tfhe] {bit_width}-bit context ready "
        f"({mode}, {num_clients} clients)"
    )
    return ctx


# Alias for unified ppflx.keys dispatcher
load = generate
