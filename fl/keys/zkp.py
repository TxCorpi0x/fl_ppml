"""
fl.keys.zkp — Zero-Knowledge Proof parameter generation and loading.

Key material:
    zkp_params.pkl  — Pedersen commitment group parameters (prime p, generator g/h)

Usage::

    from fl.keys.zkp import generate, load

    generate(output="keys/zkp/zkp_params.pkl", bit_length=2048)
    ctx = load("keys/zkp_params.pkl")
"""

from __future__ import annotations

import os


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    output: str = "keys/zkp/zkp_params.pkl",
    bit_length: int = 2048,
    overwrite: bool = False,
) -> "ZKPContext":  # noqa: F821
    """
    Generate Pedersen commitment parameters and save to *output*.

    Args:
        output:     Destination path for the .pkl file.
        bit_length: Security parameter in bits (default 2048; use 4096 for
                    higher security at the cost of speed).
        overwrite:  Raise if file exists and this is False.

    Returns:
        The generated :class:`~fl.core.zkp.ZKPContext`.
    """
    from fl.core.zkp import create_zkp_context, write_zkp_params

    if os.path.exists(output) and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Pass overwrite=True to regenerate."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    print(
        f"[zkp] Generating {bit_length}-bit Pedersen parameters … (may take a moment)"
    )
    ctx = create_zkp_context(bit_length=bit_length)
    write_zkp_params(output, ctx)

    print(f"[zkp] Prime modulus p: {bit_length} bits")
    print(f"[OK] ZKP parameters saved → {output}")
    return ctx


def load(path: str = "zkp_params.pkl") -> "ZKPContext":  # noqa: F821
    """Load and return a :class:`~fl.core.zkp.ZKPContext`."""
    from fl.core.zkp import read_zkp_params

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ZKP params file not found: {path}\n"
            f"Run: python -m fl.keys generate zkp"
        )
    return read_zkp_params(path)
