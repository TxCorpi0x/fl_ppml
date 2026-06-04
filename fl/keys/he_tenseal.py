"""
fl.keys.he_tenseal — TenSEAL CKKS key generation and loading.

Key material:
    secret.pkl      — client-side context (has secret key)
    server_key.pkl  — server-side context (public key only)

Usage::

    from fl.keys.he_tenseal import generate, load_client, load_server

    generate(secret_path="keys/he_tenseal/secret_key.pkl", public_path="keys/he_tenseal/public_key.pkl")
    client_ctx = load_client("keys/he_tenseal/secret_key.pkl")
    server_ctx = load_server("keys/he_tenseal/public_key.pkl")
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_context():
    """Create a fresh TenSEAL CKKS context with secret key."""
    import tenseal as ts

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2**40
    return ctx


def _write(path: str, ctx_bytes: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"contexte": ctx_bytes}, f)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return pickle.load(f)["contexte"]


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    secret_path: str = "keys/he_tenseal/secret_key.pkl",
    public_path: str = "keys/he_tenseal/public_key.pkl",
    overwrite: bool = False,
) -> None:
    """
    Generate a fresh CKKS key pair and write both files.

    Args:
        secret_path:  Destination for the **private** context (client).
        public_path:  Destination for the **public** context (server).
        overwrite:    If False (default) raise if either file already exists.
    """
    import tenseal as ts

    for p in (secret_path, public_path):
        if os.path.exists(p) and not overwrite:
            raise FileExistsError(
                f"{p} already exists. Pass overwrite=True to regenerate."
            )

    ctx = _make_context()
    _write(secret_path, ctx.serialize(save_secret_key=True))
    _write(public_path, ctx.serialize())

    print(f"[he_tenseal] Secret (private) context → {secret_path}")
    print(f"[he_tenseal] Public context           → {public_path}")

    # Verify
    client_ctx = ts.context_from(_read_bytes(secret_path))
    server_ctx = ts.context_from(_read_bytes(public_path))
    print(f"[he_tenseal] client private? {client_ctx.is_private()}")  # True
    print(f"[he_tenseal] server private? {server_ctx.is_private()}")  # False


def load_client(secret_path: str = "keys/he_tenseal/secret_key.pkl"):
    """Load and return a TenSEAL context with the secret key (client side)."""
    import tenseal as ts

    if not os.path.exists(secret_path):
        raise FileNotFoundError(
            f"Secret key file not found: {secret_path}\n"
            f"Run: python -m fl.keys generate he_tenseal"
        )
    return ts.context_from(_read_bytes(secret_path))


def load_server(public_path: str = "keys/he_tenseal/public_key.pkl"):
    """Load and return a TenSEAL context without the secret key (server side)."""
    import tenseal as ts

    if not os.path.exists(public_path):
        raise FileNotFoundError(
            f"Public key file not found: {public_path}\n"
            f"Run: python -m fl.keys generate he_tenseal"
        )
    return ts.context_from(_read_bytes(public_path))


def load(secret_path: str = "keys/he_tenseal/secret_key.pkl"):
    """Alias for ``load_client`` — used by the unified fl.keys.load() dispatcher."""
    return load_client(secret_path)
