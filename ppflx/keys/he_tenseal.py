"""
ppflx.keys.he_tenseal — TenSEAL CKKS key generation and loading.

Key material:
    secret_context.bin  — client-side context (has the secret key)
    public_context.bin  — server-side context (public key only)

Files hold TenSEAL's own context serialization behind a short header. They are
never unpickled: a key file can't execute code when it's loaded.

Usage::

    from ppflx.keys.he_tenseal import generate, load_client, load_server

    generate()
    client_ctx = load_client("keys/he_tenseal/secret_context.bin")
    server_ctx = load_server("keys/he_tenseal/public_context.bin")
"""

from __future__ import annotations

import os

SECRET_PATH = "keys/he_tenseal/secret_context.bin"
PUBLIC_PATH = "keys/he_tenseal/public_context.bin"
_MAGIC = b"FLTENSEAL1\n"


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_context():
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


def write_context(path: str, ctx_bytes: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(_MAGIC + ctx_bytes)


def read_context_bytes(path: str) -> bytes:
    """Serialized context bytes from a key file; refuses anything else, including legacy pickles."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(_MAGIC):
        raise ValueError(
            f"{path} is not a TenSEAL key file of this version (legacy pickle files are no longer "
            "loaded because unpickling can execute code). Regenerate: python -m ppflx.keys generate he_tenseal"
        )
    return data[len(_MAGIC):]


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    secret_path: str = SECRET_PATH,
    public_path: str = PUBLIC_PATH,
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

    ctx = make_context()
    write_context(secret_path, ctx.serialize(save_secret_key=True))
    write_context(public_path, ctx.serialize())

    print(f"[he_tenseal] Secret (private) context → {secret_path}")
    print(f"[he_tenseal] Public context           → {public_path}")

    # Verify
    client_ctx = ts.context_from(read_context_bytes(secret_path))
    server_ctx = ts.context_from(read_context_bytes(public_path))
    print(f"[he_tenseal] client private? {client_ctx.is_private()}")  # True
    print(f"[he_tenseal] server private? {server_ctx.is_private()}")  # False


def load_client(secret_path: str = SECRET_PATH):
    """Load and return a TenSEAL context with the secret key (client side)."""
    import tenseal as ts

    if not os.path.exists(secret_path):
        raise FileNotFoundError(
            f"Secret key file not found: {secret_path}\n"
            f"Run: python -m ppflx.keys generate he_tenseal"
        )
    return ts.context_from(read_context_bytes(secret_path))


def load_server(public_path: str = PUBLIC_PATH):
    """Load and return a TenSEAL context without the secret key (server side)."""
    import tenseal as ts

    if not os.path.exists(public_path):
        raise FileNotFoundError(
            f"Public key file not found: {public_path}\n"
            f"Run: python -m ppflx.keys generate he_tenseal"
        )
    ctx = ts.context_from(read_context_bytes(public_path))
    if ctx.is_private():
        raise ValueError(f"{public_path} contains a secret key; the server must only hold the public context")
    return ctx


def load(secret_path: str = SECRET_PATH):
    """Alias for ``load_client`` — used by the unified ppflx.keys.load() dispatcher."""
    return load_client(secret_path)
