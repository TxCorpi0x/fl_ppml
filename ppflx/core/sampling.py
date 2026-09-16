"""
Server-seeded coordinate sampling for commit–challenge proofs.

Protocol (docs/ZKP.md, section 6.4): a client first commits to its whole
update; only then does the server draw a fresh seed, and the client proves the
coordinates selected by that seed. Because the seed doesn't exist until the
update is fixed, a client can't choose which coordinates to leave unproven.

Selection is a partial Fisher–Yates shuffle driven by SHA-256 of the seed,
so anyone holding the seed can reproduce it exactly, independent of Python or
numpy versions.
"""

from __future__ import annotations

import hashlib
import math
import os
import secrets

import numpy as np

DEFAULT_SAMPLE_RATE = 0.1
SAMPLE_RATE_ENV = "FL_ZKP_SAMPLE_PCT"
_MIN_SEED_BYTES = 16


def sample_rate_from_env() -> float:
    """Fraction of coordinates to prove per client per round, in (0, 1]."""
    raw = os.environ.get(SAMPLE_RATE_ENV, str(DEFAULT_SAMPLE_RATE))
    rate = float(raw)
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"{SAMPLE_RATE_ENV} must be in (0, 1], got {raw!r}")
    return rate


def sample_size(n: int, rate: float) -> int:
    """Number of coordinates sampled from n at the given rate (at least 1)."""
    if n <= 0:
        raise ValueError("cannot sample from an empty model")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"sample rate must be in (0, 1], got {rate}")
    return min(n, max(1, math.ceil(rate * n)))


def new_round_seed() -> str:
    """A fresh 256-bit seed, drawn by the server after commitments are received."""
    return secrets.token_hex(32)


def _uniform_below(seed: bytes, i: int, bound: int) -> int:
    """Unbiased integer in [0, bound) derived from SHA-256(seed ‖ i ‖ counter)."""
    limit = (1 << 64) - ((1 << 64) % bound)
    counter = 0
    while True:
        digest = hashlib.sha256(seed + i.to_bytes(8, "big") + counter.to_bytes(4, "big")).digest()
        x = int.from_bytes(digest[:8], "big")
        if x < limit:
            return x % bound
        counter += 1


def sample_indices(seed_hex: str, n: int, s: int) -> np.ndarray:
    """s distinct coordinates of [0, n), sorted, reproducible from the seed alone."""
    seed = bytes.fromhex(seed_hex)
    if len(seed) < _MIN_SEED_BYTES:
        raise ValueError(f"seed must be at least {_MIN_SEED_BYTES} bytes")
    if not 0 < s <= n:
        raise ValueError(f"sample size {s} outside (0, {n}]")
    swapped: dict = {}
    for i in range(s):
        j = i + _uniform_below(seed, i, n - i)
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return np.array(sorted(swapped[i] for i in range(s)), dtype=np.int64)


def detection_probability(n: int, s: int, m: int) -> float:
    """P(at least one of m bad coordinates is among s sampled of n).

    Hypergeometric: 1 − C(n−m, s) / C(n, s). This holds against a client that
    must fix its update before the seed exists; it does not hold if the client
    can see the seed first or retry the commitment.
    """
    if not 0 <= m <= n or not 0 <= s <= n:
        raise ValueError("require 0 ≤ m ≤ n and 0 ≤ s ≤ n")
    if m == 0 or s == 0:
        return 0.0
    if m > n - s:
        return 1.0
    log_miss = sum(math.log(n - s - i) - math.log(n - i) for i in range(m))
    return 1.0 - math.exp(log_miss)
