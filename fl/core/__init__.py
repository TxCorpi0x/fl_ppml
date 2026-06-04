"""fl.core — canonical implementations of all low-level internals.

Module layout:
  fl.core.benchmark         — Benchmark utilities
  fl.core.common            — Common utilities
  fl.core.security          — Security utilities
  fl.core.model_builder     — Model builder utilities
  fl.core.data_setup        — Data setup utilities
  fl.core.engine            — Core engine utilities
  fl.core.zkp               — Pedersen ZKP primitives
  fl.core.zkp_gnark         — gnark gRPC proof service client
"""

from . import (
    benchmark,
    common,
    security,
    model_builder,
    data_setup,
    engine,
    zkp,
    zkp_gnark,
)

__all__ = [
    "benchmark",
    "common",
    "security",
    "model_builder",
    "data_setup",
    "engine",
    "zkp",
    "zkp_gnark",
]
