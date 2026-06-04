"""fl.core — canonical implementations of all low-level internals.

Module layout:
  fl.core.zkp               — Pedersen ZKP primitives
  fl.core.zkp_gnark         — gnark gRPC proof service client
"""

from . import zkp, zkp_gnark

__all__ = ["zkp", "zkp_gnark"]
