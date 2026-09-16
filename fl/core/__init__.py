"""fl.core — canonical implementations of all low-level internals.

Submodules are imported directly (``from fl.core import zkp_gnark``); this
package imports none of them, so the library path does not pull in the
benchmark-only modules (``common``, ``data_setup``) or their dependencies.

Module layout:
  fl.core.benchmark         — benchmark metric collection
  fl.core.security          — HE/DP primitives and aggregation
  fl.core.model_builder     — model builder utilities
  fl.core.engine            — training and evaluation loops
  fl.core.params            — model parameter get/set helpers
  fl.core.zkp               — Pedersen ZKP primitives (unverified stub)
  fl.core.zkp_gnark         — gnark proof service client
  fl.core.elgamal_gnark     — verifiable ElGamal transport
  fl.core.gnark_keys        — pinned Groth16 key manifest
  fl.core.sampling          — commit–challenge coordinate sampling
  fl.core.update_bound      — update-norm bound
  fl.core.common            — legacy utilities (benchmark side)
  fl.core.data_setup        — legacy dataset helpers (benchmark side)
"""
