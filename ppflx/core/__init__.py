"""ppflx.core — canonical implementations of all low-level internals.

Submodules are imported directly (``from ppflx.core import zkp_gnark``); this
package imports none of them, so the library path does not pull in the
benchmark-only modules (``common``, ``data_setup``) or their dependencies.

Module layout:
  ppflx.core.benchmark         — benchmark metric collection
  ppflx.core.security          — HE/DP primitives and aggregation
  ppflx.core.model_builder     — model builder utilities
  ppflx.core.engine            — training and evaluation loops
  ppflx.core.params            — model parameter get/set helpers
  ppflx.core.zkp               — Pedersen ZKP primitives (unverified stub)
  ppflx.core.zkp_gnark         — gnark proof service client
  ppflx.core.elgamal_gnark     — verifiable ElGamal transport
  ppflx.core.gnark_keys        — pinned Groth16 key manifest
  ppflx.core.sampling          — commit–challenge coordinate sampling
  ppflx.core.update_bound      — update-norm bound
  ppflx_bench.legacy.common            — legacy utilities (benchmark side)
  ppflx_bench.legacy.data_setup        — legacy dataset helpers (benchmark side)
"""
