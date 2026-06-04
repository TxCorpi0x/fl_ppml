"""
Triple combination: HE + ZKP + DP privacy mode.

Registered modes
----------------
he_tenseal_zkp_dp       — TenSEAL CKKS + gnark ZKP + DP-SGD
he_concrete_tfhe_zkp_dp — Concrete TFHE + gnark ZKP + DP-SGD

Security architecture
---------------------
Each mechanism addresses a distinct adversarial goal:

  ┌────────────────┬──────────────────────────────────────────────────────────┐
  │ Property       │ Mechanism                                                │
  ├────────────────┼──────────────────────────────────────────────────────────┤
  │ Confidentiality│ FHE — server never sees plaintext weights                │
  │ Integrity      │ ZKP — client proves ‖w‖² ≤ bound & MiMC hash            │
  │ Privacy        │ DP-SGD — gradient clipping + Gaussian noise (ε-DP)      │
  └────────────────┴──────────────────────────────────────────────────────────┘

Client-side flow per round:
  1. Local training with DP-SGD (gradient clipping + noise addition).
     Private training data cannot be reconstructed from shared weights.
  2. Generate gnark Groth16 ZKP proofs on the DP-noised *plaintext* weights.
     Proves: MiMC_hash(w) == committed_hash  AND  Σwᵢ² ≤ bound.
  3. Encrypt the DP-noised weights with FHE — server receives only ciphertexts.

Server-side flow per round:
  - Verifies ZKP proofs (zero-knowledge, using public inputs only).
  - Excludes clients that fail verification.
  - Aggregates remaining FHE ciphertexts homomorphically.
  - Returns encrypted aggregate; clients decrypt locally.

The composition is secure: DP noise is applied first, so ZKP proves
properties of already-noised weights, and FHE encrypts the noised weights.
The server never observes plaintext gradients at any stage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from fl.privacy.base import PrivacyMode
from fl.privacy.dp import DifferentialPrivacyMode
from fl.privacy.he_concrete_tfhe import HeConcreteThfeMode
from fl.privacy.he_tenseal import HeTensealMode
from fl.privacy.he_zkp import _HeZKPCompositeMode
from fl.privacy.registry import register_mode
from fl.privacy.zkp import ZKPMode


class _HeZKPDPCompositeMode(_HeZKPCompositeMode):
    """
    Abstract base for HE + ZKP + DP triple-combination modes.

    Extends _HeZKPCompositeMode with a DP layer:
      - Client context: {"he": ..., "zkp": ..., "dp": DifferentialPrivacyParams}
      - DP-SGD is applied during local training (via fl.client.FlowerClient).
      - ZKP proofs cover the DP-noised weights (post-training).
      - HE encrypts the DP-noised weights for transport.
    """

    def __init__(self):
        super().__init__()
        self._dp_mode = DifferentialPrivacyMode()

    @property
    def name(self) -> str:
        raise NotImplementedError

    def setup_client_context(self, config) -> Dict:
        he_ctx = self._he_mode.setup_client_context(config)
        zkp_ctx = self._zkp_mode.setup_client_context(config)
        dp_ctx = self._dp_mode.setup_client_context(config)
        return {"he": he_ctx, "zkp": zkp_ctx, "dp": dp_ctx}

    def setup_server_context(self, config) -> Any:
        # Server only needs HE context; ZKP/DP are client-side.
        return self._he_mode.setup_server_context(config)

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        """Return HE + ZKP + DP timing metrics merged together."""
        he_zkp_metrics = super().post_fit_metrics(context, benchmark)
        dp_metrics = self._dp_mode.post_fit_metrics(context.get("dp"), benchmark)
        return {**he_zkp_metrics, **dp_metrics}


@register_mode("he_tenseal_zkp_dp")
class HETenSEALZKPDPMode(_HeZKPDPCompositeMode):
    """TenSEAL CKKS + gnark ZKP + DP-SGD."""

    def __init__(self):
        self._he_mode = HeTensealMode()
        super().__init__()

    @property
    def name(self) -> str:
        return "he_tenseal_zkp_dp"


@register_mode("he_concrete_tfhe_zkp_dp")
class HEConcreteTFHEZKPDPMode(_HeZKPDPCompositeMode):
    """Concrete TFHE + gnark ZKP + DP-SGD."""

    def __init__(self):
        self._he_mode = HeConcreteThfeMode()
        super().__init__()

    @property
    def name(self) -> str:
        return "he_concrete_tfhe_zkp_dp"
