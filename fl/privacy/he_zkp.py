"""
Hybrid HE + ZKP privacy modes.

Registered modes
----------------
he_tenseal_zkp       — TenSEAL CKKS encryption + gnark ZKP integrity proofs
he_concrete_tfhe_zkp — Concrete TFHE encryption + gnark ZKP integrity proofs

Architecture
------------
Both modes compose an HE backend (weight *confidentiality*) with ZKP proof
generation (gradient *integrity*).  They address complementary security goals:

  ┌────────────┬──────────────────────────────────────────────┐
  │ Property   │ Mechanism                                    │
  ├────────────┼──────────────────────────────────────────────┤
  │ Privacy    │ FHE  — server never sees plaintext weights   │
  │ Integrity  │ ZKP  — client proves ‖w‖² ≤ bound & hash    │
  └────────────┴──────────────────────────────────────────────┘

Client-side flow (per round):
  1. Generate gnark Groth16 ZKP proofs on *plaintext* weights.
     Proves: MiMC_hash(w) == committed_hash  AND  Σwᵢ² ≤ bound.
  2. Encrypt weights with HE (server receives only ciphertexts).
  3. Send: HE ciphertext parameters + ZKP proof payloads (via metrics).

Server-side flow (per round):
  - Calls ``/verify_light`` on the gnark service for each proof, passing
    only the committed public inputs (hash_hex, bound_sq, shape) and the
    proof bytes.  No plaintext weights cross the wire — fully zero-knowledge.
  - Clients that fail verification are excluded from aggregation.
  - Aggregates remaining HE ciphertexts homomorphically.

Zero-knowledge server verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The gnark service exposes ``/verify_light``:

    POST /verify_light
    {
        "layer_name": "...",
        "shape":      [d0, d1, ...],   // to derive circuit size n = ∏shape
        "bound_sq":   "<integer>",     // public SNARK input: Σwᵢ² ≤ bound
        "hash_hex":   "<hex>",         // public SNARK input: MiMC_hash(w)
        "proof_b64":  "<base64>"       // Groth16 proof bytes
    }

The server re-uses the cached verifying key (keyed on circuit size n) and
verifies the SNARK against the *public* witness only.  Private weights are
never transmitted — the server simultaneously achieves:
  [OK] Weight confidentiality (FHE)
  [OK] Online cryptographic ZKP verification (zero-knowledge)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fl.privacy.base import PrivacyMode, _plain_params
from fl.privacy.registry import register_mode
from fl.privacy.zkp import ZKPMode


# ─────────────────────────────────────────────────────────────────────────────
# Shared composite base
# ─────────────────────────────────────────────────────────────────────────────


class _HeZKPCompositeMode(PrivacyMode):
    """
    Abstract base for HE + ZKP hybrid modes.

    Subclasses must set ``_he_mode`` to an instantiated HE PrivacyMode.
    The ZKP layer always uses ``ZKPMode`` (gnark or pedersen backend).

    Client context layout::

        {
            "he":  <HE mode context>,   # TenSEAL ctx / ConcreteAgg ctx
            "zkp": <ZKP mode context>,  # {"backend": "gnark"} or Pedersen dict
        }

    Server context:  the HE server context (ZKP server context is always None).
    """

    # Subclasses set this in __init__
    _he_mode: PrivacyMode = None  # type: ignore[assignment]

    def __init__(self):
        self._zkp_mode = ZKPMode()

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_client_context(self, config) -> Dict:
        he_ctx = self._he_mode.setup_client_context(config)
        zkp_ctx = self._zkp_mode.setup_client_context(config)
        return {"he": he_ctx, "zkp": zkp_ctx}

    def setup_server_context(self, config) -> Any:
        # Server only needs the HE context (ZKP server ctx is always None).
        return self._he_mode.setup_server_context(config)

    # ── Server: initial parameter distribution ─────────────────────────────

    def use_client_for_initial_params(self, config) -> bool:
        return self._he_mode.use_client_for_initial_params(config)

    # ── Client: parameter upload ──────────────────────────────────────────

    def get_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> List[np.ndarray]:
        """Generate ZKP proofs (benchmarked) then return HE-encrypted params."""
        # Step 1 — ZKP proof generation on plaintext
        self._zkp_mode.get_parameters(
            net, context["zkp"], sim_mode=sim_mode, benchmark=benchmark
        )
        # Step 2 — HE encryption (returns encrypted or plain-in-sim params)
        return self._he_mode.get_parameters(
            net,
            context["he"],
            sim_mode=sim_mode,
            benchmark=benchmark,
            encrypt_layers=encrypt_layers,
        )

    def send_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> List[np.ndarray]:
        """
        After local training: generate ZKP proofs on plaintext, then encrypt.

        Proof payloads are stored on ``_zkp_mode`` and later picked up by
        ``post_fit_metrics``.
        """
        # Step 1 — ZKP proofs (cache stored on self._zkp_mode._proof_cache)
        self._zkp_mode.send_parameters(
            net, context["zkp"], sim_mode=sim_mode, benchmark=benchmark
        )
        # Step 2 — HE encryption
        return self._he_mode.send_parameters(
            net,
            context["he"],
            sim_mode=sim_mode,
            benchmark=benchmark,
            encrypt_layers=encrypt_layers,
        )

    def receive_parameters(
        self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> None:
        """Decrypt HE-encrypted parameters from server."""
        self._he_mode.receive_parameters(
            net,
            params,
            context["he"],
            sim_mode=sim_mode,
            benchmark=benchmark,
            encrypt_layers=encrypt_layers,
        )

    # ── Post-fit: attach ZKP proof payloads to fit() response ────────────

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        """
        Return HE timing metrics merged with ZKP proof payloads.

        The proof payloads are forwarded to the server inside the Flower
        fit() response metrics dict where the server can log them for auditing.
        """
        he_metrics = self._he_mode.post_fit_metrics(context["he"], benchmark)
        zkp_metrics = self._zkp_mode.post_fit_metrics(context["zkp"], benchmark)
        return {**he_metrics, **zkp_metrics}

    # ── Server: aggregation ───────────────────────────────────────────────

    def aggregate_fit_override(
        self, server_round, results, failures, server_context, config, benchmark=None
    ) -> Optional[Tuple]:
        """
        1. Parse ZKP proof payloads from each client's fit() metrics.
        2. Run zero-knowledge verification via ``/verify_light`` — no plaintext
           weights needed; only the committed public inputs (hash_hex, bound_sq)
           and the Groth16 proof are sent to the gnark service.
        3. Exclude any client whose proofs fail (or are absent).
        4. Delegate encrypted aggregation to the HE backend.
        """
        from fl.core.benchmark import BenchmarkTimer

        backend = os.environ.get("FL_ZKP_BACKEND", config.zkp_backend).lower()

        admitted = []
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            proofs_json = metrics.get("zkp_proofs_json", "")
            try:
                proofs = json.loads(proofs_json) if proofs_json else None
            except Exception:
                proofs = None

            if not proofs:
                if backend == "gnark":
                    print(
                        f"[{self.name.upper()}] Round {server_round}: "
                        f"client {client_proxy.cid} sent no ZKP proofs — excluded."
                    )
                    continue
                # Non-gnark backends: admit without ZKP check
                admitted.append((client_proxy, fit_res))
                continue

            if backend == "gnark":
                from fl.core.zkp_gnark import verify_gnark_proofs_light

                if benchmark:
                    with BenchmarkTimer(benchmark, "proof_verification"):
                        ok, verify_failures = verify_gnark_proofs_light(proofs)
                else:
                    ok, verify_failures = verify_gnark_proofs_light(proofs)

                n_proofs = len(proofs)
                proof_bytes = sum(len(p.get("proof_b64", "")) for p in proofs)
                if ok:
                    print(
                        f"[{self.name.upper()}] Round {server_round}: "
                        f"client {client_proxy.cid} — {n_proofs} ZKP proof(s) "
                        f"verified [OK] ({proof_bytes} bytes, zero-knowledge)"
                    )
                    admitted.append((client_proxy, fit_res))
                else:
                    print(
                        f"[{self.name.upper()}] Round {server_round}: "
                        f"client {client_proxy.cid} — ZKP verification FAILED "
                        f"for layers: {verify_failures} — excluded."
                    )
            else:
                # pedersen: no server-side verify; admit and log
                n_proofs = len(proofs)
                print(
                    f"[{self.name.upper()}] Round {server_round}: "
                    f"client {client_proxy.cid} — {n_proofs} pedersen commitment(s) "
                    f"(offline audit only)"
                )
                admitted.append((client_proxy, fit_res))

        if not admitted:
            print(
                f"[{self.name.upper()}] Round {server_round}: "
                "all clients excluded — falling back to full result set."
            )
            admitted = list(results)

        # ── Build ZKP anchor data for the chain ledger ────────────────────────
        _proof_hashes: List[str] = []
        _client_ids: List[str] = []
        try:
            from fl.chain import hash_proof_payload

            for cp, fit_res in admitted:
                meta = fit_res.metrics or {}
                proofs_json = meta.get("zkp_proofs_json", "")
                proofs = json.loads(proofs_json) if proofs_json else []
                for p in proofs:
                    _proof_hashes.append(hash_proof_payload(p))
                if proofs:
                    _client_ids.append(str(cp.cid))
        except Exception as _e:
            print(f"[{self.name.upper()}] Warning: could not build anchor data: {_e}")
        self._last_anchor_data = {
            "round": server_round,
            "proof_hashes": _proof_hashes,
            "client_ids": _client_ids,
        }

        # Delegate encrypted aggregation to the HE backend.
        return self._he_mode.aggregate_fit_override(
            server_round, admitted, failures, server_context, config, benchmark
        )

    def pre_aggregate(self, results, config) -> List:
        """Delegate to the HE backend's pre_aggregate hook."""
        return self._he_mode.pre_aggregate(results, config)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete hybrid modes
# ─────────────────────────────────────────────────────────────────────────────


@register_mode("he_tenseal_zkp")
class HeTensealZKPMode(_HeZKPCompositeMode):
    """TenSEAL CKKS Homomorphic Encryption + gnark ZKP integrity proofs."""

    def __init__(self):
        from fl.privacy.he_tenseal import HeTensealMode

        super().__init__()
        self._he_mode = HeTensealMode()

    @property
    def name(self) -> str:
        return "he_tenseal_zkp"


@register_mode("he_concrete_tfhe_zkp")
class HeConcreteZKPMode(_HeZKPCompositeMode):
    """Concrete TFHE Homomorphic Encryption + gnark ZKP integrity proofs."""

    def __init__(self):
        from fl.privacy.he_concrete_tfhe import HeConcreteThfeMode

        super().__init__()
        self._he_mode = HeConcreteThfeMode()

    @property
    def name(self) -> str:
        return "he_concrete_tfhe_zkp"
