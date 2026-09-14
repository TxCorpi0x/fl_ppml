"""
Hybrid HE + ZKP privacy modes.

Registered modes
----------------
he_tenseal_zkp       — TenSEAL CKKS encryption + gnark ZKP integrity proofs
he_concrete_tfhe_zkp — Concrete TFHE encryption + gnark ZKP integrity proofs

SECURITY STATUS — CONFIDENTIALITY ONLY, NO INTEGRITY
----------------------------------------------------
The ZKP proof in these modes is NOT bound to the ciphertext the server
aggregates (audit/findings.md S1-01, audit/binding.md). A client can prove an
honest vector and submit a ciphertext of a poisoned one; this is demonstrated
by tests/test_zkp_binding_attack.py. The proofs measure prover cost only and
provide no Byzantine-robustness. Use ``he_elgamal_zkp`` for ciphertext-bound
proofs.

Architecture
------------
Both modes compose an HE backend (weight confidentiality) with ZKP proof
generation whose result is not tied to the uploaded ciphertext:

  ┌────────────┬──────────────────────────────────────────────┐
  │ Property   │ Mechanism                                    │
  ├────────────┼──────────────────────────────────────────────┤
  │ Privacy    │ FHE  — server never sees plaintext weights   │
  │ Integrity  │ none — proof is over a client-chosen vector  │
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
from fl.privacy.zkp import (
    ZKPMode,
    admission_quorum,
    anchor_data,
    model_schema,
    parse_proofs,
    resolve_backend,
    round_report,
    timer,
)


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
        resolve_backend(config)
        return self._he_mode.setup_server_context(config)

    def bind_server_model(self, server_context, model) -> None:
        self._server_schema = model_schema(model)

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
        1. Check each client's proofs against the server's schema and policy,
           then verify them with ``/verify_light`` (public inputs only).
        2. Reject clients whose proofs are absent, malformed, off-policy or
           invalid; abort the round if the proof service fails.
        3. Below quorum, leave the global model unchanged.
        4. Aggregate only admitted clients with the HE backend.

        The proofs are not bound to the ciphertexts (see module docstring), so
        admission here is not an integrity guarantee.
        """
        from fl.core import zkp_gnark

        self._last_anchor_data = None
        if resolve_backend(config) != "gnark":
            self.last_round_report = round_report(
                server_round, "unverified_stub", [cp.cid for cp, _ in results], {}
            )
            return self._aggregate(server_round, list(results), failures, server_context, config, benchmark)

        schema = getattr(self, "_server_schema", None)
        if schema is None:
            raise RuntimeError("server schema not bound: make_strategy must call bind_server_model")

        admitted, rejected = [], {}
        try:
            for client_proxy, fit_res in results:
                with timer(benchmark, "proof_verification"):
                    proofs = parse_proofs(fit_res.metrics or {})
                    reason = (
                        zkp_gnark.check_proof_policy(proofs, schema, require_hash=True)
                        if proofs
                        else "missing or malformed proofs"
                    )
                    if not reason:
                        ok, failed = zkp_gnark.verify_gnark_proofs_light(proofs)
                        reason = None if ok else f"proof verification failed for {failed[:5]}"
                if reason:
                    rejected[str(client_proxy.cid)] = reason
                    print(f"[{self.name.upper()}] Round {server_round}: client {client_proxy.cid} REJECTED — {reason}")
                else:
                    admitted.append((client_proxy, fit_res, proofs))
        except zkp_gnark.GnarkServiceError as exc:
            print(f"[{self.name.upper()}] Round {server_round}: ABORTED — proof service failure, not a client fault: {exc}")
            self.last_round_report = round_report(server_round, "infrastructure_abort", [], rejected, str(exc))
            return None, {"round_outcome": "infrastructure_abort"}

        quorum = admission_quorum(config)
        if len(admitted) < quorum:
            print(f"[{self.name.upper()}] Round {server_round}: {len(admitted)} admitted < quorum {quorum} — global model unchanged.")
            self.last_round_report = round_report(
                server_round, "no_quorum", [cp.cid for cp, _, _ in admitted], rejected
            )
            return None, {"round_outcome": "no_quorum", "admitted": len(admitted), "rejected": len(rejected)}

        params, metrics = self._aggregate(
            server_round, [(cp, fr) for cp, fr, _ in admitted], failures, server_context, config, benchmark
        )
        self._last_anchor_data = anchor_data(server_round, [(str(cp.cid), proofs) for cp, _, proofs in admitted])
        self.last_round_report = round_report(server_round, "aggregated", [cp.cid for cp, _, _ in admitted], rejected)
        return params, {**metrics, "round_outcome": "aggregated", "admitted": len(admitted), "rejected": len(rejected)}

    def _aggregate(self, server_round, admitted, failures, server_context, config, benchmark) -> Tuple:
        """Aggregate exactly ``admitted`` with the HE backend, never the full result set."""
        out = self._he_mode.aggregate_fit_override(
            server_round, admitted, failures, server_context, config, benchmark
        )
        if out is not None:
            return out
        # Simulation: the HE backend transports plaintext and defers to FedAvg.
        # Average the admitted subset here so rejected clients can't re-enter
        # through the strategy's FedAvg path.
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from fl.core.security import aggregate_custom

        plain = self._he_mode.pre_aggregate(admitted, config)
        with timer(benchmark, "server_aggregate"):
            aggregated = aggregate_custom(
                [(parameters_to_ndarrays(fr.parameters), fr.num_examples) for _, fr in plain]
            )
        return ndarrays_to_parameters(aggregated), {}

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
