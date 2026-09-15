"""
Zero-Knowledge Proof (ZKP) mode.

Backends:
  gnark    — Groth16 zk-SNARK via the Go gnark service. Proves an L2 bound and
             a MiMC hash over the submitted plaintext weights.
  pedersen — STUB ONLY. Pedersen commitments are computed locally but never
             sent or verified, so there is no Byzantine-fault protection.
             Refused unless FL_ZKP_ALLOW_PEDERSEN_STUB=1; when allowed, every
             round is recorded as ``unverified_stub``.

Failure handling (audit/failmodes.md):
  client — proof generation failure raises; an update is never uploaded
           without proofs.
  server — each client is admitted or rejected against the server's own model
           schema and proof policy (coverage, shapes, scale, bound). Rejection
           never raises. A proof-service failure aborts the round. With fewer
           admitted clients than the quorum, the global model is unchanged and
           nothing is committed or anchored.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fl.privacy.base import PrivacyMode, _plain_params
from fl.privacy.registry import register_mode

ALLOW_STUB_ENV = "FL_ZKP_ALLOW_PEDERSEN_STUB"


# ─────────────────────────────────────────────────────────────────────────────
# Shared admission helpers (also used by the HE + ZKP composites)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_backend(config) -> str:
    """Return the ZKP backend, refusing the unverified pedersen stub by default."""
    backend = os.environ.get("FL_ZKP_BACKEND", getattr(config, "zkp_backend", "gnark")).lower()
    if backend == "gnark":
        return backend
    if backend == "pedersen":
        if os.environ.get(ALLOW_STUB_ENV, "0") == "1":
            print("[ZKP] [WARN] pedersen stub: proofs are NOT verified; rounds are recorded as unverified.")
            return backend
        raise RuntimeError(
            "ZKP backend 'pedersen' performs no verification. Use gnark, or set "
            f"{ALLOW_STUB_ENV}=1 to run the stub knowingly."
        )
    raise ValueError(f"Unknown ZKP backend: '{backend}'. Use 'gnark' or 'pedersen'.")


def admission_quorum(config) -> int:
    """Minimum admitted clients for a round to update the model."""
    return max(1, int(getattr(config, "min_fit_clients", 1) or 1))


def round_report(server_round: int, outcome: str, admitted, rejected: Dict, detail: str = "") -> Dict:
    report = {
        "round": server_round,
        "outcome": outcome,
        "admitted": [str(cid) for cid in admitted],
        "rejected": {str(cid): reason for cid, reason in rejected.items()},
    }
    if detail:
        report["detail"] = detail
    try:
        from fl.core.gnark_keys import manifest_sha256

        # Which pinned verifying keys this round's proofs were checked under (audit/setup.md).
        report["key_manifest_sha256"] = manifest_sha256()
    except (OSError, ValueError, KeyError):
        pass
    return report


def parse_proofs(metrics: Dict) -> Optional[list]:
    try:
        proofs = json.loads(metrics.get("zkp_proofs_json", ""))
    except (TypeError, ValueError):
        return None
    return proofs if isinstance(proofs, list) and proofs else None


def anchor_data(server_round: int, admitted_proofs: List[Tuple[str, list]]) -> Dict:
    from fl.chain import hash_proof_payload

    return {
        "round": server_round,
        "proof_hashes": [hash_proof_payload(p) for _, proofs in admitted_proofs for p in proofs],
        "client_ids": [cid for cid, _ in admitted_proofs],
    }


def timer(benchmark, name: str):
    from fl.core.benchmark import BenchmarkTimer

    return BenchmarkTimer(benchmark, name) if benchmark else nullcontext()


def model_schema(model) -> List[Tuple[str, Tuple[int, ...]]]:
    return [(name, tuple(int(d) for d in t.shape)) for name, t in model.state_dict().items()]


# ─────────────────────────────────────────────────────────────────────────────
# Mode
# ─────────────────────────────────────────────────────────────────────────────


@register_mode("zkp")
class ZKPMode(PrivacyMode):
    """Federated learning with ZKP integrity proofs on submitted weights."""

    @property
    def name(self) -> str:
        return "zkp"

    def setup_client_context(self, config) -> Dict:
        """Return {"backend": "gnark"}, or the pedersen stub context when explicitly allowed."""
        backend = resolve_backend(config)
        if backend == "gnark":
            print(f"[ZKP] Backend: gnark (service at {config.zkp_gnark_host})")
            return {"backend": "gnark"}

        from fl.core.zkp import read_zkp_params

        path = config.zkp_params_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"ZKP params not found: {path}\nRun: python -m fl.keys generate zkp")
        print(f"[ZKP] Backend: pedersen stub (params from {path})")
        return {"backend": "pedersen", "context": read_zkp_params(path)}

    def setup_server_context(self, config) -> None:
        resolve_backend(config)
        return None

    def bind_server_model(self, server_context, model) -> None:
        self._server_schema = model_schema(model)

    # ── Client ─────────────────────────────────────────────────────────────

    def get_parameters(self, net, context, *, sim_mode, benchmark=None) -> List[np.ndarray]:
        """Generate ZKP proofs (benchmarking) then return plain params."""
        self._generate_proofs(net, context, phase="get_params", benchmark=benchmark)
        return _plain_params(net)

    def send_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None) -> List[np.ndarray]:
        """Generate ZKP proofs after training, then return plain params."""
        self._proof_cache = self._generate_proofs(net, context, phase="send", benchmark=benchmark)
        return _plain_params(net)

    def receive_parameters(self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None) -> None:
        from fl.core.params import set_parameters

        set_parameters(net, params, None, None)

    def post_fit_metrics(self, context: Any, benchmark=None) -> Dict:
        """Include proof payloads in the fit() response for server verification."""
        cache = getattr(self, "_proof_cache", None)
        if not cache:
            return {}
        backend, proof_payloads, proof_bytes = cache
        if backend != "gnark":
            return {}
        return {
            "zkp_proofs_json": json.dumps(proof_payloads),
            "gnark_num_proofs": len(proof_payloads),
            "gnark_proof_bytes": proof_bytes,
        }

    # ── Server ─────────────────────────────────────────────────────────────

    def aggregate_fit_override(self, server_round, results, failures, server_context, config, benchmark=None) -> Optional[Tuple]:
        """Verify each client against server policy and FedAvg the admitted subset."""
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        from fl.core import zkp_gnark
        from fl.core.security import aggregate_custom

        self._last_anchor_data = None
        if resolve_backend(config) != "gnark":
            self.last_round_report = round_report(
                server_round, "unverified_stub", [cp.cid for cp, _ in results], {}
            )
            return None  # FedAvg over every client, recorded as unverified

        schema = getattr(self, "_server_schema", None)
        if schema is None:
            raise RuntimeError("server schema not bound: make_strategy must call bind_server_model")

        admitted, rejected = [], {}
        try:
            for client_proxy, fit_res in results:
                with timer(benchmark, "proof_verification"):
                    reason, proofs = self._check_client(fit_res, schema, parameters_to_ndarrays)
                if reason:
                    rejected[str(client_proxy.cid)] = reason
                    print(f"[ZKP] Round {server_round}: client {client_proxy.cid} REJECTED — {reason}")
                else:
                    admitted.append((client_proxy, fit_res, proofs))
        except zkp_gnark.GnarkServiceError as exc:
            print(f"[ZKP] Round {server_round}: ABORTED — proof service failure, not a client fault: {exc}")
            self.last_round_report = round_report(server_round, "infrastructure_abort", [], rejected, str(exc))
            return None, {"round_outcome": "infrastructure_abort"}

        quorum = admission_quorum(config)
        if len(admitted) < quorum:
            print(f"[ZKP] Round {server_round}: {len(admitted)} admitted < quorum {quorum} — global model unchanged.")
            self.last_round_report = round_report(
                server_round, "no_quorum", [cp.cid for cp, _, _ in admitted], rejected
            )
            return None, {"round_outcome": "no_quorum", "admitted": len(admitted), "rejected": len(rejected)}

        with timer(benchmark, "server_aggregate"):
            aggregated = aggregate_custom(
                [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res, _ in admitted]
            )
            params = ndarrays_to_parameters(aggregated)

        self._last_anchor_data = anchor_data(server_round, [(str(cp.cid), proofs) for cp, _, proofs in admitted])
        self.last_round_report = round_report(server_round, "aggregated", [cp.cid for cp, _, _ in admitted], rejected)
        return params, {"round_outcome": "aggregated", "admitted": len(admitted), "rejected": len(rejected)}

    @staticmethod
    def _check_client(fit_res, schema, parameters_to_ndarrays) -> Tuple[Optional[str], list]:
        """Return (rejection reason or None, proofs). Raises GnarkServiceError on infrastructure failure."""
        from fl.core import zkp_gnark

        proofs = parse_proofs(fit_res.metrics or {})
        if not proofs:
            return "missing or malformed proofs", []
        reason = zkp_gnark.check_proof_policy(proofs, schema, require_hash=False)
        if reason:
            return reason, []
        params = parameters_to_ndarrays(fit_res.parameters)
        if len(params) != len(schema) or any(
            tuple(np.shape(p)) != shape for p, (_, shape) in zip(params, schema)
        ):
            return "update does not match the server's model schema", []
        ok, failed = zkp_gnark.verify_gnark_proofs(params, [name for name, _ in schema], proofs)
        if not ok:
            return f"proof verification failed for {failed[:5]}", []
        return None, proofs

    # ── Internal helpers ───────────────────────────────────────────────────

    def _generate_proofs(self, net, context: Dict, phase: str, benchmark=None, layers=None):
        """Generate proofs and return (backend, proof_payloads, proof_bytes). Raises on failure."""
        backend = context.get("backend") if isinstance(context, dict) else None

        if backend == "gnark":
            from fl.core.zkp_gnark import generate_gnark_proofs

            with timer(benchmark, "proof_generation"):
                proof_payloads, proof_bytes = generate_gnark_proofs(net.state_dict(), layers=layers)
            if not proof_payloads:
                raise RuntimeError("[ZKP] no proofs generated; refusing to upload an unproven update")
            return backend, proof_payloads, proof_bytes

        if backend == "pedersen":
            # STUB: commitments are computed and discarded. There is no
            # serialised proof format; the server verifies nothing.
            from fl.core.zkp import zkp_commit_model

            with timer(benchmark, "proof_generation"):
                zkp_commit_model(net.state_dict(), context.get("context"), layers)
            return backend, [], 0

        raise RuntimeError(f"[ZKP] unknown client context backend: {backend!r}")


