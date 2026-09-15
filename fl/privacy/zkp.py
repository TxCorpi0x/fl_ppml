"""
Zero-Knowledge Proof (ZKP) mode.

Backends:
  gnark    — Groth16 zk-SNARK via the Go gnark service. Proves an L2 bound and
             a MiMC hash over the quantized update Δq = round(w·s) − round(g·s)
             against the global model g (audit/norm.md). Clients clip the
             update to the server's bound B first; the server recomputes Δq
             from the upload and its own global model, and requires the
             per-proof declared bounds to sum to ⌈B·s + √n⌉².
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


def state_arrays(model) -> List[np.ndarray]:
    return [t.detach().cpu().numpy().copy() for t in model.state_dict().values()]


def cast_like(arrays, like) -> List[np.ndarray]:
    """Arrays in the dtypes clients hold after loading them into the model."""
    return [np.asarray(a).astype(ref.dtype) for a, ref in zip(arrays, like)]


def quantized_update(params, global_arrays) -> List[np.ndarray]:
    """Δq = round(w·scale) − round(g·scale) per tensor, as both sides compute it."""
    from fl.core.zkp_gnark import quantize

    if len(params) != len(global_arrays):
        raise ValueError("update and global model have different tensor counts")
    return [quantize(p) - quantize(g) for p, g in zip(params, global_arrays)]


def clip_update_in_place(net, context: Dict) -> Tuple[Dict[str, np.ndarray], int]:
    """Clip net's update to the server's bound, write it back, and return (Δq per tensor, total bound).

    The clip is checked against the exact integer statement the server will
    check, including one unit per proof for proofs whose update is zero.
    """
    from fl.core import zkp_gnark
    from fl.core.update_bound import clip_update

    global_arrays = context.get("global")
    bound = context.get("max_update_norm")
    enforce = context.get("enforce_update_bound", True)
    if global_arrays is None:
        raise RuntimeError("[ZKP] no global model received; refusing to prove an update without its base")
    if bound is None and enforce:
        raise RuntimeError("[ZKP] server sent no update-norm bound")
    sd = net.state_dict()
    names = list(sd.keys())
    local = [t.detach().cpu().numpy() for t in sd.values()]
    shapes, sizes = [a.shape for a in local], [a.size for a in local]
    n = int(sum(sizes))
    total = zkp_gnark.policy_bound_sq(bound, n) if enforce else None
    n_proofs = len(zkp_gnark.expected_proof_layout([(k, s) for k, s in zip(names, shapes)]))
    g_q = np.concatenate([zkp_gnark.quantize(g).reshape(-1) for g in global_arrays])

    def fits(flat32):
        return zkp_gnark.energy(zkp_gnark.quantize(flat32) - g_q) + n_proofs <= total

    g_flat = np.concatenate([np.asarray(g, dtype=np.float64).reshape(-1) for g in global_arrays])
    l_flat = np.concatenate([a.astype(np.float64).reshape(-1) for a in local])
    if enforce:
        new_flat, norm, clipped = clip_update(g_flat, l_flat, bound, fits)
    else:
        new_flat, norm, clipped = l_flat, float(np.linalg.norm(l_flat - g_flat)), False
    context["update_norm"], context["update_clipped"] = norm, clipped

    import torch

    out, new_state, offset = {}, {}, 0
    for i, (name, shape, size, ref) in enumerate(zip(names, shapes, sizes, local)):
        # Unclipped: prove exactly the tensors that will be uploaded.
        values = new_flat[offset : offset + size].reshape(shape).astype(ref.dtype) if clipped else ref
        new_state[name] = torch.from_numpy(np.ascontiguousarray(values))
        out[name] = zkp_gnark.quantize(values) - zkp_gnark.quantize(global_arrays[i])
        offset += size
    if clipped:
        net.load_state_dict(new_state)
    return out, total


def update_metrics(context) -> Dict:
    """Clip status of the update just prepared; reported once, then cleared."""
    if not isinstance(context, dict) or "update_norm" not in context:
        return {}
    norm, clipped = context.pop("update_norm"), context.pop("update_clipped")
    return {"zkp_update_norm": float(norm), "zkp_update_clipped": int(bool(clipped))}


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

    def __init__(self, enforce_update_bound: bool = True):
        # False only for compositions whose proofs aren't bound to what the server
        # aggregates (the CKKS/TFHE + DP composites): there the bound adds no
        # integrity, and clipping DP-noised updates would distort DP training.
        self._enforce_update_bound = enforce_update_bound

    def setup_server_context(self, config) -> None:
        if resolve_backend(config) == "gnark" and self._enforce_update_bound:
            from fl.core.update_bound import max_update_norm

            self._max_update_norm = max_update_norm(config)
        return None

    def bind_server_model(self, server_context, model) -> None:
        self._server_schema = model_schema(model)
        # The global model every update is measured against; the same float32
        # values the strategy sends clients in round 1.
        self._global = state_arrays(model)

    def fit_config(self, server_round: int) -> Dict:
        from fl.core.update_bound import FIT_CONFIG_KEY

        bound = getattr(self, "_max_update_norm", None)
        return {} if bound is None else {FIT_CONFIG_KEY: str(bound)}

    # ── Client ─────────────────────────────────────────────────────────────

    def on_fit_config(self, context, fit_config: Dict) -> None:
        if isinstance(context, dict) and context.get("backend") == "gnark":
            from fl.core.update_bound import bound_from_fit_config

            context["max_update_norm"] = bound_from_fit_config(fit_config) if self._enforce_update_bound else None
            context["enforce_update_bound"] = self._enforce_update_bound

    def get_parameters(self, net, context, *, sim_mode, benchmark=None) -> List[np.ndarray]:
        """Initial parameters carry no update, so no proof."""
        return _plain_params(net)

    def send_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None) -> List[np.ndarray]:
        """Clip the update to the server's bound, prove it, then return the clipped plain params."""
        self._proof_cache = self._generate_proofs(net, context, phase="send", benchmark=benchmark)
        return _plain_params(net)

    def receive_parameters(self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None) -> None:
        from fl.core.params import set_parameters

        set_parameters(net, params, None, None)
        if isinstance(context, dict):
            context["global"] = state_arrays(net)

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
            **update_metrics(context),
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
        self._global = cast_like(aggregated, self._global)

        self._last_anchor_data = anchor_data(server_round, [(str(cp.cid), proofs) for cp, _, proofs in admitted])
        self.last_round_report = round_report(server_round, "aggregated", [cp.cid for cp, _, _ in admitted], rejected)
        return params, {"round_outcome": "aggregated", "admitted": len(admitted), "rejected": len(rejected)}

    def _check_client(self, fit_res, schema, parameters_to_ndarrays) -> Tuple[Optional[str], list]:
        """Return (rejection reason or None, proofs). Raises GnarkServiceError on infrastructure failure."""
        from fl.core import zkp_gnark

        proofs = parse_proofs(fit_res.metrics or {})
        if not proofs:
            return "missing or malformed proofs", []
        reason = zkp_gnark.check_proof_policy(
            proofs, schema, require_hash=False, total_bound_sq=self._total_bound_sq(schema)
        )
        if reason:
            return reason, []
        params = parameters_to_ndarrays(fit_res.parameters)
        if len(params) != len(schema) or any(
            tuple(np.shape(p)) != shape for p, (_, shape) in zip(params, schema)
        ):
            return "update does not match the server's model schema", []
        # The proofs cover the update against the server's own global model.
        delta = quantized_update(params, self._global)
        ok, failed = zkp_gnark.verify_gnark_proofs(delta, [name for name, _ in schema], proofs)
        if not ok:
            return f"proof verification failed for {failed[:5]}", []
        return None, proofs

    def _total_bound_sq(self, schema, n: Optional[int] = None) -> int:
        from fl.core import zkp_gnark

        if not self._enforce_update_bound:
            return None
        bound = getattr(self, "_max_update_norm", None)
        if bound is None:
            raise RuntimeError("update-norm bound not set: setup_server_context must run first")
        return zkp_gnark.policy_bound_sq(bound, n if n is not None else sum(int(np.prod(s)) for _, s in schema))

    # ── Internal helpers ───────────────────────────────────────────────────

    def _generate_proofs(self, net, context: Dict, phase: str, benchmark=None, layers=None):
        """Generate proofs and return (backend, proof_payloads, proof_bytes). Raises on failure."""
        backend = context.get("backend") if isinstance(context, dict) else None

        if backend == "gnark":
            from fl.core.zkp_gnark import generate_gnark_proofs

            with timer(benchmark, "proof_generation"):
                delta, total = clip_update_in_place(net, context)
                proof_payloads, proof_bytes = generate_gnark_proofs(delta, layers=layers, total_bound_sq=total)
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


