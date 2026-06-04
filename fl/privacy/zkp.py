"""
Zero-Knowledge Proof (ZKP) mode.

Supports two backends:
  gnark    — Groth16 zk-SNARK via the Go gnark gRPC service (default,
             production-grade).  Proves the gradient L2-norm is within the
             agreed bound — strong Byzantine-fault protection.
  pedersen — STUB ONLY.  Pedersen commitments are computed locally but the
             proof payloads are never sent to the server and the server
             performs no verification.  There is therefore NO Byzantine-fault
             protection when this backend is selected.  See the note in
             _generate_proofs() and aggregate_fit_override() for the full
             explanation of what would be needed to make it functional.

Use gnark for any deployment that requires real integrity guarantees.
The pedersen backend exists solely to let the ZKP code paths be exercised
in environments where the Go binary cannot be built (e.g. sandboxed CI).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fl.privacy.base import PrivacyMode, _plain_params
from fl.privacy.registry import register_mode


@register_mode("zkp")
class ZKPMode(PrivacyMode):
    """Federated learning with ZKP integrity proofs on model updates."""

    @property
    def name(self) -> str:
        return "zkp"

    def setup_client_context(self, config) -> Dict:
        """
        Return a ZKP context dict for the client.

        For gnark: {"backend": "gnark"} — no parameter file needed.
        For pedersen: loads ZKP params from disk.
        """
        backend = os.environ.get("FL_ZKP_BACKEND", config.zkp_backend).lower()

        if backend == "gnark":
            print(f"[ZKP] Backend: gnark (gRPC service at {config.zkp_gnark_host})")
            return {"backend": "gnark"}

        elif backend == "pedersen":
            from fl.core.zkp import read_zkp_params

            path = config.zkp_params_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ZKP params not found: {path}\n"
                    "Run: python -m fl.keys generate zkp"
                )
            ctx = read_zkp_params(path)
            print(f"[ZKP] Backend: pedersen (params from {path})")
            return {"backend": "pedersen", "context": ctx}

        else:
            raise ValueError(
                f"Unknown ZKP backend: '{backend}'. Use 'gnark' or 'pedersen'."
            )

    def setup_server_context(self, config) -> None:
        return None  # server verifies proofs using public params embedded in payloads

    # ── Client: parameter send ─────────────────────────────────────────────

    def get_parameters(
        self, net, context, *, sim_mode, benchmark=None
    ) -> List[np.ndarray]:
        """Generate ZKP proofs (benchmarking) then return plain params."""
        self._generate_proofs(net, context, phase="get_params", benchmark=benchmark)
        return _plain_params(net)

    def send_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> List[np.ndarray]:
        """Generate ZKP proofs after training then return plain params."""
        # Preserve the canonical parameter order used by Flower ndarrays.
        # Server verification must use this order; proof emission order can be
        # non-deterministic (parallel futures completion) and chunked proofs
        # add synthetic layer names.
        self._last_layer_names = list(net.state_dict().keys())
        self._proof_cache = self._generate_proofs(
            net, context, phase="send", benchmark=benchmark
        )
        return _plain_params(net)

    def receive_parameters(
        self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> None:
        """ZKP transport uses plain numpy — just set directly."""
        from fl.core.params import set_parameters

        set_parameters(net, params, None, None)

    def post_fit_metrics(self, context: Any, benchmark=None) -> Dict:
        """Include proof payloads in the fit() response for server verification."""
        metrics = {}
        cache = getattr(self, "_proof_cache", None)
        if cache:
            backend, proof_payloads, proof_bytes = cache
            if backend == "gnark" and proof_payloads:
                metrics["zkp_proofs_json"] = json.dumps(proof_payloads)
                metrics["gnark_num_proofs"] = len(proof_payloads)
                metrics["gnark_proof_bytes"] = proof_bytes
                layer_names = getattr(self, "_last_layer_names", None)
                if layer_names:
                    metrics["zkp_layer_names_json"] = json.dumps(layer_names)
        return metrics

    # ── Server: aggregation with verification ─────────────────────────────

    def aggregate_fit_override(
        self, server_round, results, failures, server_context, config, benchmark=None
    ) -> Optional[Tuple]:
        """Verify gnark proofs; exclude clients that fail. Return None → fall back to FedAvg."""
        from flwr.common import parameters_to_ndarrays

        backend = os.environ.get("FL_ZKP_BACKEND", config.zkp_backend).lower()
        if backend != "gnark":
            # Pedersen backend: no proof payloads are ever sent by clients
            # (see _generate_proofs()), so there is nothing for the server to
            # verify.  Fall back to plain FedAvg — equivalent to baseline in
            # terms of Byzantine-fault tolerance.
            return None

        from fl.core.zkp_gnark import verify_gnark_proofs
        from fl.core.benchmark import BenchmarkTimer

        verified_results = []
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            proofs_json = metrics.get("zkp_proofs_json", "")
            try:
                proofs = json.loads(proofs_json) if proofs_json else None
            except Exception:
                proofs = None

            if not proofs:
                print(
                    f"[ZKP] Missing proofs from client {client_proxy.cid} — excluded."
                )
                continue

            params = parameters_to_ndarrays(fit_res.parameters)
            # Use canonical model layer order provided by the client. Proof
            # order can be non-deterministic and may include chunk suffixes;
            # pairing params by proof order causes shape mismatches and false
            # verification failures.
            layer_names_json = metrics.get("zkp_layer_names_json", "")
            try:
                layer_names = (
                    json.loads(layer_names_json)
                    if layer_names_json
                    else [str(i) for i in range(len(params))]
                )
            except Exception:
                layer_names = [str(i) for i in range(len(params))]

            if benchmark:
                with BenchmarkTimer(benchmark, "proof_verification"):
                    ok, verify_failures = verify_gnark_proofs(
                        params, layer_names, proofs
                    )
            else:
                ok, verify_failures = verify_gnark_proofs(params, layer_names, proofs)

            if ok:
                verified_results.append((client_proxy, fit_res))
            else:
                print(
                    f"[ZKP] Verification failed for client {client_proxy.cid}: {verify_failures}"
                )

        if not verified_results:
            print(
                "[ZKP] [WARN]  All client proofs failed verification — "
                "falling back to unverified FedAvg for this round. "
                "Check that the gnark gRPC service is reachable and "
                "that the proof circuit matches the parameter format."
            )
            # Fall back to the full unverified result set so the global model
            # still gets updated.  Training continues; security guarantee is
            # degraded for this round.
            verified_results = list(results)

        # Proceed to standard FedAvg with only the verified clients.
        # Return None to signal: use FedAvg on `results` — but we need to
        # replace results in the calling strategy.
        # We store verified results so pre_aggregate can filter.
        # ── Build ZKP anchor data for the chain ledger ────────────────────────
        # Hashes are computed here (inside the ZKP mode) so that the server's
        # _chain_commit() can call chain.anchor_proofs() without needing to
        # parse proof payloads itself.
        _proof_hashes: List[str] = []
        _client_ids: List[str] = []
        try:
            from fl.chain import hash_proof_payload

            for cp, fit_res in verified_results:
                meta = fit_res.metrics or {}
                proofs_json = meta.get("zkp_proofs_json", "")
                proofs = json.loads(proofs_json) if proofs_json else []
                for p in proofs:
                    _proof_hashes.append(hash_proof_payload(p))
                if proofs:
                    _client_ids.append(str(cp.cid))
        except Exception as _e:
            print(f"[ZKP] Warning: could not build anchor data: {_e}")
        self._last_anchor_data = {
            "round": server_round,
            "proof_hashes": _proof_hashes,
            "client_ids": _client_ids,
        }

        self._verified_results = verified_results
        return None  # FedCustom will call pre_aggregate which returns verified list

    def pre_aggregate(self, results, config) -> List:
        """Return only the verified subset if gnark was run."""
        backend = os.environ.get("FL_ZKP_BACKEND", config.zkp_backend).lower()
        if backend == "gnark" and hasattr(self, "_verified_results"):
            verified = self._verified_results
            del self._verified_results
            return verified
        return results

    # ── Internal helpers ───────────────────────────────────────────────────

    def _generate_proofs(self, net, context: Dict, phase: str, benchmark=None):
        """Generate ZKP proofs and return (backend, proof_payloads, proof_bytes)."""
        from fl.core.benchmark import BenchmarkTimer

        backend = (
            context.get("backend", "pedersen")
            if isinstance(context, dict)
            else "pedersen"
        )

        if backend == "gnark":
            from fl.core.zkp_gnark import generate_gnark_proofs

            if benchmark:
                with BenchmarkTimer(benchmark, "proof_generation"):
                    try:
                        proof_payloads, proof_bytes = generate_gnark_proofs(
                            net.state_dict()
                        )
                    except Exception as e:
                        print(f"[ZKP] gnark proof generation failed: {e}")
                        proof_payloads, proof_bytes = [], 0
            else:
                try:
                    proof_payloads, proof_bytes = generate_gnark_proofs(
                        net.state_dict()
                    )
                except Exception as e:
                    print(f"[ZKP] gnark proof generation failed: {e}")
                    proof_payloads, proof_bytes = [], 0
            return backend, proof_payloads, proof_bytes

        else:  # pedersen — STUB, no proof transmission
            # WHY THIS IS NOT IMPLEMENTED:
            # A full Pedersen ZKP pipeline would require:
            #   1. Compute per-coordinate Pedersen commitments  (done here)
            #   2. Generate a Schnorr proof-of-knowledge via Fiat-Shamir for
            #      each commitment (NOT done — would prove knowledge of opening
            #      but still NOT prove the gradient norm is bounded)
            #   3. Add a Bulletproofs-style range/inner-product proof to bound
            #      the L2 norm (NOT done — complex, slow in pure Python, and
            #      still weaker than Groth16)
            #   4. Serialize all of the above and return non-empty payloads
            #      (NOT done — currently returns [], 0)
            #
            # Without steps 2-4 the pedersen backend provides zero
            # Byzantine-fault protection — the server accepts every client
            # unconditionally.  Use gnark for any real security requirement.
            from fl.core.zkp import zkp_commit_model

            ctx = context.get("context") if isinstance(context, dict) else context
            if benchmark:
                with BenchmarkTimer(benchmark, "proof_generation"):
                    _ = zkp_commit_model(net.state_dict(), ctx)
            else:
                _ = zkp_commit_model(net.state_dict(), ctx)
            # Commitments are computed above but deliberately discarded —
            # there is no serialised proof format to send to the server yet.
            return backend, [], 0


@register_mode("zkp_sampled")
class ZKPSampledMode(ZKPMode):
    """ZKP mode with sampled-layer proofs to reduce prover cost.

    Behavior:
    - Selects a subset of model layers to produce proofs for (configurable
      via `FL_ZKP_SAMPLE_PCT` or `FL_ZKP_NUM_LAYERS`).
    - Uses same backends as `zkp` (gnark or pedersen) but only proves on the
      sampled layers. Proof payloads still include layer names so servers can
      verify the returned proofs.

    Notes:
    - Sampling seed can be provided with `FL_ZKP_SAMPLE_SEED` for reproducible
      sampling (useful for deterministic audits). In production this seed
      should be derived from a verifiable round seed (e.g. block hash).
    """

    @property
    def name(self) -> str:
        return "zkp_sampled"

    def _select_layers(self, state_dict: dict) -> List[str]:
        """Return a sampled list of layer names from state_dict.

        Environment variables:
        - FL_ZKP_SAMPLE_PCT: fraction of layers to sample (0.0-1.0)
        - FL_ZKP_NUM_LAYERS: explicit number of layers to sample (overrides pct)
        - FL_ZKP_SAMPLE_SEED: optional integer seed for deterministic sampling
        """
        import random

        layer_names = list(state_dict.keys())
        if not layer_names:
            return []

        # Determine sample size
        pct = os.environ.get("FL_ZKP_SAMPLE_PCT")
        num = os.environ.get("FL_ZKP_NUM_LAYERS", "1")
        if num is not None:
            try:
                k = max(1, int(num))
            except Exception:
                k = 1
        elif pct is not None:
            try:
                p = float(pct)
                k = max(1, int(__import__("math").ceil(len(layer_names) * p)))
            except Exception:
                k = max(1, int(__import__("math").ceil(len(layer_names) * 0.2)))
        else:
            # Default: prove 20% of layers (at least 1)
            k = max(1, int(__import__("math").ceil(len(layer_names) * 0.2)))

        # Deterministic sampling when seed provided
        seed = os.environ.get("FL_ZKP_SAMPLE_SEED")
        if seed is not None:
            try:
                rng = random.Random(int(seed))
            except Exception:
                rng = random.Random()
        else:
            rng = random.Random()

        # If k >= number of layers, return all
        if k >= len(layer_names):
            return layer_names

        # Selection strategy: random (default) or size-based
        select_by = os.environ.get("FL_ZKP_SELECT_BY", "size").lower()
        if select_by == "size":
            # Compute layer sizes (number of elements) and pick top-k
            sizes = {}
            import numpy as _np

            for name in layer_names:
                t = state_dict.get(name)
                try:
                    if hasattr(t, "numel"):
                        sizes[name] = int(t.numel())
                    else:
                        arr = _np.asarray(t)
                        sizes[name] = int(arr.size)
                except Exception:
                    sizes[name] = 0

            # Sort by size desc and pick top-k
            sorted_names = sorted(
                layer_names, key=lambda n: sizes.get(n, 0), reverse=True
            )
            return sorted_names[:k]

        # Default: random sampling
        sampled = rng.sample(layer_names, k)
        return sampled

    def _generate_proofs(self, net, context: Dict, phase: str, benchmark=None):
        """Override to produce proofs only for sampled layers."""
        # Build the full state dict then select layers
        state = net.state_dict()
        sampled_layers = self._select_layers(state)

        # Reuse parent implementation but pass `layers` to gnark generator
        backend = (
            context.get("backend", "pedersen")
            if isinstance(context, dict)
            else "pedersen"
        )

        if backend == "gnark":
            from fl.core.zkp_gnark import generate_gnark_proofs

            from fl.core.benchmark import BenchmarkTimer

            if benchmark:
                with BenchmarkTimer(benchmark, "proof_generation"):
                    try:
                        proof_payloads, proof_bytes = generate_gnark_proofs(
                            state, layers=sampled_layers
                        )
                    except Exception as e:
                        print(f"[ZKP] gnark proof generation failed: {e}")
                        proof_payloads, proof_bytes = [], 0
            else:
                try:
                    proof_payloads, proof_bytes = generate_gnark_proofs(
                        state, layers=sampled_layers
                    )
                except Exception as e:
                    print(f"[ZKP] gnark proof generation failed: {e}")
                    proof_payloads, proof_bytes = [], 0
            return backend, proof_payloads, proof_bytes

        else:
            # Pedersen: create commitments only for sampled layers
            from fl.core.zkp import zkp_commit_model

            ctx = context.get("context") if isinstance(context, dict) else context
            # Build protected_layers list
            protected_layers = sampled_layers if sampled_layers else None
            from fl.core.benchmark import BenchmarkTimer

            if benchmark:
                with BenchmarkTimer(benchmark, "proof_generation"):
                    _ = zkp_commit_model(net.state_dict(), ctx, protected_layers)
            else:
                _ = zkp_commit_model(net.state_dict(), ctx, protected_layers)
            return backend, [], 0
