"""
Verifiable additive HE + ZKP mode (audit/binding.md Design B + A).

Unlike the CKKS/TFHE composites in he_zkp.py, the proof here is bound to the
exact ciphertexts the server aggregates:

  Client, per round
    q = round(w · scale)                               per coordinate, |q| < 2^17
    C = (r·G, (q + 2^17)·G + r·PK)                     exponential ElGamal, BabyJubJub
    π_chunk proves: every C in the chunk encrypts an in-range q, and Σq² ≤ bound_chunk
    public inputs: PK, bound_chunk, context(round, layer, chunk), the chunk's ciphertexts

  Server, per round
    schema, chunking, bounds and context come from the server's own model and
    Policy, never from client metadata. Each chunk proof is verified against the
    ciphertext bytes the server received. A client is aggregated only if every
    expected chunk verifies. Aggregation is Σ num_examples_k · C_k.
    No admitted client, or an unreachable service, leaves the global model
    unchanged for that round (fail closed).

  Client, on download
    decrypt Σ n_k·(q_k + 2^17) with the shared secret key, subtract the offset,
    divide by Σ n_k · scale.

Guarantees: an aggregated update is exactly the norm-bounded, range-checked
vector its proofs cover; a proof cannot be moved to another ciphertext,
position or round.

Does NOT guarantee: honest training; confidentiality against holders of the
shared client key (the same trust model as the TenSEAL composites); a bound on
the update rather than the weights (findings.md S1-07); a trusted setup
independent of the proving service (S1-08); client-identity binding (Flower's
server-side client ids differ from client-side ids, so identity is not a
public input). Round-1 download is the server's plaintext initial model.
"""

from __future__ import annotations

import concurrent.futures
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fl.core import elgamal_gnark as eg
from fl.privacy.base import PrivacyMode
from fl.privacy.registry import register_mode


def _timer(benchmark, name):
    from contextlib import nullcontext

    from fl.core.benchmark import BenchmarkTimer

    return BenchmarkTimer(benchmark, name) if benchmark else nullcontext()


@register_mode("he_elgamal_zkp")
class HeElGamalZKPMode(PrivacyMode):
    """Exponential ElGamal on BabyJubJub with ciphertext-bound Groth16 proofs."""

    @property
    def name(self) -> str:
        return "he_elgamal_zkp"

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_client_context(self, config) -> Dict:
        if config.sim_mode:
            raise RuntimeError("he_elgamal_zkp transports real ciphertexts; simulation mode is not supported")
        from fl.keys.he_elgamal import load_client

        keys = load_client(config.he_elgamal_secret_path)
        return {"sk": keys["sk"], "pk": keys["pk"], "policy": eg.Policy.from_env(), "round": None}

    def setup_server_context(self, config) -> Dict:
        if config.sim_mode:
            raise RuntimeError("he_elgamal_zkp transports real ciphertexts; simulation mode is not supported")
        from fl.keys.he_elgamal import load_server

        return {"pk": load_server(config.he_elgamal_public_path), "policy": eg.Policy.from_env(), "schema": None}

    def bind_server_model(self, server_context, model) -> None:
        server_context["schema"] = eg.schema_of(model.state_dict())

    def on_fit_config(self, context, fit_config: Dict) -> None:
        context["round"] = int(fit_config["server_round"])

    def use_client_for_initial_params(self, config) -> bool:
        return False

    # ── Client ────────────────────────────────────────────────────────────

    def get_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        raise RuntimeError("he_elgamal_zkp never uploads parameters outside a proven fit()")

    def send_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        server_round = context.get("round")
        if server_round is None:
            raise RuntimeError("fit round unknown: on_fit_config was not called before send_parameters")
        policy: eg.Policy = context["policy"]
        state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        schema = eg.schema_of(state)
        chunks = eg.chunks_for(schema, policy.chunk_size)
        total = sum(eg.numel(shape) for _, shape in schema)
        quantized = [eg.quantize(state[name].reshape(-1), policy.scale, name) for name, _ in schema]

        def _prove(chunk: eg.Chunk):
            q = quantized[chunk.layer][chunk.start : chunk.start + chunk.size]
            return eg.prove_chunk(
                context["pk"],
                q,
                eg.chunk_bound_sq(policy, chunk, total),
                eg.context_value(server_round, chunk),
            )

        workers = max(1, zkp_parallelism())
        with _timer(benchmark, "proof_generation"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                outputs = list(pool.map(_prove, chunks))

        layer_bytes = [bytearray(eg.numel(shape) * eg.CIPHERTEXT_BYTES) for _, shape in schema]
        proofs = []
        for chunk, (ct, proof_b64) in zip(chunks, outputs):
            offset = chunk.start * eg.CIPHERTEXT_BYTES
            layer_bytes[chunk.layer][offset : offset + len(ct)] = ct
            proofs.append({"layer": chunk.layer, "chunk": chunk.index, "proof_b64": proof_b64, "vk_sha256": eg.pinned_vk()})

        self._proof_cache = (proofs, sum(len(p["proof_b64"]) for p in proofs))
        header = np.array([eg.HEADER_MAGIC, 1, server_round], dtype=np.int64)
        return [header] + [np.frombuffer(bytes(b), dtype=np.uint8) for b in layer_bytes]

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        proofs, proof_bytes = getattr(self, "_proof_cache", ([], 0))
        if not proofs:
            return {}
        return {
            "zkp_proofs_json": json.dumps(proofs),
            "gnark_num_proofs": len(proofs),
            "gnark_proof_bytes": proof_bytes,
        }

    def receive_parameters(self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        from fl.core.params import set_parameters

        params = list(params)
        names = list(net.state_dict().keys())
        shapes = [tuple(t.shape) for t in net.state_dict().values()]
        if params and _is_header(params[0]):
            total_weight = int(params[0][1])
            layers = params[1:]
            if total_weight <= 0 or len(layers) != len(names):
                raise ValueError("malformed encrypted global model")
            policy: eg.Policy = context["policy"]
            plain = []
            with _timer(benchmark, "decryption"):
                for name, shape, ct in zip(names, shapes, layers):
                    sums = eg.decrypt(context["sk"], ct.tobytes(), total_weight)
                    if sums.size != eg.numel(shape):
                        raise ValueError(f"layer '{name}': decrypted {sums.size} values for shape {shape}")
                    plain.append((sums / (total_weight * policy.scale)).astype(np.float32).reshape(shape))
            set_parameters(net, plain, None, None)
            return
        # Round-1 download: the server's plaintext initial model.
        if len(params) != len(names) or any(np.asarray(p).shape != s for p, s in zip(params, shapes)):
            raise ValueError("unexpected parameters: neither an encrypted aggregate nor the initial model")
        set_parameters(net, [np.asarray(p, dtype=np.float32) for p in params], None, None)

    # ── Server ────────────────────────────────────────────────────────────

    def aggregate_fit_override(
        self, server_round, results, failures, server_context, config, benchmark=None
    ) -> Optional[Tuple]:
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        schema = server_context.get("schema")
        if schema is None:
            raise RuntimeError("server schema not bound: make_strategy must call bind_server_model")
        policy: eg.Policy = server_context["policy"]
        chunks = eg.chunks_for(schema, policy.chunk_size)
        total = sum(eg.numel(shape) for _, shape in schema)
        self._last_anchor_data = None

        from fl.privacy.zkp import admission_quorum, anchor_data, round_report

        admitted, rejected = [], {}
        try:
            for client_proxy, fit_res in results:
                arrays = parameters_to_ndarrays(fit_res.parameters)
                with _timer(benchmark, "proof_verification"):
                    reason, proofs = self._check_client(
                        arrays, fit_res.metrics or {}, server_round, schema, chunks, total, policy, server_context["pk"]
                    )
                if reason:
                    rejected[str(client_proxy.cid)] = reason
                    print(f"[HE-ElGamal] Round {server_round}: client {client_proxy.cid} REJECTED — {reason}")
                else:
                    admitted.append((client_proxy, fit_res, arrays, proofs))

            quorum = admission_quorum(config)
            if len(admitted) < quorum:
                print(
                    f"[HE-ElGamal] Round {server_round}: {len(admitted)} verified client(s) < quorum {quorum} "
                    "— global model unchanged."
                )
                self.last_round_report = round_report(
                    server_round, "no_quorum", [cp.cid for cp, _, _, _ in admitted], rejected
                )
                return None, {"elgamal_admitted": len(admitted), "elgamal_rejected": len(rejected)}

            weights = [int(fit_res.num_examples) for _, fit_res, _, _ in admitted]
            with _timer(benchmark, "server_aggregate"):
                layers = [
                    np.frombuffer(eg.aggregate([arrays[1 + li].tobytes() for _, _, arrays, _ in admitted], weights), dtype=np.uint8)
                    for li in range(len(schema))
                ]
        except eg.ElGamalServiceError as exc:
            print(f"[HE-ElGamal] Round {server_round}: ABORTED — proof service failure, not a client fault: {exc}")
            self.last_round_report = round_report(server_round, "infrastructure_abort", [], rejected, str(exc))
            return None, {"elgamal_round_aborted": 1}

        self._last_anchor_data = anchor_data(server_round, [(str(cp.cid), proofs) for cp, _, _, proofs in admitted])
        self.last_round_report = round_report(server_round, "aggregated", [cp.cid for cp, _, _, _ in admitted], rejected)
        header = np.array([eg.HEADER_MAGIC, sum(weights), server_round], dtype=np.int64)
        print(f"[HE-ElGamal] Round {server_round}: aggregated {len(admitted)} verified client(s), rejected {len(rejected)}")
        return ndarrays_to_parameters([header] + layers), {
            "elgamal_admitted": len(admitted),
            "elgamal_rejected": len(rejected),
        }

    @staticmethod
    def _check_client(arrays, metrics, server_round, schema, chunks, total, policy, pk) -> Tuple[Optional[str], List[Dict]]:
        """Return (rejection reason or None, proofs). Raises ElGamalServiceError on infrastructure failure."""
        if len(arrays) != 1 + len(schema) or not _is_header(arrays[0]) or int(arrays[0][1]) != 1:
            return "upload does not match the server's model schema", []
        for li, (name, shape) in enumerate(schema):
            layer = arrays[1 + li]
            if layer.dtype != np.uint8 or layer.size != eg.numel(shape) * eg.CIPHERTEXT_BYTES:
                return f"layer '{name}' ciphertext has the wrong size", []

        try:
            proofs = json.loads(metrics.get("zkp_proofs_json", ""))
            proof_map = {(int(p["layer"]), int(p["chunk"])): str(p["proof_b64"]) for p in proofs}
            off_key = any(p.get("vk_sha256") != eg.pinned_vk() for p in proofs)
        except (ValueError, TypeError, KeyError, AttributeError):
            return "missing or malformed proofs", []
        if off_key:
            return "proofs were not made under the pinned verifying key", []
        expected = {(c.layer, c.index) for c in chunks}
        if len(proof_map) != len(proofs) or set(proof_map) != expected:
            return f"proof coverage mismatch: {len(proof_map)} distinct proofs for {len(expected)} required chunks", []

        for chunk in chunks:
            layer_bytes = arrays[1 + chunk.layer].tobytes()
            ct = layer_bytes[chunk.start * eg.CIPHERTEXT_BYTES : (chunk.start + chunk.size) * eg.CIPHERTEXT_BYTES]
            ok = eg.verify_chunk(
                pk,
                ct,
                eg.chunk_bound_sq(policy, chunk, total),
                eg.context_value(server_round, chunk),
                proof_map[(chunk.layer, chunk.index)],
            )
            if not ok:
                return f"proof for layer {chunk.layer} chunk {chunk.index} does not verify against the received ciphertext", []
        return None, proofs


def _is_header(arr) -> bool:
    arr = np.asarray(arr)
    return arr.dtype == np.int64 and arr.shape == (3,) and int(arr[0]) == eg.HEADER_MAGIC


def zkp_parallelism() -> int:
    from fl.core.zkp_gnark import DEFAULT_PARALLELISM

    return DEFAULT_PARALLELISM
