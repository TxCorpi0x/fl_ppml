"""
Verifiable additive HE + ZKP mode (docs/ZKP.md, section 6.3).

Unlike the CKKS/TFHE composites in he_zkp.py, the proof here is bound to the
exact ciphertexts the server aggregates, and it bounds the update rather than
the weights:

  Client, per round
    g = decrypted global model (sums S, weight W), or the plaintext initial model
    Δ = w − g, clipped to ‖Δ‖ ≤ B (B sent by the server)
    q = round(S/W + scale·Δ)                            per coordinate, |q| < 2^17
    C = (r·G, (q + 2^17)·G + r·PK)                      exponential ElGamal, BabyJubJub
    π_chunk proves: every C encrypts an in-range q, the global slots decrypt to
                    S under the shared client key, and Σ(W·q − S)² ≤ bound_chunk
    public inputs: PK, bound_chunk, context(round, layer, chunk), the chunk's
                   ciphertexts, the global aggregate at the same slots and W

  Server, per round
    schema, chunking, context and the global model come from the server's own
    model and previous aggregate, never from client metadata. The declared
    chunk bounds must sum to at most W²·⌈B·scale + √n/2⌉². Each chunk proof is
    verified against the ciphertext bytes the server received. A client is
    aggregated only if every expected chunk verifies. Aggregation is
    Σ num_examples_k · C_k, which becomes the next round's global model.
    No admitted client, or an unreachable service, leaves the global model
    unchanged for that round (fail closed).

  Client, on download
    decrypt Σ n_k·(q_k + 2^17) with the shared secret key, subtract the offset,
    divide by Σ n_k · scale; keep the sums as the next update's base.

Guarantees: an aggregated update is exactly the range-checked vector its proofs
cover, and its distance from the global model is at most B + √n/(2·scale); a
proof cannot be moved to another ciphertext, position, round or global model.
The server learns neither the global model nor the update, but it does learn
each chunk's declared bound, i.e. the squared norm of the update restricted to
that chunk.

Does NOT guarantee: honest training or a useful update direction (a bounded
update can still be malicious); confidentiality against holders
of the shared client key; a trusted setup beyond a single party (docs/ZKP.md, section 7);
client-identity binding. Round-1 download is the server's plaintext initial model.
"""

from __future__ import annotations

import concurrent.futures
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from ppflx.core import elgamal_gnark as eg
from ppflx.core.update_bound import FIT_CONFIG_KEY, bound_from_fit_config, max_update_norm, split_bound
from ppflx.privacy.base import PrivacyMode
from ppflx.privacy.registry import register_mode


def _timer(benchmark, name):
    from contextlib import nullcontext

    from ppflx.core.benchmark import BenchmarkTimer

    return BenchmarkTimer(benchmark, name) if benchmark else nullcontext()


def _flat_state(net) -> Tuple[eg.Schema, np.ndarray]:
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return eg.schema_of(state), np.concatenate([v.astype(np.float64).reshape(-1) for v in state.values()])


def _layer_chunk(p) -> Tuple[int, int]:
    return int(p["layer"]), int(p["chunk"])


def parse_bounded_proofs(proofs, keys, key=_layer_chunk) -> Tuple[Optional[str], Dict]:
    """key → (proof_b64, declared bound) for a proof list, or a rejection reason."""
    try:
        entries = {key(p): p for p in proofs}
        off_key = any(p.get("vk_sha256") != eg.pinned_vk() for p in proofs)
    except (ValueError, TypeError, KeyError, AttributeError):
        return "missing or malformed proofs", {}
    if off_key:
        return "proofs were not made under the pinned verifying key", {}
    if len(entries) != len(proofs) or set(entries) != set(keys):
        return f"proof coverage mismatch: {len(entries)} distinct proofs for {len(keys)} required chunks", {}
    bounds = eg.parse_declared_bounds([entries[k] for k in keys], keys)
    if bounds is None:
        return "a proof declares a malformed or non-positive bound", {}
    return None, {k: (str(entries[k].get("proof_b64", "")), bounds[k]) for k in keys}


@register_mode("he_elgamal_zkp")
class HeElGamalZKPMode(PrivacyMode):
    """Exponential ElGamal on BabyJubJub with ciphertext-bound Groth16 update proofs."""

    @property
    def name(self) -> str:
        return "he_elgamal_zkp"

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_client_context(self, config) -> Dict:
        if config.sim_mode:
            raise RuntimeError("he_elgamal_zkp transports real ciphertexts; simulation mode is not supported")
        from ppflx.keys.he_elgamal import load_client

        keys = load_client(config.he_elgamal_secret_path)
        return {"sk": keys["sk"], "pk": keys["pk"], "policy": eg.Policy.from_env(), "round": None, "global": None}

    def setup_server_context(self, config) -> Dict:
        if config.sim_mode:
            raise RuntimeError("he_elgamal_zkp transports real ciphertexts; simulation mode is not supported")
        from ppflx.keys.he_elgamal import load_server

        self._max_update_norm = max_update_norm(config)
        return {
            "pk": load_server(config.he_elgamal_public_path),
            "policy": eg.Policy.from_env(),
            "schema": None,
            "global": None,
            "max_update_norm": self._max_update_norm,
        }

    def bind_server_model(self, server_context, model) -> None:
        server_context["schema"] = eg.schema_of(model.state_dict())
        arrays = [t.detach().cpu().numpy() for t in model.state_dict().values()]
        server_context["global"] = eg.GlobalModel.initial(arrays, server_context["policy"].scale)

    def fit_config(self, server_round: int) -> Dict:
        bound = getattr(self, "_max_update_norm", None)
        return {} if bound is None else {FIT_CONFIG_KEY: str(bound)}

    def on_fit_config(self, context, fit_config: Dict) -> None:
        context["round"] = int(fit_config["server_round"])
        context["max_update_norm"] = bound_from_fit_config(fit_config)

    def use_client_for_initial_params(self, config) -> bool:
        return False

    # ── Client ────────────────────────────────────────────────────────────

    def get_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        raise RuntimeError("he_elgamal_zkp never uploads parameters outside a proven fit()")

    @staticmethod
    def _require_client_state(context) -> Tuple[int, eg.Policy, float, eg.GlobalModel]:
        server_round = context.get("round")
        if server_round is None:
            raise RuntimeError("fit round unknown: on_fit_config was not called before send_parameters")
        glob = context.get("global")
        if glob is None:
            raise RuntimeError("no global model received; refusing to prove an update without its base")
        return server_round, context["policy"], context["max_update_norm"], glob

    def send_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        server_round, policy, bound, glob = self._require_client_state(context)
        schema, local = _flat_state(net)
        chunks = eg.chunks_for(schema, policy.chunk_size)
        total = local.size
        if glob.size != total:
            raise RuntimeError(f"global model has {glob.size} coordinates, local model {total}")

        with _timer(benchmark, "proof_generation"):
            q, norm, clipped = eg.quantize_update(glob, local, bound, policy.scale)
            sums = glob.centered_sums()
            indices = [eg.chunk_indices(schema, c) for c in chunks]
            bounds = split_bound(
                [eg.update_energy(q[i], sums[i], glob.weight) for i in indices],
                eg.total_bound_sq(bound, policy.scale, glob.weight, total),
            )

            def _prove(k: int):
                idx = indices[k]
                return eg.prove_chunk(
                    context["pk"],
                    context["sk"],
                    q[idx],
                    bounds[k],
                    eg.context_value(server_round, chunks[k]),
                    glob.request(idx, prover=True),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, zkp_parallelism())) as pool:
                outputs = list(pool.map(_prove, range(len(chunks))))
        context["update_norm"], context["update_clipped"] = norm, clipped

        layer_bytes = [bytearray(eg.numel(shape) * eg.CIPHERTEXT_BYTES) for _, shape in schema]
        proofs = []
        for k, (chunk, (ct, proof_b64)) in enumerate(zip(chunks, outputs)):
            offset = chunk.start * eg.CIPHERTEXT_BYTES
            layer_bytes[chunk.layer][offset : offset + len(ct)] = ct
            proofs.append({
                "layer": chunk.layer,
                "chunk": chunk.index,
                "bound_sq": str(bounds[k]),
                "proof_b64": proof_b64,
                "vk_sha256": eg.pinned_vk(),
            })

        self._proof_cache = (proofs, sum(len(p["proof_b64"]) for p in proofs))
        header = np.array([eg.HEADER_MAGIC, 1, server_round], dtype=np.int64)
        return [header] + [np.frombuffer(bytes(b), dtype=np.uint8) for b in layer_bytes]

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        from ppflx.privacy.zkp import update_metrics

        proofs, proof_bytes = getattr(self, "_proof_cache", None) or ([], 0)
        if not proofs:
            return {}
        return {
            "zkp_proofs_json": json.dumps(proofs),
            "gnark_num_proofs": len(proofs),
            "gnark_proof_bytes": proof_bytes,
            **update_metrics(context),
        }

    def receive_parameters(self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        from ppflx.core.params import set_parameters

        params = list(params)
        names = list(net.state_dict().keys())
        shapes = [tuple(t.shape) for t in net.state_dict().values()]
        policy: eg.Policy = context["policy"]
        if params and _is_header(params[0]):
            total_weight = int(params[0][1])
            layers = params[1:]
            if total_weight <= 0 or len(layers) != len(names):
                raise ValueError("malformed encrypted global model")
            plain, sums = [], []
            with _timer(benchmark, "decryption"):
                for name, shape, ct in zip(names, shapes, layers):
                    layer_sums = eg.decrypt(context["sk"], ct.tobytes(), total_weight)
                    if layer_sums.size != eg.numel(shape):
                        raise ValueError(f"layer '{name}': decrypted {layer_sums.size} values for shape {shape}")
                    sums.append(layer_sums)
                    plain.append((layer_sums / (total_weight * policy.scale)).astype(np.float32).reshape(shape))
            set_parameters(net, plain, None, None)
            context["global"] = eg.GlobalModel.aggregate(total_weight, layers, sums=sums)
            return
        # Round-1 download: the server's plaintext initial model.
        if len(params) != len(names) or any(np.asarray(p).shape != s for p, s in zip(params, shapes)):
            raise ValueError("unexpected parameters: neither an encrypted aggregate nor the initial model")
        arrays = [np.asarray(p, dtype=np.float32) for p in params]
        set_parameters(net, arrays, None, None)
        context["global"] = eg.GlobalModel.initial(arrays, policy.scale)

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
        self._last_anchor_data = None

        from ppflx.privacy.zkp import admission_quorum, anchor_data, round_report

        admitted, rejected = [], {}
        try:
            for client_proxy, fit_res in results:
                arrays = parameters_to_ndarrays(fit_res.parameters)
                with _timer(benchmark, "proof_verification"):
                    reason, proofs = self._check_client(arrays, fit_res.metrics or {}, server_round, schema, chunks, server_context)
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
            check_aggregate_weight(weights)
            with _timer(benchmark, "server_aggregate"):
                layers = [
                    np.frombuffer(eg.aggregate([arrays[1 + li].tobytes() for _, _, arrays, _ in admitted], weights), dtype=np.uint8)
                    for li in range(len(schema))
                ]
        except eg.ElGamalServiceError as exc:
            print(f"[HE-ElGamal] Round {server_round}: ABORTED — proof service failure, not a client fault: {exc}")
            self.last_round_report = round_report(server_round, "infrastructure_abort", [], rejected, str(exc))
            return None, {"elgamal_round_aborted": 1}

        server_context["global"] = eg.GlobalModel.aggregate(sum(weights), layers)
        self._last_anchor_data = anchor_data(server_round, [(str(cp.cid), proofs) for cp, _, _, proofs in admitted])
        self.last_round_report = round_report(server_round, "aggregated", [cp.cid for cp, _, _, _ in admitted], rejected)
        header = np.array([eg.HEADER_MAGIC, sum(weights), server_round], dtype=np.int64)
        print(f"[HE-ElGamal] Round {server_round}: aggregated {len(admitted)} verified client(s), rejected {len(rejected)}")
        return ndarrays_to_parameters([header] + layers), {
            "elgamal_admitted": len(admitted),
            "elgamal_rejected": len(rejected),
        }

    @staticmethod
    def _check_client(arrays, metrics, server_round, schema, chunks, server_context) -> Tuple[Optional[str], List[Dict]]:
        """Return (rejection reason or None, proofs). Raises ElGamalServiceError on infrastructure failure."""
        policy: eg.Policy = server_context["policy"]
        glob: eg.GlobalModel = server_context["global"]
        if len(arrays) != 1 + len(schema) or not _is_header(arrays[0]) or int(arrays[0][1]) != 1:
            return "upload does not match the server's model schema", []
        for li, (name, shape) in enumerate(schema):
            layer = arrays[1 + li]
            if layer.dtype != np.uint8 or layer.size != eg.numel(shape) * eg.CIPHERTEXT_BYTES:
                return f"layer '{name}' ciphertext has the wrong size", []

        try:
            proofs = json.loads(metrics.get("zkp_proofs_json", ""))
        except (TypeError, ValueError):
            return "missing or malformed proofs", []
        if not isinstance(proofs, list) or not proofs:
            return "missing or malformed proofs", []
        keys = [(c.layer, c.index) for c in chunks]
        reason, entries = parse_bounded_proofs(proofs, keys)
        if reason:
            return reason, []
        total = sum(eg.numel(shape) for _, shape in schema)
        limit = eg.total_bound_sq(server_context["max_update_norm"], policy.scale, glob.weight, total)
        declared = sum(bound for _, bound in entries.values())
        if declared > limit:
            return f"declared update bounds sum to {declared}, above the server's bound {limit}", []

        for chunk in chunks:
            proof_b64, bound = entries[(chunk.layer, chunk.index)]
            layer_bytes = arrays[1 + chunk.layer].tobytes()
            ct = layer_bytes[chunk.start * eg.CIPHERTEXT_BYTES : (chunk.start + chunk.size) * eg.CIPHERTEXT_BYTES]
            ok = eg.verify_chunk(
                server_context["pk"],
                ct,
                bound,
                eg.context_value(server_round, chunk),
                proof_b64,
                glob.request(eg.chunk_indices(schema, chunk), prover=False),
            )
            if not ok:
                return f"proof for layer {chunk.layer} chunk {chunk.index} does not verify against the received ciphertext", []
        return None, proofs


def check_aggregate_weight(weights) -> None:
    """The aggregate is the next round's global model; its weight must fit the circuit."""
    if sum(weights) >= eg.WEIGHT_LIMIT:
        # Not a client fault: abort the round rather than aggregate something no one can prove against.
        raise eg.ElGamalServiceError(
            f"total weight {sum(weights)} ≥ {eg.WEIGHT_LIMIT}: clients' num_examples too large for the ElGamal circuit"
        )


def _is_header(arr) -> bool:
    arr = np.asarray(arr)
    return arr.dtype == np.int64 and arr.shape == (3,) and int(arr[0]) == eg.HEADER_MAGIC


def zkp_parallelism() -> int:
    from ppflx.core.zkp_gnark import DEFAULT_PARALLELISM

    return DEFAULT_PARALLELISM
