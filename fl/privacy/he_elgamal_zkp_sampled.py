"""
Verifiable ElGamal HE with commit–challenge coordinate sampling.

Per federated round (fl.privacy.commit_challenge):

  commit    — the client clips its update against the global model it
              downloaded (as in he_elgamal_zkp), encrypts every coordinate,
              keeps the values, randomness and that global model locally, and
              uploads the ciphertexts with no proofs.
  challenge — the server's fresh seed selects s of n coordinates. The client
              proves, for each chunk of sampled coordinates, that the committed
              ciphertexts encrypt in-range values whose update against the
              global model (the aggregate at the same indices) is within the
              chunk's declared bound. The declared bounds must sum to at most
              W²·⌈B·scale + √s/2⌉². Proofs are made from the stored values and
              randomness, so they only verify against the committed
              ciphertexts. The server checks each chunk against those and
              against the global model it held at commit time, and aggregates
              the committed ciphertexts of clients that pass.

Guarantee: the sampled coordinates' update is bounded by B (plus rounding
slack); a coordinate that makes that impossible is detected if it is sampled,
with probability 1 − C(n−m, s)/C(n, s) over m such coordinates, because the
client committed before the seed existed.

Does NOT guarantee anything about unsampled coordinates. An out-of-range
unsampled ciphertext makes aggregate decryption fail on every client (a denial
of service, detected only after aggregation). All limitations of
he_elgamal_zkp also apply.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np

from fl.core import elgamal_gnark as eg
from fl.core.sampling import sample_rate_from_env
from fl.core.update_bound import split_bound
from fl.privacy.commit_challenge import CommitChallengeMixin, parse_proof_list
from fl.privacy.he_elgamal_zkp import (
    HeElGamalZKPMode,
    _flat_state,
    check_aggregate_weight,
    parse_bounded_proofs,
    zkp_parallelism,
)
from fl.privacy.registry import register_mode
from fl.privacy.zkp import timer


@register_mode("he_elgamal_zkp_sampled")
class HeElGamalZKPSampledMode(CommitChallengeMixin, HeElGamalZKPMode):
    """Exponential ElGamal with update proofs over server-sampled committed coordinates."""

    @property
    def name(self) -> str:
        return "he_elgamal_zkp_sampled"

    def setup_server_context(self, config):
        context = super().setup_server_context(config)
        self._sample_rate = sample_rate_from_env()
        return context

    def _schema(self, server_context):
        return server_context.get("schema")

    @staticmethod
    def _chunks(indices, chunk_size):
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # ── Client hooks ───────────────────────────────────────────────────────

    def _client_commit(self, net, context, benchmark):
        server_round, policy, bound, glob = self._require_client_state(context)
        schema, local = _flat_state(net)
        if glob.size != local.size:
            raise RuntimeError(f"global model has {glob.size} coordinates, local model {local.size}")
        q, norm, clipped = eg.quantize_update(glob, local, bound, policy.scale)
        with timer(benchmark, "encryption"):
            ct, rand = eg.encrypt_values(context["pk"], q)
        context["commitment"] = {
            "round": server_round, "q": q, "rand": rand, "n": int(q.size), "global": glob, "max_update_norm": bound,
        }
        context["update_norm"], context["update_clipped"] = norm, clipped

        layers, offset = [], 0
        for _, shape in schema:
            size = eg.numel(shape) * eg.CIPHERTEXT_BYTES
            layers.append(np.frombuffer(ct[offset : offset + size], dtype=np.uint8))
            offset += size
        header = np.array([eg.HEADER_MAGIC, 1, server_round], dtype=np.int64)
        return [header] + layers

    def _client_respond(self, context, commitment, indices, benchmark):
        policy: eg.Policy = context["policy"]
        glob: eg.GlobalModel = commitment["global"]
        rand, q, sums = commitment["rand"], commitment["q"], glob.centered_sums()
        chunks = self._chunks(indices, policy.chunk_size)
        bounds = split_bound(
            [eg.update_energy(q[idx], sums[idx], glob.weight) for idx in chunks],
            eg.total_bound_sq(commitment["max_update_norm"], policy.scale, glob.weight, len(indices)),
        )

        def _prove(j):
            chunk_idx = chunks[j]
            chunk = eg.Chunk(layer=0, index=j, start=0, size=len(chunk_idx))
            chunk_rand = b"".join(rand[i * eg.SCALAR_BYTES : (i + 1) * eg.SCALAR_BYTES] for i in chunk_idx)
            _, proof = eg.prove_with(
                context["pk"],
                context["sk"],
                q[chunk_idx],
                chunk_rand,
                bounds[j],
                eg.context_value(context["round"], chunk),
                glob.request(chunk_idx, prover=True),
            )
            return {"chunk": j, "bound_sq": str(bounds[j]), "proof_b64": proof, "vk_sha256": eg.pinned_vk()}

        with timer(benchmark, "proof_generation"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, zkp_parallelism())) as pool:
                proofs = list(pool.map(_prove, range(len(chunks))))
        return proofs, sum(len(p["proof_b64"]) for p in proofs)

    # ── Server hooks ───────────────────────────────────────────────────────

    def _accept_commit(self, fit_res, schema, server_context):
        from flwr.common import parameters_to_ndarrays

        from fl.privacy.he_elgamal_zkp import _is_header

        arrays = parameters_to_ndarrays(fit_res.parameters)
        if len(arrays) != 1 + len(schema) or not _is_header(arrays[0]) or int(arrays[0][1]) != 1:
            return "commitment does not match the server's model schema", None
        for li, (name, shape) in enumerate(schema):
            layer = arrays[1 + li]
            if layer.dtype != np.uint8 or layer.size != eg.numel(shape) * eg.CIPHERTEXT_BYTES:
                return f"layer '{name}' ciphertext has the wrong size", None
        # The update is measured against the global model the client downloaded this round.
        return None, {
            "layers": arrays[1:],
            "flat": b"".join(a.tobytes() for a in arrays[1:]),
            "global": server_context["global"],
        }

    def _verify_response(self, payload, fit_res, indices, server_round, server_context):
        policy: eg.Policy = server_context["policy"]
        glob: eg.GlobalModel = payload["global"]
        proofs = parse_proof_list(fit_res.metrics or {})
        if not proofs:
            return "missing or malformed proofs", []
        chunks = self._chunks(indices, policy.chunk_size)
        reason, entries = parse_bounded_proofs(proofs, list(range(len(chunks))), key=lambda p: int(p["chunk"]))
        if reason:
            return reason, []
        limit = eg.total_bound_sq(server_context["max_update_norm"], policy.scale, glob.weight, len(indices))
        declared = sum(bound for _, bound in entries.values())
        if declared > limit:
            return f"declared update bounds sum to {declared}, above the server's bound {limit}", []

        flat = payload["flat"]
        for j, chunk_idx in enumerate(chunks):
            proof_b64, bound = entries[j]
            chunk = eg.Chunk(layer=0, index=j, start=0, size=len(chunk_idx))
            ct = b"".join(flat[i * eg.CIPHERTEXT_BYTES : (i + 1) * eg.CIPHERTEXT_BYTES] for i in chunk_idx)
            ok = eg.verify_chunk(
                server_context["pk"],
                ct,
                bound,
                eg.context_value(server_round, chunk),
                proof_b64,
                glob.request(chunk_idx, prover=False),
            )
            if not ok:
                return f"proof for sampled chunk {j} does not verify against the committed ciphertexts", []
        return None, proofs

    def _aggregate_commitments(self, admitted, server_context):
        from flwr.common import ndarrays_to_parameters

        weights = [weight for _, _, weight, _ in admitted]
        check_aggregate_weight(weights)
        n_layers = len(admitted[0][1]["layers"])
        layers = [
            np.frombuffer(eg.aggregate([payload["layers"][li].tobytes() for _, payload, _, _ in admitted], weights), dtype=np.uint8)
            for li in range(n_layers)
        ]
        server_context["global"] = eg.GlobalModel.aggregate(sum(weights), layers)
        header = np.array([eg.HEADER_MAGIC, sum(weights), 0], dtype=np.int64)
        return ndarrays_to_parameters([header] + layers)
