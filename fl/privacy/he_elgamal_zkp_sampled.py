"""
Verifiable ElGamal HE with commit–challenge coordinate sampling.

Per federated round (fl.privacy.commit_challenge):

  commit    — the client encrypts every coordinate (exponential ElGamal on
              BabyJubJub, as in he_elgamal_zkp), keeps the encryption
              randomness locally, and uploads the ciphertexts with no proofs.
  challenge — the server's fresh seed selects s of n coordinates. The client
              proves, for each chunk of sampled coordinates, that the
              committed ciphertexts encrypt in-range values whose squared
              norm is within that chunk's share of the bound. The proof is
              made from the stored values and randomness, so it only verifies
              against the ciphertexts that were committed. The server checks
              each chunk against the committed ciphertexts at those indices
              and aggregates the committed ciphertexts of clients that pass.

Guarantee: a coordinate that violates the range or the chunk bound is detected
if it is sampled, with probability 1 − C(n−m, s)/C(n, s) over m bad
coordinates, because the client committed before the seed existed.

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
from fl.privacy.commit_challenge import CommitChallengeMixin, parse_proof_list
from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode, zkp_parallelism
from fl.privacy.registry import register_mode
from fl.privacy.zkp import timer


@register_mode("he_elgamal_zkp_sampled")
class HeElGamalZKPSampledMode(CommitChallengeMixin, HeElGamalZKPMode):
    """Exponential ElGamal with proofs over server-sampled committed coordinates."""

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
        policy: eg.Policy = context["policy"]
        state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        schema = eg.schema_of(state)
        q = np.concatenate([eg.quantize(state[name].reshape(-1), policy.scale, name) for name, _ in schema])
        with timer(benchmark, "encryption"):
            ct, rand = eg.encrypt_values(context["pk"], q)
        context["commitment"] = {"round": context["round"], "q": q, "rand": rand, "n": int(q.size)}

        layers, offset = [], 0
        for _, shape in schema:
            size = eg.numel(shape) * eg.CIPHERTEXT_BYTES
            layers.append(np.frombuffer(ct[offset : offset + size], dtype=np.uint8))
            offset += size
        header = np.array([eg.HEADER_MAGIC, 1, context["round"]], dtype=np.int64)
        return [header] + layers

    def _client_respond(self, context, commitment, indices, benchmark):
        policy: eg.Policy = context["policy"]
        n, rand = commitment["n"], commitment["rand"]

        def _prove(item):
            j, chunk_idx = item
            chunk = eg.Chunk(layer=0, index=j, start=0, size=len(chunk_idx))
            chunk_rand = b"".join(rand[i * eg.SCALAR_BYTES : (i + 1) * eg.SCALAR_BYTES] for i in chunk_idx)
            _, proof = eg.prove_with(
                context["pk"],
                commitment["q"][chunk_idx],
                chunk_rand,
                eg.chunk_bound_sq(policy, chunk, n),
                eg.context_value(context["round"], chunk),
            )
            return {"chunk": j, "proof_b64": proof, "vk_sha256": eg.pinned_vk()}

        items = list(enumerate(self._chunks(indices, policy.chunk_size)))
        with timer(benchmark, "proof_generation"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, zkp_parallelism())) as pool:
                proofs = list(pool.map(_prove, items))
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
        return None, {"layers": arrays[1:], "flat": b"".join(a.tobytes() for a in arrays[1:])}

    def _verify_response(self, payload, fit_res, indices, server_round, server_context):
        policy: eg.Policy = server_context["policy"]
        proofs = parse_proof_list(fit_res.metrics or {})
        chunks = self._chunks(indices, policy.chunk_size)
        try:
            proof_map = {int(p["chunk"]): str(p["proof_b64"]) for p in proofs or []}
            off_key = any(p.get("vk_sha256") != eg.pinned_vk() for p in proofs or [])
        except (KeyError, TypeError, ValueError, AttributeError):
            return "missing or malformed proofs", []
        if off_key:
            return "proofs were not made under the pinned verifying key", []
        if not proofs or len(proof_map) != len(proofs) or set(proof_map) != set(range(len(chunks))):
            return f"proof coverage mismatch: {len(proof_map)} distinct proofs for {len(chunks)} sampled chunks", []

        n, flat = int(len(payload["flat"]) // eg.CIPHERTEXT_BYTES), payload["flat"]
        for j, chunk_idx in enumerate(chunks):
            chunk = eg.Chunk(layer=0, index=j, start=0, size=len(chunk_idx))
            ct = b"".join(flat[i * eg.CIPHERTEXT_BYTES : (i + 1) * eg.CIPHERTEXT_BYTES] for i in chunk_idx)
            ok = eg.verify_chunk(
                server_context["pk"],
                ct,
                eg.chunk_bound_sq(policy, chunk, n),
                eg.context_value(server_round, chunk),
                proof_map[j],
            )
            if not ok:
                return f"proof for sampled chunk {j} does not verify against the committed ciphertexts", []
        return None, proofs

    def _aggregate_commitments(self, admitted, server_context):
        from flwr.common import ndarrays_to_parameters

        weights = [weight for _, _, weight, _ in admitted]
        n_layers = len(admitted[0][1]["layers"])
        layers = [
            np.frombuffer(eg.aggregate([payload["layers"][li].tobytes() for _, payload, _, _ in admitted], weights), dtype=np.uint8)
            for li in range(n_layers)
        ]
        header = np.array([eg.HEADER_MAGIC, sum(weights), 0], dtype=np.int64)
        return ndarrays_to_parameters([header] + layers)
