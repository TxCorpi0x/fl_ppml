"""
Plaintext ZKP with commit–challenge coordinate sampling.

NO SECURITY BENEFIT. The server receives every plaintext weight in the commit
round, so it could check the exact norm of the whole update itself, for free
and without any proof. This mode exists only so the benchmark can compare
sampled against full proving cost with the same selection mechanism as
``he_elgamal_zkp_sampled``, where the server can't see the values and
sampling is meaningful (docs/ZKP.md, section 6.4).

Per federated round (ppflx.privacy.commit_challenge): the client clips its update
against the global model and commits the clipped plaintext weights; after the
server's seed arrives, it proves the norm bound and MiMC hash over the sampled
coordinates of its quantized update Δq; the server recomputes those
coordinates from the committed weights and its own global model, verifies, and
FedAvgs the admitted commitments.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ppflx.core.sampling import sample_rate_from_env
from ppflx.privacy.base import _plain_params
from ppflx.privacy.commit_challenge import CommitChallengeMixin, parse_proof_list
from ppflx.privacy.registry import register_mode
from ppflx.privacy.zkp import (
    ZKPMode,
    cast_like,
    clip_update_in_place,
    quantized_update,
    resolve_backend,
    timer,
)

SAMPLE_NAME = "sample"


@register_mode("zkp_sampled")
class ZKPSampledMode(CommitChallengeMixin, ZKPMode):
    """Plaintext weights, update proofs over server-sampled coordinates (benchmark only)."""

    @property
    def name(self) -> str:
        return "zkp_sampled"

    def setup_client_context(self, config):
        if config.sim_mode:
            raise RuntimeError("zkp_sampled needs two Flower rounds per round; simulation mode is not supported")
        if resolve_backend(config) != "gnark":
            raise RuntimeError("zkp_sampled requires the gnark backend")
        return super().setup_client_context(config)

    def setup_server_context(self, config):
        if config.sim_mode:
            raise RuntimeError("zkp_sampled needs two Flower rounds per round; simulation mode is not supported")
        if resolve_backend(config) != "gnark":
            raise RuntimeError("zkp_sampled requires the gnark backend")
        ZKPMode.setup_server_context(self, config)
        self._sample_rate = sample_rate_from_env()
        return None

    def get_parameters(self, net, context, *, sim_mode, benchmark=None) -> List[np.ndarray]:
        return _plain_params(net)

    # ── Client hooks ───────────────────────────────────────────────────────

    def _client_commit(self, net, context, benchmark):
        delta, _ = clip_update_in_place(net, context)
        values = np.concatenate([d.reshape(-1) for d in delta.values()])
        context["commitment"] = {
            "round": context["round"],
            "values": values,
            "n": int(values.size),
            "max_update_norm": context["max_update_norm"],
        }
        return _plain_params(net)

    def _client_respond(self, context, commitment, indices, benchmark):
        from ppflx.core.zkp_gnark import generate_gnark_proofs, policy_bound_sq

        total = policy_bound_sq(commitment["max_update_norm"], len(indices))
        with timer(benchmark, "proof_generation"):
            proofs, proof_bytes = generate_gnark_proofs({SAMPLE_NAME: commitment["values"][indices]}, total_bound_sq=total)
        if not proofs:
            raise RuntimeError("[ZKP-SAMPLED] no proofs generated for the challenge")
        return proofs, proof_bytes

    # ── Server hooks ───────────────────────────────────────────────────────

    def _accept_commit(self, fit_res, schema, server_context):
        from flwr.common import parameters_to_ndarrays

        params = parameters_to_ndarrays(fit_res.parameters)
        if len(params) != len(schema) or any(tuple(p.shape) != shape for p, (_, shape) in zip(params, schema)):
            return "commitment does not match the server's model schema", None
        # The update against the global model the client downloaded this round.
        delta = quantized_update(params, self._global)
        return None, {"params": params, "flat": np.concatenate([d.reshape(-1) for d in delta])}

    def _verify_response(self, payload, fit_res, indices, server_round, server_context):
        from ppflx.core import zkp_gnark

        proofs = parse_proof_list(fit_res.metrics or {})
        if not proofs:
            return "missing or malformed proofs", []
        reason = zkp_gnark.check_proof_policy(
            proofs,
            [(SAMPLE_NAME, (len(indices),))],
            require_hash=False,
            total_bound_sq=self._total_bound_sq(None, n=len(indices)),
        )
        if reason:
            return reason, []
        ok, failed = zkp_gnark.verify_gnark_proofs([payload["flat"][indices]], [SAMPLE_NAME], proofs)
        if not ok:
            return f"sampled-coordinate proof failed for {failed[:5]}", []
        return None, proofs

    def _aggregate_commitments(self, admitted, server_context):
        from flwr.common import ndarrays_to_parameters

        from ppflx.core.security import aggregate_custom

        aggregated = aggregate_custom([(payload["params"], weight) for _, payload, weight, _ in admitted])
        self._global = cast_like(aggregated, self._global)
        return ndarrays_to_parameters(aggregated)
