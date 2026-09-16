"""
Commit–challenge coordinate sampling (docs/ZKP.md, section 6.4).

Each federated round takes two Flower rounds:

  commit    (odd Flower round)  — clients download the global model, train,
                                  and upload their full update with no
                                  proofs. The server stores each commitment
                                  and does not aggregate.
  challenge (even Flower round) — only now does the server draw a fresh seed.
                                  Clients don't train: they prove the
                                  coordinates the seed selects, over the
                                  update they committed. The server verifies
                                  against the stored commitment and aggregates
                                  the committed updates of clients that pass.

A client can't pick which coordinates go unproven, because the seed doesn't
exist until its update is fixed. Against a client that corrupts m of n
coordinates, detection probability is 1 − C(n−m, s)/C(n, s) for s sampled
coordinates (ppflx.core.sampling.detection_probability). Coordinates that are not
sampled carry no proof.

Modes mix this in before their base mode class and implement the hooks at the
bottom of the class.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ppflx.core.sampling import new_round_seed, sample_indices, sample_size
from ppflx.privacy.zkp import admission_quorum, anchor_data, round_report, timer


def _service_errors() -> Tuple[type, ...]:
    from ppflx.core.elgamal_gnark import ElGamalServiceError
    from ppflx.core.zkp_gnark import GnarkServiceError

    return (GnarkServiceError, ElGamalServiceError)


class CommitChallengeMixin:
    rounds_per_fl_round = 2

    @staticmethod
    def phase(server_round: int) -> str:
        return "commit" if server_round % 2 == 1 else "challenge"

    # ── Server → client configuration ─────────────────────────────────────

    def fit_config(self, server_round: int) -> Dict:
        """Phase for this Flower round; the challenge seed is created here, after commits arrived."""
        config: Dict[str, Any] = {**super().fit_config(server_round), "zkp_phase": self.phase(server_round)}
        if config["zkp_phase"] == "challenge":
            seeds = self.__dict__.setdefault("_round_seeds", {})
            seeds.setdefault(server_round, new_round_seed())
            config["zkp_sample_seed"] = seeds[server_round]
            config["zkp_sample_rate"] = str(self._server_sample_rate())
        return config

    def evaluates_this_round(self, server_round: int) -> bool:
        # The global model only changes after a challenge round.
        return self.phase(server_round) == "challenge"

    def _server_sample_rate(self) -> float:
        rate = getattr(self, "_sample_rate", None)
        if rate is None:
            raise RuntimeError("sample rate not set: setup_server_context must run first")
        return rate

    # ── Client ────────────────────────────────────────────────────────────

    def on_fit_config(self, context, fit_config: Dict) -> None:
        if "zkp_phase" not in fit_config:
            raise RuntimeError("server did not send a commit/challenge phase; it is not running this protocol")
        super().on_fit_config(context, fit_config)  # the base mode's policy, e.g. the update-norm bound
        context["round"] = int(fit_config["server_round"])
        context["phase"] = fit_config["zkp_phase"]
        context["sample_seed"] = fit_config.get("zkp_sample_seed")
        context["sample_rate"] = float(fit_config["zkp_sample_rate"]) if "zkp_sample_rate" in fit_config else None

    def trains_this_round(self, context) -> bool:
        return context.get("phase") == "commit"

    def send_parameters(self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None):
        phase = context.get("phase")
        if phase == "commit":
            self._proof_cache = None
            return self._client_commit(net, context, benchmark)
        if phase == "challenge":
            commitment = context.get("commitment")
            if not commitment or commitment["round"] != context["round"] - 1:
                # E.g. the client connected after the commit round. Answer with
                # no proofs; the server rejects it as having no commitment.
                print(f"[{self.name.upper()}] round {context['round']}: challenge without a commitment; sending no proofs")
                context["commitment"] = None
                self._proof_cache = None
                return []
            n = commitment["n"]
            indices = sample_indices(context["sample_seed"], n, sample_size(n, context["sample_rate"]))
            proofs, proof_bytes = self._client_respond(context, commitment, indices, benchmark)
            context["commitment"] = None
            self._proof_cache = (proofs, proof_bytes)
            return []
        raise RuntimeError(f"unknown protocol phase: {phase!r}")

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        from ppflx.privacy.zkp import update_metrics

        cache = getattr(self, "_proof_cache", None)
        if not cache:
            return update_metrics(context)  # commit round: the update was clipped here
        proofs, proof_bytes = cache
        return {
            "zkp_proofs_json": json.dumps(proofs),
            "gnark_num_proofs": len(proofs),
            "gnark_proof_bytes": proof_bytes,
            **update_metrics(context),
        }

    # ── Server ────────────────────────────────────────────────────────────

    def aggregate_fit_override(self, server_round, results, failures, server_context, config, benchmark=None) -> Optional[Tuple]:
        self._last_anchor_data = None
        schema = self._schema(server_context)
        if schema is None:
            raise RuntimeError("server schema not bound: make_strategy must call bind_server_model")
        if self.phase(server_round) == "commit":
            return self._commit_round(server_round, results, schema, server_context)
        return self._challenge_round(server_round, results, schema, server_context, config, benchmark)

    def _commit_round(self, server_round, results, schema, server_context) -> Tuple:
        self._commitments: Dict[str, Tuple[Any, int]] = {}
        rejected = {}
        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            reason, payload = self._accept_commit(fit_res, schema, server_context)
            if reason:
                rejected[cid] = reason
                print(f"[{self.name.upper()}] Round {server_round}: commitment from {cid} REJECTED — {reason}")
            else:
                self._commitments[cid] = (payload, int(fit_res.num_examples))
        self.last_round_report = round_report(server_round, "committed", list(self._commitments), rejected)
        return None, {"round_outcome": "committed", "committed": len(self._commitments), "rejected": len(rejected)}

    def _challenge_round(self, server_round, results, schema, server_context, config, benchmark) -> Tuple:
        commitments = getattr(self, "_commitments", {}) or {}
        self._commitments = {}
        seed = self.__dict__.get("_round_seeds", {}).get(server_round)
        if seed is None:
            raise RuntimeError(f"no challenge seed was issued for round {server_round}")
        n = sum(_numel(shape) for _, shape in schema)
        s = sample_size(n, self._server_sample_rate())
        indices = sample_indices(seed, n, s)

        admitted, rejected, responded = [], {}, set()
        try:
            for client_proxy, fit_res in results:
                cid = str(client_proxy.cid)
                responded.add(cid)
                if cid not in commitments:
                    rejected[cid] = "challenge response without a commitment"
                    continue
                payload, weight = commitments[cid]
                with timer(benchmark, "proof_verification"):
                    reason, proofs = self._verify_response(payload, fit_res, indices, server_round, server_context)
                if reason:
                    rejected[cid] = reason
                    print(f"[{self.name.upper()}] Round {server_round}: client {cid} REJECTED — {reason}")
                else:
                    admitted.append((cid, payload, weight, proofs))
            for cid in commitments:
                if cid not in responded:
                    rejected[cid] = "committed but did not answer the challenge"

            quorum = admission_quorum(config)
            if len(admitted) < quorum:
                print(f"[{self.name.upper()}] Round {server_round}: {len(admitted)} admitted < quorum {quorum} — global model unchanged.")
                self.last_round_report = self._with_sampling(
                    round_report(server_round, "no_quorum", [a[0] for a in admitted], rejected), seed, s
                )
                return None, {"round_outcome": "no_quorum", "admitted": len(admitted), "rejected": len(rejected)}

            with timer(benchmark, "server_aggregate"):
                params = self._aggregate_commitments(admitted, server_context)
        except _service_errors() as exc:
            print(f"[{self.name.upper()}] Round {server_round}: ABORTED — proof service failure, not a client fault: {exc}")
            self.last_round_report = self._with_sampling(
                round_report(server_round, "infrastructure_abort", [], rejected, str(exc)), seed, s
            )
            return None, {"round_outcome": "infrastructure_abort"}

        self._last_anchor_data = anchor_data(server_round, [(cid, proofs) for cid, _, _, proofs in admitted])
        self.last_round_report = self._with_sampling(
            round_report(server_round, "aggregated", [a[0] for a in admitted], rejected), seed, s
        )
        print(f"[{self.name.upper()}] Round {server_round}: aggregated {len(admitted)} client(s) after proving {s}/{n} sampled coordinates")
        return params, {"round_outcome": "aggregated", "admitted": len(admitted), "rejected": len(rejected)}

    @staticmethod
    def _with_sampling(report: Dict, seed: str, s: int) -> Dict:
        # The seed lets anyone reproduce which coordinates were proven.
        return {**report, "sample_seed": seed, "sampled_coordinates": s}

    # ── Hooks for concrete modes ──────────────────────────────────────────

    def _schema(self, server_context):
        return getattr(self, "_server_schema", None)

    def _client_commit(self, net, context, benchmark) -> List:
        raise NotImplementedError

    def _client_respond(self, context, commitment: Dict, indices, benchmark) -> Tuple[List[Dict], int]:
        raise NotImplementedError

    def _accept_commit(self, fit_res, schema, server_context) -> Tuple[Optional[str], Any]:
        raise NotImplementedError

    def _verify_response(self, payload, fit_res, indices, server_round, server_context) -> Tuple[Optional[str], List[Dict]]:
        raise NotImplementedError

    def _aggregate_commitments(self, admitted: List[Tuple[str, Any, int, list]], server_context):
        raise NotImplementedError


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def parse_proof_list(metrics: Dict) -> Optional[list]:
    try:
        proofs = json.loads(metrics.get("zkp_proofs_json", ""))
    except (TypeError, ValueError):
        return None
    return proofs if isinstance(proofs, list) and proofs else None
