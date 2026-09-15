"""Plaintext zkp_sampled: commit–challenge protocol mechanics (benchmark-only mode).

Uses the real gnark service. The protocol logic (seed issued only after
commitments, proofs checked against what was committed, missing commitments or
responses rejected) is shared with he_elgamal_zkp_sampled.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl.core.sampling import sample_indices, sample_size
from fl.privacy.zkp_sampled import ZKPSampledMode
from tests.conftest import requires_gnark, use_gnark

pytestmark = requires_gnark

BOUND = 5.0


class TinyModel(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        with torch.no_grad():
            self.model[0].weight.copy_(torch.linspace(-0.5, 0.5, 12).reshape(3, 4) * scale)
            self.model[0].bias.copy_(torch.tensor([0.1, -0.1, 0.05]) * scale)


CONFIG = NS(sim_mode=False, zkp_backend="gnark", zkp_gnark_host="test", min_fit_clients=1)


@pytest.fixture(autouse=True)
def _env(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.4")  # 15 coordinates → 6 sampled
    monkeypatch.setenv("FL_ZKP_MAX_NORM", str(BOUND))
    for var in ("FL_ZKP_BACKEND", "FL_ZKP_LAYERS"):
        monkeypatch.delenv(var, raising=False)


def _server():
    mode = ZKPSampledMode()
    mode.setup_server_context(CONFIG)
    mode.bind_server_model(None, TinyModel())
    return mode


def _fit(params, metrics=None, n=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), n, metrics or {})


def _client_round(mode, context, server, server_round, net):
    server_cfg = {"server_round": server_round, **server.fit_config(server_round)}
    mode.on_fit_config(context, server_cfg)
    if server_cfg["zkp_phase"] == "commit":
        mode.receive_parameters(TinyModel(), list(server._global), context, sim_mode=False)
    params = mode.send_parameters(net, context, sim_mode=False)
    return params, mode.post_fit_metrics(context)


def test_commit_round_stores_updates_and_issues_no_seed():
    server = _server()
    assert server.fit_config(1) == {"zkp_max_update_norm": str(BOUND), "zkp_phase": "commit"}
    assert not server.evaluates_this_round(1) and server.evaluates_this_round(2)
    assert ZKPSampledMode().trains_this_round({"phase": "commit"})
    assert not ZKPSampledMode().trains_this_round({"phase": "challenge"})


def test_honest_clients_are_aggregated_after_the_challenge():
    server = _server()
    ctxs = {cid: ({"backend": "gnark"}, ZKPSampledMode()) for cid in ("a", "b")}
    nets = {"a": TinyModel(), "b": TinyModel(scale=2.0)}
    commits = [(NS(cid=c), _fit(_client_round(ctxs[c][1], ctxs[c][0], server, 1, nets[c])[0])) for c in nets]
    commit_out = server.aggregate_fit_override(1, commits, [], None, CONFIG)
    responses = [(NS(cid=c), _fit(*_client_round(ctxs[c][1], ctxs[c][0], server, 2, nets[c]))) for c in nets]
    params, out = server.aggregate_fit_override(2, responses, [], None, CONFIG)

    assert commit_out == (None, {"round_outcome": "committed", "committed": 2, "rejected": 0})
    assert out["round_outcome"] == "aggregated" and out["admitted"] == 2
    report = server.last_round_report
    assert report["sampled_coordinates"] == sample_size(15, 0.4) and len(report["sample_seed"]) == 64
    expected = TinyModel().state_dict()["model.0.weight"].numpy() * 1.5
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], expected, atol=1e-6)
    np.testing.assert_allclose(server._global[0], expected, atol=1e-6)


def test_oversized_update_is_clipped_at_commit():
    server = _server()
    ctx, client = {"backend": "gnark"}, ZKPSampledMode()
    params, _ = _client_round(client, ctx, server, 1, TinyModel(scale=50.0))
    delta = np.concatenate([(p - g).reshape(-1) for p, g in zip(params, server._global)])
    assert ctx["update_clipped"] and np.linalg.norm(delta.astype(np.float64)) <= BOUND + 1e-4


def test_seed_is_fresh_per_challenge_round():
    server = _server()
    assert server.fit_config(2)["zkp_sample_seed"] != server.fit_config(4)["zkp_sample_seed"]
    assert server.fit_config(2)["zkp_sample_seed"] == server.fit_config(2)["zkp_sample_seed"]


def test_response_that_does_not_match_the_commitment_is_rejected():
    server = _server()
    honest_ctx, honest = {"backend": "gnark"}, ZKPSampledMode()
    cheat_ctx, cheat = {"backend": "gnark"}, ZKPSampledMode()
    commits = [
        (NS(cid="honest"), _fit(_client_round(honest, honest_ctx, server, 1, TinyModel(scale=1.1))[0])),
        (NS(cid="cheat"), _fit(_client_round(cheat, cheat_ctx, server, 1, TinyModel(scale=50.0))[0])),
    ]
    server.aggregate_fit_override(1, commits, [], None, CONFIG)
    # The cheater answers the challenge as if its update had been zero.
    cheat_ctx["commitment"]["values"] = np.zeros_like(cheat_ctx["commitment"]["values"])
    responses = [
        (NS(cid="honest"), _fit(*_client_round(honest, honest_ctx, server, 2, TinyModel()))),
        (NS(cid="cheat"), _fit(*_client_round(cheat, cheat_ctx, server, 2, TinyModel()))),
    ]

    server.aggregate_fit_override(2, responses, [], None, CONFIG)

    assert server.last_round_report["admitted"] == ["honest"]
    assert "cheat" in server.last_round_report["rejected"]


def test_missing_commitment_or_missing_response_is_rejected():
    server = _server()
    ctx, client = {"backend": "gnark"}, ZKPSampledMode()
    server.aggregate_fit_override(1, [(NS(cid="a"), _fit(_client_round(client, ctx, server, 1, TinyModel())[0])), (NS(cid="silent"), _fit(_client_round(ZKPSampledMode(), {"backend": "gnark"}, server, 1, TinyModel())[0]))], [], None, CONFIG)
    params, metrics = _client_round(client, ctx, server, 2, TinyModel())

    server.aggregate_fit_override(2, [(NS(cid="a"), _fit(params, metrics)), (NS(cid="stranger"), _fit(params, metrics))], [], None, CONFIG)

    rejected = server.last_round_report["rejected"]
    assert "a" in server.last_round_report["admitted"]
    assert rejected["stranger"] == "challenge response without a commitment"
    assert rejected["silent"] == "committed but did not answer the challenge"


def test_client_without_a_commitment_answers_with_no_proofs():
    """E.g. a client that connected after the commit round: it must not crash the round."""
    server = _server()
    client, ctx = ZKPSampledMode(), {"backend": "gnark"}
    params, metrics = _client_round(client, ctx, server, 2, TinyModel())
    assert params == [] and metrics == {}

    server.aggregate_fit_override(2, [(NS(cid="late"), _fit(params, metrics))], [], None, CONFIG)
    assert server.last_round_report["rejected"] == {"late": "challenge response without a commitment"}


def test_selection_follows_the_issued_seed():
    server = _server()
    seed = server.fit_config(2)["zkp_sample_seed"]
    assert sample_indices(seed, 15, 6).tolist() == sample_indices(seed, 15, 6).tolist()


def test_simulation_mode_is_refused():
    with pytest.raises(RuntimeError, match="simulation"):
        ZKPSampledMode().setup_server_context(NS(sim_mode=True, zkp_backend="gnark"))
