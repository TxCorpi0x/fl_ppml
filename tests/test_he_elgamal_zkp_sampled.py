"""he_elgamal_zkp_sampled: commit–challenge sampling over committed ciphertexts.

Includes the Step 5 adversary: a client that commits a poisoned coordinate.
It is rejected when the challenge samples that coordinate and admitted when it
doesn't. The probability of the first case is analysed in audit/sampling.md.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.sampling as sampling
import fl.core.zkp_gnark as zkp_gnark
from fl.core import elgamal_gnark as eg
from fl.privacy.he_elgamal_zkp_sampled import HeElGamalZKPSampledMode
from tests.conftest import requires_gnark

pytestmark = requires_gnark

N_COORDS = 10  # Linear(4, 2): 8 weights + 2 biases
POISON_Q = 90_000  # in range (< 2^17) but far above any chunk's share of the bound


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    from fl.keys.he_elgamal import generate

    d = tmp_path_factory.mktemp("elgamal_sampled_keys")
    secret, public = str(d / "secret.json"), str(d / "public.json")
    generate(secret_path=secret, public_path=public)
    return NS(sim_mode=False, he_elgamal_secret_path=secret, he_elgamal_public_path=public, min_fit_clients=1)


@pytest.fixture(autouse=True)
def _env(monkeypatch, gnark_service_url):
    monkeypatch.setattr(zkp_gnark, "DEFAULT_SERVICE_URL", gnark_service_url)
    monkeypatch.setenv("FL_ELGAMAL_CHUNK", "2")
    monkeypatch.setenv("FL_ELGAMAL_SCALE", "1000")
    monkeypatch.setenv("FL_ZKP_MAX_NORM", "100.0")
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.3")  # 3 of 10 coordinates


def _net(offset=0.0):
    net = torch.nn.Linear(4, 2)
    with torch.no_grad():
        net.weight.copy_(torch.linspace(-0.4, 0.4, 8).reshape(2, 4) + offset)
        net.bias.copy_(torch.tensor([0.05, -0.05]) + offset)
    return net


def _server(keys):
    mode = HeElGamalZKPSampledMode()
    ctx = mode.setup_server_context(keys)
    mode.bind_server_model(ctx, torch.nn.Linear(4, 2))
    return mode, ctx


def _fit(params, metrics=None, n=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), n, metrics or {})


def _step(client, ctx, server, server_round, net):
    client.on_fit_config(ctx, {"server_round": server_round, **server.fit_config(server_round)})
    params = client.send_parameters(net, ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def _decrypt(keys, params):
    mode = HeElGamalZKPSampledMode()
    ctx = mode.setup_client_context(keys)
    net = torch.nn.Linear(4, 2)
    mode.receive_parameters(net, parameters_to_ndarrays(params), ctx, sim_mode=False)
    return np.concatenate([net.weight.detach().numpy().reshape(-1), net.bias.detach().numpy()])


def _poison_commitment(keys, ctx, params, coordinate):
    """Replace one committed ciphertext with an encryption of POISON_Q.

    The client keeps its honest values and randomness, so its challenge
    response proves the honest value, which no longer matches the commitment.
    """
    pk = ctx["pk"]
    bad_ct, _ = eg.encrypt_values(pk, np.array([POISON_Q], dtype=np.int64))
    flat = bytearray(b"".join(p.tobytes() for p in params[1:]))
    flat[coordinate * eg.CIPHERTEXT_BYTES : (coordinate + 1) * eg.CIPHERTEXT_BYTES] = bad_ct
    return [params[0], np.frombuffer(bytes(flat[: 8 * 64]), dtype=np.uint8), np.frombuffer(bytes(flat[8 * 64 :]), dtype=np.uint8)]


def test_honest_clients_commit_then_prove_and_aggregate(keys):
    server, sctx = _server(keys)
    clients = {cid: (HeElGamalZKPSampledMode(), None) for cid in ("a", "b")}
    ctxs = {cid: clients[cid][0].setup_client_context(keys) for cid in clients}
    nets = {"a": _net(), "b": _net(0.1)}

    commits = [(NS(cid=c), _fit(_step(clients[c][0], ctxs[c], server, 1, nets[c])[0], n=10 if c == "a" else 30)) for c in clients]
    assert server.aggregate_fit_override(1, commits, [], sctx, keys)[1]["round_outcome"] == "committed"
    responses = [(NS(cid=c), _fit(*_step(clients[c][0], ctxs[c], server, 2, nets[c]))) for c in clients]
    params, out = server.aggregate_fit_override(2, responses, [], sctx, keys)

    assert out["round_outcome"] == "aggregated" and out["admitted"] == 2
    assert server.last_round_report["sampled_coordinates"] == 3
    flat = lambda net: np.concatenate([net.weight.detach().numpy().reshape(-1), net.bias.detach().numpy()])
    np.testing.assert_allclose(_decrypt(keys, params), (10 * flat(nets["a"]) + 30 * flat(nets["b"])) / 40, atol=1e-3)


@pytest.mark.parametrize("poison_is_sampled", [True, False])
def test_poisoned_commitment_is_caught_only_if_sampled(keys, monkeypatch, poison_is_sampled):
    seed = "11" * 32
    monkeypatch.setattr("fl.privacy.commit_challenge.new_round_seed", lambda: seed)
    sampled = set(sampling.sample_indices(seed, N_COORDS, 3).tolist())
    coordinate = min(sampled) if poison_is_sampled else min(set(range(N_COORDS)) - sampled)

    server, sctx = _server(keys)
    honest, attacker = HeElGamalZKPSampledMode(), HeElGamalZKPSampledMode()
    hctx, actx = honest.setup_client_context(keys), attacker.setup_client_context(keys)

    honest_commit, _ = _step(honest, hctx, server, 1, _net())
    attack_commit, _ = _step(attacker, actx, server, 1, _net())
    attack_commit = _poison_commitment(keys, actx, attack_commit, coordinate)
    server.aggregate_fit_override(1, [(NS(cid="honest"), _fit(honest_commit)), (NS(cid="attacker"), _fit(attack_commit))], [], sctx, keys)

    responses = [
        (NS(cid="honest"), _fit(*_step(honest, hctx, server, 2, _net()))),
        (NS(cid="attacker"), _fit(*_step(attacker, actx, server, 2, _net()))),
    ]
    params, out = server.aggregate_fit_override(2, responses, [], sctx, keys)
    report = server.last_round_report

    if poison_is_sampled:
        assert report["admitted"] == ["honest"] and "attacker" in report["rejected"]
    else:
        # Undetected by design: an unsampled coordinate carries no proof.
        assert sorted(report["admitted"]) == ["attacker", "honest"]
        poisoned_mean = _decrypt(keys, params)[coordinate]
        assert poisoned_mean > 10.0


def test_server_verification_outage_aborts_the_challenge(keys, monkeypatch):
    server, sctx = _server(keys)
    client = HeElGamalZKPSampledMode()
    ctx = client.setup_client_context(keys)
    commit, _ = _step(client, ctx, server, 1, _net())
    server.aggregate_fit_override(1, [(NS(cid="c"), _fit(commit))], [], sctx, keys)
    response = _step(client, ctx, server, 2, _net())
    monkeypatch.setattr(zkp_gnark, "DEFAULT_SERVICE_URL", "http://127.0.0.1:1")

    params, out = server.aggregate_fit_override(2, [(NS(cid="c"), _fit(*response))], [], sctx, keys)

    assert params is None and out == {"round_outcome": "infrastructure_abort"}
    assert server._last_anchor_data is None
