"""he_elgamal_zkp_sampled: commit–challenge sampling over committed update ciphertexts.

Includes a commitment adversary (a committed coordinate that doesn't match what
the client later proves) and a bound adversary (a committed coordinate whose
update is far outside the bound). Each is rejected when the challenge samples
that coordinate and admitted when it doesn't. The probability of the first case
is analysed in docs/ZKP.md, section 6.4.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import ppflx.core.sampling as sampling
from ppflx.core import elgamal_gnark as eg
from ppflx.privacy.he_elgamal_zkp_sampled import HeElGamalZKPSampledMode
from tests.lib.conftest import break_gnark, requires_gnark, use_gnark

pytestmark = requires_gnark

N_COORDS = 10  # Linear(4, 2): 8 weights + 2 biases
SCALE, BOUND = 1000, 0.5
POISON_Q = 90_000  # in range (< 2^17) but an update of 90 ≫ BOUND


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    from ppflx.keys.he_elgamal import generate

    d = tmp_path_factory.mktemp("elgamal_sampled_keys")
    secret, public = str(d / "secret.json"), str(d / "public.json")
    generate(secret_path=secret, public_path=public)
    return NS(sim_mode=False, he_elgamal_secret_path=secret, he_elgamal_public_path=public, min_fit_clients=1)


@pytest.fixture(autouse=True)
def _env(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)
    monkeypatch.setenv("FL_ELGAMAL_SCALE", str(SCALE))
    monkeypatch.setenv("FL_ZKP_MAX_NORM", str(BOUND))
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.3")  # 3 of 10 coordinates


def _net(offset=0.0):
    net = torch.nn.Linear(4, 2)
    with torch.no_grad():
        net.weight.copy_(torch.linspace(-0.4, 0.4, 8).reshape(2, 4) + offset)
        net.bias.copy_(torch.tensor([0.05, -0.05]) + offset)
    return net


def _arrays(net):
    return [t.detach().numpy().copy() for t in net.state_dict().values()]


def _flat(arrays):
    return np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)


def _server(keys):
    mode = HeElGamalZKPSampledMode()
    ctx = mode.setup_server_context(keys)
    mode.bind_server_model(ctx, _net())
    return mode, ctx


def _fit(params, metrics=None, n=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), n, metrics or {})


def _step(client, ctx, server, server_round, local, global_params=None):
    client.on_fit_config(ctx, {"server_round": server_round, **server.fit_config(server_round)})
    if global_params is not None:  # commit rounds download the global model
        client.receive_parameters(torch.nn.Linear(4, 2), list(global_params), ctx, sim_mode=False)
    params = client.send_parameters(local, ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def _decrypt(keys, params):
    mode = HeElGamalZKPSampledMode()
    ctx = mode.setup_client_context(keys)
    net = torch.nn.Linear(4, 2)
    mode.receive_parameters(net, parameters_to_ndarrays(params), ctx, sim_mode=False)
    return _flat(_arrays(net))


def _replace_committed(ctx, params, coordinate, q_value):
    """Swap one committed ciphertext for an encryption of q_value, keeping the client's stored values."""
    bad_ct, _ = eg.encrypt_values(ctx["pk"], np.array([q_value], dtype=np.int64))
    flat = bytearray(b"".join(p.tobytes() for p in params[1:]))
    flat[coordinate * eg.CIPHERTEXT_BYTES : (coordinate + 1) * eg.CIPHERTEXT_BYTES] = bad_ct
    return [params[0], np.frombuffer(bytes(flat[: 8 * 64]), dtype=np.uint8), np.frombuffer(bytes(flat[8 * 64 :]), dtype=np.uint8)]


def test_honest_clients_commit_then_prove_and_aggregate(keys):
    server, sctx = _server(keys)
    g = _arrays(_net())
    clients = {c: HeElGamalZKPSampledMode() for c in ("a", "b")}
    ctxs = {c: clients[c].setup_client_context(keys) for c in clients}
    locals_ = {"a": _net(0.01), "b": _net(-0.02)}

    commits = [(NS(cid=c), _fit(_step(clients[c], ctxs[c], server, 1, locals_[c], g)[0], n=10 if c == "a" else 30)) for c in clients]
    assert server.aggregate_fit_override(1, commits, [], sctx, keys)[1]["round_outcome"] == "committed"
    responses = [(NS(cid=c), _fit(*_step(clients[c], ctxs[c], server, 2, locals_[c]))) for c in clients]
    params, out = server.aggregate_fit_override(2, responses, [], sctx, keys)

    assert out["round_outcome"] == "aggregated" and out["admitted"] == 2
    assert server.last_round_report["sampled_coordinates"] == 3
    expected = (10 * _flat(_arrays(locals_["a"])) + 30 * _flat(_arrays(locals_["b"]))) / 40
    np.testing.assert_allclose(_decrypt(keys, params), expected, atol=1e-3)
    assert sctx["global"].weight == 40


@pytest.mark.parametrize("attack", ["mismatched_commitment", "over_bound_commitment"])
@pytest.mark.parametrize("poison_is_sampled", [True, False])
def test_poisoned_commitment_is_caught_only_if_sampled(keys, monkeypatch, attack, poison_is_sampled):
    seed = "11" * 32
    monkeypatch.setattr("ppflx.privacy.commit_challenge.new_round_seed", lambda: seed)
    sampled = set(sampling.sample_indices(seed, N_COORDS, 3).tolist())
    coordinate = min(sampled) if poison_is_sampled else min(set(range(N_COORDS)) - sampled)

    server, sctx = _server(keys)
    g = _arrays(_net())
    honest, attacker = HeElGamalZKPSampledMode(), HeElGamalZKPSampledMode()
    hctx, actx = honest.setup_client_context(keys), attacker.setup_client_context(keys)

    honest_commit, _ = _step(honest, hctx, server, 1, _net(0.01), g)
    attack_commit, _ = _step(attacker, actx, server, 1, _net(0.01), g)
    attack_commit = _replace_committed(actx, attack_commit, coordinate, POISON_Q)
    if attack == "over_bound_commitment":
        # Keep the stored values consistent with the poisoned ciphertext, so
        # the response is a valid proof of a genuinely over-bound update.
        _, rand = eg.encrypt_values(actx["pk"], np.array([POISON_Q], dtype=np.int64))
        ct, rand_all = eg.encrypt_values(actx["pk"], np.where(np.arange(N_COORDS) == coordinate, POISON_Q, actx["commitment"]["q"]))
        actx["commitment"]["q"] = np.where(np.arange(N_COORDS) == coordinate, POISON_Q, actx["commitment"]["q"]).astype("<i8")
        actx["commitment"]["rand"] = rand_all
        attack_commit = [attack_commit[0], np.frombuffer(ct[: 8 * 64], dtype=np.uint8), np.frombuffer(ct[8 * 64 :], dtype=np.uint8)]
    server.aggregate_fit_override(1, [(NS(cid="honest"), _fit(honest_commit)), (NS(cid="attacker"), _fit(attack_commit))], [], sctx, keys)

    try:
        attacker_response = _fit(*_step(attacker, actx, server, 2, _net(0.01)))
    except RuntimeError as exc:  # an honest client library refuses to declare an over-bound update
        assert attack == "over_bound_commitment" and poison_is_sampled and "exceeds the bound" in str(exc)
        return
    responses = [(NS(cid="honest"), _fit(*_step(honest, hctx, server, 2, _net(0.01)))), (NS(cid="attacker"), attacker_response)]
    params, out = server.aggregate_fit_override(2, responses, [], sctx, keys)
    report = server.last_round_report

    if poison_is_sampled:
        assert report["admitted"] == ["honest"] and "attacker" in report["rejected"]
    else:
        # Undetected by design: an unsampled coordinate carries no proof.
        assert sorted(report["admitted"]) == ["attacker", "honest"]
        assert _decrypt(keys, params)[coordinate] > 10.0


def test_over_bound_sampled_update_with_valid_proofs_is_rejected(keys, monkeypatch):
    """A client library that skips the declaration check still can't get an over-bound update admitted."""
    seed = "22" * 32
    monkeypatch.setattr("ppflx.privacy.commit_challenge.new_round_seed", lambda: seed)
    coordinate = int(sampling.sample_indices(seed, N_COORDS, 3)[0])
    monkeypatch.setattr("ppflx.privacy.he_elgamal_zkp_sampled.split_bound", lambda energies, total: [max(1, int(e)) for e in energies])

    server, sctx = _server(keys)
    g = _arrays(_net())
    attacker = HeElGamalZKPSampledMode()
    actx = attacker.setup_client_context(keys)
    _step(attacker, actx, server, 1, _net(0.01), g)
    q = np.where(np.arange(N_COORDS) == coordinate, POISON_Q, actx["commitment"]["q"]).astype("<i8")
    ct, rand = eg.encrypt_values(actx["pk"], q)
    actx["commitment"].update(q=q, rand=rand)
    commit = [np.array([eg.HEADER_MAGIC, 1, 1], dtype=np.int64), np.frombuffer(ct[: 8 * 64], dtype=np.uint8), np.frombuffer(ct[8 * 64 :], dtype=np.uint8)]
    server.aggregate_fit_override(1, [(NS(cid="attacker"), _fit(commit))], [], sctx, keys)

    params, out = server.aggregate_fit_override(2, [(NS(cid="attacker"), _fit(*_step(attacker, actx, server, 2, _net(0.01))))], [], sctx, keys)

    assert params is None
    assert "declared update bounds" in server.last_round_report["rejected"]["attacker"]


def test_server_verification_outage_aborts_the_challenge(keys, monkeypatch):
    server, sctx = _server(keys)
    client = HeElGamalZKPSampledMode()
    ctx = client.setup_client_context(keys)
    commit, _ = _step(client, ctx, server, 1, _net(0.01), _arrays(_net()))
    server.aggregate_fit_override(1, [(NS(cid="c"), _fit(commit))], [], sctx, keys)
    response = _step(client, ctx, server, 2, _net(0.01))
    break_gnark(monkeypatch)

    params, out = server.aggregate_fit_override(2, [(NS(cid="c"), _fit(*response))], [], sctx, keys)

    assert params is None and out == {"round_outcome": "infrastructure_abort"}
    assert server._last_anchor_data is None
