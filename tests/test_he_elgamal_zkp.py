"""he_elgamal_zkp: ciphertext-bound proofs (audit/binding.md Design B + A).

Uses the real gnark service and real keys. The attack tests mirror
tests/test_zkp_binding_attack.py and must pass as rejections here.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.zkp_gnark as zkp_gnark
from fl.core import elgamal_gnark as eg
from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode
from tests.conftest import break_gnark, requires_gnark, use_gnark

pytestmark = requires_gnark


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    from fl.keys.he_elgamal import generate

    d = tmp_path_factory.mktemp("elgamal_keys")
    secret, public = str(d / "secret_key.json"), str(d / "public_key.json")
    generate(secret_path=secret, public_path=public)
    return SimpleNamespace(sim_mode=False, he_elgamal_secret_path=secret, he_elgamal_public_path=public)


@pytest.fixture(autouse=True)
def _service(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)  # test keys: ElGamal chunk 4, so 8 + 2 parameters → chunks of 4, 4, 2
    monkeypatch.setenv("FL_ELGAMAL_SCALE", "1000")
    monkeypatch.setenv("FL_ZKP_MAX_NORM", "100.0")


def _net(weight, bias):
    net = torch.nn.Linear(4, 2)
    with torch.no_grad():
        net.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        net.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return net


HONEST = ([[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]], [0.05, -0.05])
OTHER = ([[0.2, 0.1, -0.1, 0.0], [-0.3, 0.4, 0.2, 0.1]], [0.01, 0.02])
POISON = ([[99.0] * 4, [99.0] * 4], [99.0, 99.0])  # ||w|| ≈ 280 > 100, each |q| < 2^17


def _upload(keys, weights, server_round):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_client_context(keys)
    mode.on_fit_config(ctx, {"server_round": server_round})
    params = mode.send_parameters(_net(*weights), ctx, sim_mode=False)
    return params, mode.post_fit_metrics(ctx)


def _fit_res(params, metrics, num_examples):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), num_examples, metrics)


def _server(keys):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_server_context(keys)
    mode.bind_server_model(ctx, torch.nn.Linear(4, 2))
    return mode, ctx


def _decrypt(keys, aggregated):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_client_context(keys)
    net = torch.nn.Linear(4, 2)
    mode.receive_parameters(net, parameters_to_ndarrays(aggregated), ctx, sim_mode=False)
    return net.weight.detach().numpy(), net.bias.detach().numpy()


def test_honest_clients_aggregate_to_weighted_mean(keys):
    up_a = _upload(keys, HONEST, 1)
    up_b = _upload(keys, OTHER, 1)
    mode, ctx = _server(keys)
    results = [
        (SimpleNamespace(cid="a"), _fit_res(*up_a, 10)),
        (SimpleNamespace(cid="b"), _fit_res(*up_b, 30)),
    ]

    aggregated, metrics = mode.aggregate_fit_override(1, results, [], ctx, None)

    assert metrics == {"elgamal_admitted": 2, "elgamal_rejected": 0}
    assert mode._last_anchor_data["client_ids"] == ["a", "b"]
    weight, bias = _decrypt(keys, aggregated)
    np.testing.assert_allclose(weight, (10 * np.array(HONEST[0]) + 30 * np.array(OTHER[0])) / 40, atol=1e-3)
    np.testing.assert_allclose(bias, (10 * np.array(HONEST[1]) + 30 * np.array(OTHER[1])) / 40, atol=1e-3)


def test_client_cannot_prove_a_poisoned_vector(keys):
    with pytest.raises(eg.ElGamalRejected):
        _upload(keys, POISON, 1)


def _poisoned_ciphertexts(keys):
    """Ciphertexts of POISON, obtainable only by proving under an inflated bound."""
    pk = HeElGamalZKPMode().setup_client_context(keys)["pk"]
    flat = [eg.quantize(np.array(POISON[0]).reshape(-1), 1000, "w"), eg.quantize(np.array(POISON[1]), 1000, "b")]
    layers = []
    for q in flat:
        chunks = [eg.prove_chunk(pk, q[i : i + 4], 10**15, 0)[0] for i in range(0, len(q), 4)]
        layers.append(np.frombuffer(b"".join(chunks), dtype=np.uint8))
    return layers


def test_proof_over_honest_vector_does_not_admit_poisoned_ciphertext(keys):
    honest = _upload(keys, HONEST, 1)
    attacker_params, attacker_metrics = _upload(keys, HONEST, 1)  # valid proofs over A
    attacker_params = [attacker_params[0]] + _poisoned_ciphertexts(keys)  # ciphertexts of B
    mode, ctx = _server(keys)
    results = [
        (SimpleNamespace(cid="honest"), _fit_res(*honest, 10)),
        (SimpleNamespace(cid="attacker"), _fit_res(attacker_params, attacker_metrics, 10)),
    ]

    aggregated, metrics = mode.aggregate_fit_override(1, results, [], ctx, None)

    assert metrics == {"elgamal_admitted": 1, "elgamal_rejected": 1}
    assert mode._last_anchor_data["client_ids"] == ["honest"]
    weight, _ = _decrypt(keys, aggregated)
    np.testing.assert_allclose(weight, HONEST[0], atol=1e-3)


def test_proof_replayed_from_an_earlier_round_is_rejected(keys):
    params, metrics = _upload(keys, HONEST, 1)
    mode, ctx = _server(keys)

    aggregated, out = mode.aggregate_fit_override(2, [(SimpleNamespace(cid="c"), _fit_res(params, metrics, 10))], [], ctx, None)

    assert aggregated is None
    assert out == {"elgamal_admitted": 0, "elgamal_rejected": 1}
    assert mode._last_anchor_data is None


@pytest.mark.parametrize("tamper", ["drop_chunk", "duplicate_chunk", "swap_chunks"])
def test_incomplete_or_rearranged_proof_coverage_is_rejected(keys, tamper):
    params, metrics = _upload(keys, HONEST, 1)
    proofs = json.loads(metrics["zkp_proofs_json"])
    if tamper == "drop_chunk":
        proofs = proofs[:-1]
    elif tamper == "duplicate_chunk":
        proofs = proofs + [proofs[0]]
    else:
        proofs[0]["proof_b64"], proofs[1]["proof_b64"] = proofs[1]["proof_b64"], proofs[0]["proof_b64"]
    metrics = dict(metrics, zkp_proofs_json=json.dumps(proofs))
    mode, ctx = _server(keys)

    aggregated, out = mode.aggregate_fit_override(1, [(SimpleNamespace(cid="c"), _fit_res(params, metrics, 10))], [], ctx, None)

    assert aggregated is None
    assert out["elgamal_admitted"] == 0


def test_unreachable_service_aborts_round_without_aggregating(keys, monkeypatch):
    params, metrics = _upload(keys, HONEST, 1)
    mode, ctx = _server(keys)
    break_gnark(monkeypatch)

    aggregated, out = mode.aggregate_fit_override(1, [(SimpleNamespace(cid="c"), _fit_res(params, metrics, 10))], [], ctx, None)

    assert aggregated is None
    assert out == {"elgamal_round_aborted": 1}
    assert mode._last_anchor_data is None


def test_server_without_bound_schema_refuses_to_aggregate(keys):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_server_context(keys)
    with pytest.raises(RuntimeError, match="schema not bound"):
        mode.aggregate_fit_override(1, [], [], ctx, None)


def test_server_key_file_must_not_contain_secret(keys):
    from fl.keys.he_elgamal import load_server

    with pytest.raises(ValueError, match="secret key"):
        load_server(keys.he_elgamal_secret_path)


def test_simulation_mode_is_refused(keys):
    with pytest.raises(RuntimeError, match="simulation"):
        HeElGamalZKPMode().setup_client_context(SimpleNamespace(**{**vars(keys), "sim_mode": True}))


# ─── B-1 / B-2 regressions ───────────────────────────────────────────────────


def test_tenseal_encrypts_every_layer_by_default(monkeypatch):
    from fl.core.security import _CVEC_MAGIC, make_tenseal_context
    from fl.privacy.he_tenseal import HeTensealMode, _unpack_arrays
    import zlib

    monkeypatch.delenv("FL_ENCRYPT_LAYERS", raising=False)
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.Flatten(), torch.nn.Linear(8, 2))
    arrays = _unpack_arrays(HeTensealMode()._encrypt_params(net, make_tenseal_context()))

    assert len(arrays) == len(net.state_dict())
    for arr in arrays:
        assert zlib.decompress(arr.tobytes())[:4] == _CVEC_MAGIC


def test_tenseal_rejects_encrypt_layer_names_missing_from_model(monkeypatch):
    from fl.core.security import make_tenseal_context
    from fl.privacy.he_tenseal import HeTensealMode

    monkeypatch.setenv("FL_ENCRYPT_LAYERS", "model.0.weight")
    with pytest.raises(ValueError, match="not in the model"):
        HeTensealMode()._encrypt_params(torch.nn.Linear(2, 2), make_tenseal_context())


def test_tfhe_on_image_dataset_refuses_silent_plaintext(monkeypatch):
    from fl.privacy.he_concrete_tfhe import _auto_disable_real_tfhe_for_images

    config = SimpleNamespace(dataset="mnist")
    monkeypatch.delenv("FL_CONCRETE_TFHE_FORCE_REAL", raising=False)
    monkeypatch.delenv("FL_CONCRETE_TFHE_ALLOW_SIMULATED", raising=False)
    with pytest.raises(RuntimeError, match="does not encrypt"):
        _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False)

    monkeypatch.setenv("FL_CONCRETE_TFHE_ALLOW_SIMULATED", "1")
    assert _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False) is True
    monkeypatch.setenv("FL_CONCRETE_TFHE_FORCE_REAL", "1")
    assert _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False) is False
    assert _auto_disable_real_tfhe_for_images(SimpleNamespace(dataset="healthcare"), False) is False


def test_harness_does_not_force_partial_encryption():
    from fl.compare.experiment import _grpc_env

    assert "FL_ENCRYPT_LAYERS" not in _grpc_env({})
    assert _grpc_env({"FL_ENCRYPT_LAYERS": "ALL"})["FL_ENCRYPT_LAYERS"] == "ALL"
