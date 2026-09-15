"""he_elgamal_zkp: ciphertext-bound update proofs (audit/binding.md Design B + A, audit/norm.md).

Uses the real gnark service and real keys. The attack tests mirror
tests/test_zkp_binding_attack.py and must pass as rejections here.
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl.core import elgamal_gnark as eg
from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode
from tests.conftest import break_gnark, requires_gnark, use_gnark

pytestmark = requires_gnark

SCALE, BOUND = 1000, 0.5


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    from fl.keys.he_elgamal import generate

    d = tmp_path_factory.mktemp("elgamal_keys")
    secret, public = str(d / "secret_key.json"), str(d / "public_key.json")
    generate(secret_path=secret, public_path=public)
    return NS(sim_mode=False, he_elgamal_secret_path=secret, he_elgamal_public_path=public, min_fit_clients=1)


@pytest.fixture(autouse=True)
def _service(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)  # test keys: ElGamal chunk 4, so 8 + 2 parameters → chunks of 4, 4, 2
    monkeypatch.setenv("FL_ELGAMAL_SCALE", str(SCALE))
    monkeypatch.setenv("FL_ZKP_MAX_NORM", str(BOUND))


def _arrays(weight, bias):
    return [np.array(weight, dtype=np.float32), np.array(bias, dtype=np.float32)]


def _net(arrays):
    net = torch.nn.Linear(4, 2)
    with torch.no_grad():
        net.weight.copy_(torch.from_numpy(arrays[0]))
        net.bias.copy_(torch.from_numpy(arrays[1]))
    return net


def _plus(arrays, delta):
    return [(a + np.asarray(d, dtype=np.float32)).astype(np.float32) for a, d in zip(arrays, delta)]


def _flat(arrays):
    return np.concatenate([a.reshape(-1) for a in arrays]).astype(np.float64)


INIT = _arrays([[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]], [0.05, -0.05])
UP_A = _arrays([[0.01, 0, 0, 0], [0, 0.02, 0, 0]], [0, 0.01])
UP_B = _arrays([[0, -0.03, 0, 0.01], [0, 0, 0.02, 0]], [0.01, 0])
HUGE = _arrays([[9.0] * 4, [9.0] * 4], [9.0, 9.0])  # ‖Δ‖ ≈ 28 ≫ BOUND


def _server(keys):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_server_context(keys)
    mode.bind_server_model(ctx, _net(INIT))
    return mode, ctx


def _client(keys, server, server_round, global_params):
    client = HeElGamalZKPMode()
    ctx = client.setup_client_context(keys)
    client.on_fit_config(ctx, {"server_round": server_round, **server.fit_config(server_round)})
    client.receive_parameters(torch.nn.Linear(4, 2), list(global_params), ctx, sim_mode=False)
    return client, ctx


def _upload(keys, server, server_round, global_params, local):
    client, ctx = _client(keys, server, server_round, global_params)
    params = client.send_parameters(_net(local), ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def _fit(params, metrics, n=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), n, metrics)


def _decrypt(keys, aggregated):
    mode = HeElGamalZKPMode()
    ctx = mode.setup_client_context(keys)
    net = torch.nn.Linear(4, 2)
    mode.receive_parameters(net, parameters_to_ndarrays(aggregated), ctx, sim_mode=False)
    return [net.weight.detach().numpy(), net.bias.detach().numpy()]


def _forge(keys, server, server_round, global_params, local, understate=False):
    """A client that skips clipping: encrypts its raw model and proves each chunk.

    Declared bounds are the true chunk energies (valid proofs, over the server's
    total), or, with ``understate``, the honest total split evenly (no proof exists).
    """
    client, ctx = _client(keys, server, server_round, global_params)
    policy, glob = ctx["policy"], ctx["global"]
    schema = [("weight", (2, 4)), ("bias", (2,))]
    q = np.round(_flat(local) * policy.scale).astype(np.int64)
    chunks = eg.chunks_for(schema, policy.chunk_size)
    indices = [eg.chunk_indices(schema, c) for c in chunks]
    bounds = [max(1, eg.update_energy(q[i], glob.centered_sums()[i], glob.weight)) for i in indices]
    if understate:
        total = eg.total_bound_sq(BOUND, policy.scale, glob.weight, q.size)
        bounds = [total // len(chunks)] * len(chunks)
    layers = [bytearray(8 * eg.CIPHERTEXT_BYTES), bytearray(2 * eg.CIPHERTEXT_BYTES)]
    proofs = []
    for chunk, idx, bound in zip(chunks, indices, bounds):
        ct, proof = eg.prove_chunk(
            ctx["pk"], ctx["sk"], q[idx], bound, eg.context_value(server_round, chunk), glob.request(idx, prover=True)
        )
        layers[chunk.layer][chunk.start * eg.CIPHERTEXT_BYTES : (chunk.start + chunk.size) * eg.CIPHERTEXT_BYTES] = ct
        proofs.append({"layer": chunk.layer, "chunk": chunk.index, "bound_sq": str(bound), "proof_b64": proof, "vk_sha256": eg.pinned_vk()})
    header = np.array([eg.HEADER_MAGIC, 1, server_round], dtype=np.int64)
    params = [header] + [np.frombuffer(bytes(b), dtype=np.uint8) for b in layers]
    return params, {"zkp_proofs_json": json.dumps(proofs)}


# ─── Honest protocol ─────────────────────────────────────────────────────────


def test_honest_updates_aggregate_across_two_rounds(keys):
    server, sctx = _server(keys)
    a = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    b = _upload(keys, server, 1, INIT, _plus(INIT, UP_B))

    agg1, out = server.aggregate_fit_override(1, [(NS(cid="a"), _fit(*a, 10)), (NS(cid="b"), _fit(*b, 30))], [], sctx, keys)

    assert out == {"elgamal_admitted": 2, "elgamal_rejected": 0}
    assert server._last_anchor_data["client_ids"] == ["a", "b"]
    mean1 = [(10 * x + 30 * y) / 40 for x, y in zip(_plus(INIT, UP_A), _plus(INIT, UP_B))]
    for got, want in zip(_decrypt(keys, agg1), mean1):
        np.testing.assert_allclose(got, want, atol=1e-3)

    # Round 2 proves against the encrypted aggregate (W = 40), which the server can't read.
    assert sctx["global"].weight == 40 and sctx["global"].plain is None
    g2 = parameters_to_ndarrays(agg1)
    c = _upload(keys, server, 2, g2, _plus(mean1, UP_A))
    agg2, out2 = server.aggregate_fit_override(2, [(NS(cid="c"), _fit(*c, 10))], [], sctx, keys)

    assert out2 == {"elgamal_admitted": 1, "elgamal_rejected": 0}
    for got, want in zip(_decrypt(keys, agg2), _plus(mean1, UP_A)):
        np.testing.assert_allclose(got, want, atol=2e-3)


def test_client_clips_an_oversized_update_and_is_admitted(keys):
    server, sctx = _server(keys)
    params, metrics = _upload(keys, server, 1, INIT, _plus(INIT, HUGE))

    agg, out = server.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, metrics))], [], sctx, keys)

    assert out["elgamal_admitted"] == 1
    assert metrics["zkp_update_clipped"] == 1 and metrics["zkp_update_norm"] > 20
    moved = np.linalg.norm(_flat(_decrypt(keys, agg)) - _flat(INIT))
    assert moved <= BOUND + np.sqrt(10) / (2 * SCALE) + 1e-6


# ─── Attacks ─────────────────────────────────────────────────────────────────


def test_unclipped_large_update_with_valid_proofs_is_rejected_by_the_total_bound(keys):
    """audit/norm.md A1/A2: the proofs are valid, but their declared bounds exceed the server's total."""
    server, sctx = _server(keys)
    honest = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    attacker = _forge(keys, server, 1, INIT, _plus(INIT, HUGE))

    agg, out = server.aggregate_fit_override(
        1, [(NS(cid="honest"), _fit(*honest)), (NS(cid="attacker"), _fit(*attacker))], [], sctx, keys
    )

    assert out == {"elgamal_admitted": 1, "elgamal_rejected": 1}
    assert "declared update bounds" in server.last_round_report["rejected"]["attacker"]
    for got, want in zip(_decrypt(keys, agg), _plus(INIT, UP_A)):
        np.testing.assert_allclose(got, want, atol=1e-3)


def test_large_update_cannot_be_proved_under_bounds_that_fit_the_total(keys):
    server, _ = _server(keys)
    with pytest.raises(eg.ElGamalRejected, match="not satisfied"):
        _forge(keys, server, 1, INIT, _plus(INIT, HUGE), understate=True)


def test_update_proved_against_a_stale_global_model_is_rejected(keys):
    server, sctx = _server(keys)
    a = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    server.aggregate_fit_override(1, [(NS(cid="a"), _fit(*a))], [], sctx, keys)

    # Round 2, but measured against the initial model instead of the aggregate.
    stale = _upload(keys, server, 2, INIT, _plus(INIT, UP_B))
    agg, out = server.aggregate_fit_override(2, [(NS(cid="stale"), _fit(*stale))], [], sctx, keys)

    assert agg is None and out == {"elgamal_admitted": 0, "elgamal_rejected": 1}


def test_proof_over_honest_vector_does_not_admit_another_ciphertext(keys):
    server, sctx = _server(keys)
    honest = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    other_params, _ = _upload(keys, server, 1, INIT, _plus(INIT, UP_B))
    swapped = (other_params, honest[1])  # B's ciphertexts with A's valid proofs

    agg, out = server.aggregate_fit_override(1, [(NS(cid="swap"), _fit(*swapped))], [], sctx, keys)

    assert agg is None and out == {"elgamal_admitted": 0, "elgamal_rejected": 1}


def test_tampered_declared_bound_is_rejected(keys):
    server, sctx = _server(keys)
    params, metrics = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    proofs = json.loads(metrics["zkp_proofs_json"])
    proofs[0]["bound_sq"] = str(int(proofs[0]["bound_sq"]) + 1)

    agg, out = server.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, {**metrics, "zkp_proofs_json": json.dumps(proofs)}))], [], sctx, keys)

    assert agg is None and out["elgamal_admitted"] == 0


def test_proof_replayed_from_an_earlier_round_is_rejected(keys):
    params, metrics = _upload(keys, _server(keys)[0], 1, INIT, _plus(INIT, UP_A))
    server, sctx = _server(keys)

    aggregated, out = server.aggregate_fit_override(2, [(NS(cid="c"), _fit(params, metrics))], [], sctx, keys)

    assert aggregated is None
    assert out == {"elgamal_admitted": 0, "elgamal_rejected": 1}
    assert server._last_anchor_data is None


@pytest.mark.parametrize("tamper", ["drop_chunk", "duplicate_chunk", "swap_chunks"])
def test_incomplete_or_rearranged_proof_coverage_is_rejected(keys, tamper):
    server, sctx = _server(keys)
    params, metrics = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    proofs = json.loads(metrics["zkp_proofs_json"])
    if tamper == "drop_chunk":
        proofs = proofs[:-1]
    elif tamper == "duplicate_chunk":
        proofs = proofs + [proofs[0]]
    else:
        proofs[0]["proof_b64"], proofs[1]["proof_b64"] = proofs[1]["proof_b64"], proofs[0]["proof_b64"]
    metrics = dict(metrics, zkp_proofs_json=json.dumps(proofs))

    aggregated, out = server.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, metrics))], [], sctx, keys)

    assert aggregated is None
    assert out["elgamal_admitted"] == 0


# ─── Fail closed ─────────────────────────────────────────────────────────────


def test_unreachable_service_aborts_round_without_aggregating(keys, monkeypatch):
    server, sctx = _server(keys)
    params, metrics = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))
    break_gnark(monkeypatch)

    aggregated, out = server.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, metrics))], [], sctx, keys)

    assert aggregated is None
    assert out == {"elgamal_round_aborted": 1}
    assert server._last_anchor_data is None


def test_aggregate_weight_beyond_the_circuit_aborts_the_round(keys):
    server, sctx = _server(keys)
    params, metrics = _upload(keys, server, 1, INIT, _plus(INIT, UP_A))

    aggregated, out = server.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, metrics, n=eg.WEIGHT_LIMIT))], [], sctx, keys)

    assert aggregated is None and out == {"elgamal_round_aborted": 1}
    assert sctx["global"].plain is not None  # global model unchanged


def test_client_refuses_to_prove_without_a_global_model(keys):
    server, _ = _server(keys)
    client = HeElGamalZKPMode()
    ctx = client.setup_client_context(keys)
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    with pytest.raises(RuntimeError, match="no global model"):
        client.send_parameters(_net(INIT), ctx, sim_mode=False)


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
        HeElGamalZKPMode().setup_client_context(NS(**{**vars(keys), "sim_mode": True}))


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

    config = NS(dataset="mnist")
    monkeypatch.delenv("FL_CONCRETE_TFHE_FORCE_REAL", raising=False)
    monkeypatch.delenv("FL_CONCRETE_TFHE_ALLOW_SIMULATED", raising=False)
    with pytest.raises(RuntimeError, match="does not encrypt"):
        _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False)

    monkeypatch.setenv("FL_CONCRETE_TFHE_ALLOW_SIMULATED", "1")
    assert _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False) is True
    monkeypatch.setenv("FL_CONCRETE_TFHE_FORCE_REAL", "1")
    assert _auto_disable_real_tfhe_for_images(config, requested_sim_mode=False) is False
    assert _auto_disable_real_tfhe_for_images(NS(dataset="healthcare"), False) is False


def test_harness_does_not_force_partial_encryption():
    from fl.compare.experiment import _grpc_env

    assert "FL_ENCRYPT_LAYERS" not in _grpc_env({})
    assert _grpc_env({"FL_ENCRYPT_LAYERS": "ALL"})["FL_ENCRYPT_LAYERS"] == "ALL"
