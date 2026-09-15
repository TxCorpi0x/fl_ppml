"""Proof-to-update binding and update-bound attacks.

Each attack test asserts the SECURE outcome — the attacker is not aggregated.
Tests for defects that still exist are marked xfail(strict=True), so a fix
must deliberately remove the marker. Run with ``--runxfail`` to see an attack
succeed against current code.
"""

import json
import zlib
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl.core import zkp_gnark
from fl.core.zkp_gnark import generate_gnark_proofs
from tests.conftest import requires_gnark, use_gnark

pytestmark = requires_gnark

LAYERS = ("model.0.weight", "model.0.bias")
BOUND = 1.0


class _ServerModel(torch.nn.Module):
    """Server-side model whose schema matches LAYERS."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        with torch.no_grad():
            self.model[0].weight.copy_(torch.tensor([[0.1, -0.2], [0.3, -0.4]]))
            self.model[0].bias.copy_(torch.tensor([0.05, -0.05]))


def _weights(offset):
    base = [t.detach().numpy() for t in _ServerModel().state_dict().values()]
    return OrderedDict((name, (b + np.float32(offset)).astype(np.float32)) for name, b in zip(LAYERS, base))


def _honest_weights():
    return _weights(0.01)


def _poisoned_weights():
    return _weights(1000.0)  # ‖Δ‖ = 2000 ≫ BOUND


class _Net(_ServerModel):
    def __init__(self, weights):
        super().__init__()
        self.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})


def _fit_res(params, metrics, layer_names=None):
    metrics = dict(metrics)
    if layer_names is not None:
        metrics["zkp_layer_names_json"] = json.dumps(layer_names)
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), 10, metrics)


@pytest.fixture(autouse=True)
def _gnark_backend(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)
    monkeypatch.setenv("FL_ZKP_MAX_NORM", str(BOUND))
    monkeypatch.delenv("FL_ZKP_BACKEND", raising=False)
    monkeypatch.delenv("FL_ZKP_LAYERS", raising=False)


@pytest.fixture
def config():
    return SimpleNamespace(zkp_backend="gnark", sim_mode=False)


def _plaintext_server(config):
    from fl.privacy.zkp import ZKPMode

    mode = ZKPMode()
    mode.setup_server_context(config)
    mode.bind_server_model(None, _ServerModel())
    return mode


def _upload(server, weights):
    """An honest client: download the server's global model, then clip, prove and upload `weights`."""
    from fl.privacy.zkp import ZKPMode

    client, ctx = ZKPMode(), {"backend": "gnark"}
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    client.receive_parameters(_ServerModel(), list(server._global), ctx, sim_mode=False)
    params = client.send_parameters(_Net(weights), ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def _raw_update_proofs(server, weights):
    """A client that skips clipping: proves its raw update with true per-proof bounds."""
    from fl.privacy.zkp import quantized_update

    delta = quantized_update(list(weights.values()), server._global)
    proofs, _ = generate_gnark_proofs(OrderedDict(zip(LAYERS, delta)))
    return list(weights.values()), {"zkp_proofs_json": json.dumps(proofs)}


# ─── Controls ────────────────────────────────────────────────────────────────


def test_control_client_library_refuses_to_declare_an_over_bound_update(config):
    server = _plaintext_server(config)
    params, metrics = _raw_update_proofs(server, _honest_weights())
    assert metrics  # a small update proves fine
    from fl.privacy.zkp import quantized_update

    delta = quantized_update(list(_poisoned_weights().values()), server._global)
    with pytest.raises(RuntimeError, match="exceeds the server's bound"):
        generate_gnark_proofs(OrderedDict(zip(LAYERS, delta)), total_bound_sq=zkp_gnark.policy_bound_sq(BOUND, 6))


def test_control_plaintext_zkp_rejects_mismatched_params_with_full_layer_names(config):
    """Plaintext zkp binds proofs to the parameters the server received."""
    server = _plaintext_server(config)
    honest_params, honest_metrics = _upload(server, _honest_weights())
    honest = (SimpleNamespace(cid="honest"), _fit_res(honest_params, honest_metrics, list(LAYERS)))
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(list(_poisoned_weights().values()), honest_metrics, list(LAYERS)))

    server.aggregate_fit_override(1, [honest, attacker], [], None, config)

    assert server.last_round_report["admitted"] == ["honest"]


# ─── Attacks ─────────────────────────────────────────────────────────────────


def test_plaintext_zkp_rejects_update_with_empty_layer_names(config):
    """The server verifies against its own schema, not client layer names."""
    server = _plaintext_server(config)
    honest_params, honest_metrics = _upload(server, _honest_weights())
    honest = (SimpleNamespace(cid="honest"), _fit_res(honest_params, honest_metrics, list(LAYERS)))
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(list(_poisoned_weights().values()), honest_metrics, []))

    server.aggregate_fit_override(1, [honest, attacker], [], None, config)

    admitted = server.last_round_report["admitted"]
    assert "attacker" not in admitted, f"poisoned update admitted to FedAvg: {admitted}"


def test_unclipped_large_update_with_valid_proofs_is_rejected_by_the_total_bound(config):
    """Every proof verifies, but the declared bounds exceed the server's total."""
    server = _plaintext_server(config)
    honest = (SimpleNamespace(cid="honest"), _fit_res(*_upload(server, _honest_weights())))
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(*_raw_update_proofs(server, _poisoned_weights())))

    params, _ = server.aggregate_fit_override(1, [honest, attacker], [], None, config)

    assert server.last_round_report["admitted"] == ["honest"]
    assert "declared update bounds" in server.last_round_report["rejected"]["attacker"]
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], _honest_weights()[LAYERS[0]], atol=1e-6)


def test_understated_bound_does_not_verify(config):
    server = _plaintext_server(config)
    params, metrics = _raw_update_proofs(server, _poisoned_weights())
    proofs = json.loads(metrics["zkp_proofs_json"])
    for p in proofs:
        p["bound_sq"] = "1"  # fits the total, but it isn't what was proved
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(params, {"zkp_proofs_json": json.dumps(proofs)}))

    server.aggregate_fit_override(1, [attacker], [], None, config)

    assert server.last_round_report["rejected"]["attacker"].startswith("proof verification failed")


def test_oversized_honest_update_is_clipped_and_admitted(config):
    server = _plaintext_server(config)
    params, metrics = _upload(server, _poisoned_weights())

    aggregated, _ = server.aggregate_fit_override(1, [(SimpleNamespace(cid="c"), _fit_res(params, metrics))], [], None, config)

    assert server.last_round_report["admitted"] == ["c"] and metrics["zkp_update_clipped"] == 1
    moved = np.concatenate([(a - g).reshape(-1) for a, g in zip(parameters_to_ndarrays(aggregated), _plaintext_server(config)._global)])
    assert np.linalg.norm(moved.astype(np.float64)) <= BOUND + 1e-5


def test_update_measured_against_the_wrong_global_model_is_rejected(config):
    server = _plaintext_server(config)
    params, metrics = _upload(server, _honest_weights())
    server._global = [g + np.float32(0.5) for g in server._global]  # the server's global moved on

    server.aggregate_fit_override(1, [(SimpleNamespace(cid="c"), _fit_res(params, metrics))], [], None, config)

    assert server.last_round_report["admitted"] == []


def _decrypt_aggregate(parameters, secret_ctx):
    import tenseal as ts
    from fl.core.security import _parse_cvec
    from fl.privacy.he_tenseal import _unpack_arrays

    layers = []
    for arr in _unpack_arrays(parameters_to_ndarrays(parameters)):
        raw = arr.tobytes()
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        shape, ckks_bytes = _parse_cvec(raw)
        vec = ts.ckks_vector_from(secret_ctx, ckks_bytes)
        layers.append(np.array(vec.decrypt()).reshape(shape))
    return layers


@pytest.mark.xfail(
    strict=True,
    reason="By design the CKKS composite's proof is not "
    "bound to its ciphertext; the mode is relabelled confidentiality-only. The "
    "ciphertext-bound replacement is tested in tests/test_he_elgamal_zkp.py",
)
def test_he_tenseal_zkp_rejects_ciphertext_that_does_not_match_proof(config):
    import tenseal as ts
    from fl.core.security import make_tenseal_context
    from fl.privacy.he_zkp import HeTensealZKPMode

    secret_ctx = make_tenseal_context()
    server_ctx = ts.context_from(secret_ctx.serialize(save_secret_key=False))
    mode = HeTensealZKPMode()
    mode._zkp_mode.setup_server_context(config)
    mode.bind_server_model(None, _ServerModel())
    he = mode._he_mode

    zero =OrderedDict((k, np.zeros(v.shape, dtype=np.int64)) for k, v in _honest_weights().items())
    honest_proofs, _ = generate_gnark_proofs(zero)
    attacker_proofs, _ = generate_gnark_proofs(zero)  # proves a zero update

    honest_ct = he._encrypt_params(_Net(_honest_weights()), secret_ctx, encrypt_layers=list(LAYERS))
    attacker_ct = he._encrypt_params(_Net(_poisoned_weights()), secret_ctx, encrypt_layers=list(LAYERS))  # submits vector B

    results = [
        (SimpleNamespace(cid="honest"), _fit_res(honest_ct, {"zkp_proofs_json": json.dumps(honest_proofs)})),
        (SimpleNamespace(cid="attacker"), _fit_res(attacker_ct, {"zkp_proofs_json": json.dumps(attacker_proofs)})),
    ]
    aggregated, _ = mode.aggregate_fit_override(1, results, [], server_ctx, config)

    admitted = mode.last_round_report["admitted"]
    decrypted = _decrypt_aggregate(aggregated, secret_ctx)
    assert "attacker" not in admitted, (
        f"attacker admitted {admitted}; decrypted aggregate model.0.weight = "
        f"{np.round(decrypted[0], 3).tolist()}"
    )
    np.testing.assert_allclose(decrypted[0], _honest_weights()[LAYERS[0]], atol=1e-3)
