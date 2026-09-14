"""Proof-to-update binding attacks (audit/binding.md).

Each attack test asserts the SECURE outcome — the attacker is not aggregated —
and is marked xfail(strict=True) while the defect exists. Strict xfail turns
green-by-accident into a failure, so a fix must deliberately remove the marker.
Run with ``--runxfail`` to see the attack succeed against current code.

The control tests pass today and pin down what currently does work.
"""

import json
import zlib
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.zkp_gnark as zkp_gnark
from fl.core.zkp_gnark import generate_gnark_proofs
from tests.conftest import requires_gnark

pytestmark = requires_gnark

LAYERS = ("model.0.weight", "model.0.bias")


def _honest_weights():
    return OrderedDict(
        [
            ("model.0.weight", np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)),
            ("model.0.bias", np.array([0.05, -0.05], dtype=np.float32)),
        ]
    )


def _poisoned_weights():
    # ||w||_2 ~ 2236 per layer, far above FL_ZKP_MAX_NORM=100.
    return OrderedDict((k, np.full_like(v, 1000.0)) for k, v in _honest_weights().items())


class _Net:
    def __init__(self, weights):
        self._sd = OrderedDict((k, torch.from_numpy(v.copy())) for k, v in weights.items())

    def state_dict(self):
        return self._sd


def _fit_res(params, proofs, layer_names):
    metrics = {"zkp_proofs_json": json.dumps(proofs)}
    if layer_names is not None:
        metrics["zkp_layer_names_json"] = json.dumps(layer_names)
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), 10, metrics)


@pytest.fixture(autouse=True)
def _gnark_backend(monkeypatch, gnark_service_url):
    monkeypatch.setattr(zkp_gnark, "DEFAULT_SERVICE_URL", gnark_service_url)
    monkeypatch.delenv("FL_ZKP_BACKEND", raising=False)
    monkeypatch.delenv("FL_ZKP_LAYERS", raising=False)


@pytest.fixture
def config():
    return SimpleNamespace(zkp_backend="gnark", sim_mode=False)


# ─── Controls ────────────────────────────────────────────────────────────────


def test_control_poisoned_vector_cannot_be_proved(gnark_service_url):
    """The norm bound bites: the service refuses to prove the poisoned vector."""
    with pytest.raises(RuntimeError):
        generate_gnark_proofs(_poisoned_weights(), service_url=gnark_service_url)


def test_control_plaintext_zkp_rejects_mismatched_params_with_full_layer_names(config):
    """With complete, honest layer names, plaintext zkp binds proofs to params."""
    from fl.privacy.zkp import ZKPMode

    proofs, _ = generate_gnark_proofs(_honest_weights())
    honest = (SimpleNamespace(cid="honest"), _fit_res(list(_honest_weights().values()), proofs, list(LAYERS)))
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(list(_poisoned_weights().values()), proofs, list(LAYERS)))

    mode = ZKPMode()
    mode.aggregate_fit_override(1, [honest, attacker], [], None, config)
    admitted = [cp.cid for cp, _ in mode.pre_aggregate([honest, attacker], config)]

    assert admitted == ["honest"]


# ─── Attacks ─────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="audit/findings.md S1-04: client-supplied empty zkp_layer_names_json "
    "makes verify_gnark_proofs check nothing",
)
def test_plaintext_zkp_rejects_update_with_empty_layer_names(config):
    from fl.privacy.zkp import ZKPMode

    proofs, _ = generate_gnark_proofs(_honest_weights())
    honest = (SimpleNamespace(cid="honest"), _fit_res(list(_honest_weights().values()), proofs, list(LAYERS)))
    attacker = (SimpleNamespace(cid="attacker"), _fit_res(list(_poisoned_weights().values()), proofs, []))

    mode = ZKPMode()
    mode.aggregate_fit_override(1, [honest, attacker], [], None, config)
    admitted = [cp.cid for cp, _ in mode.pre_aggregate([honest, attacker], config)]

    assert "attacker" not in admitted, f"poisoned update admitted to FedAvg: {admitted}"


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
    reason="audit/findings.md S1-01: by design the CKKS composite's proof is not "
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
    he = mode._he_mode

    honest_proofs, _ = generate_gnark_proofs(_honest_weights())
    attacker_proofs, _ = generate_gnark_proofs(_honest_weights())  # proves vector A

    # Encrypt every layer, the strongest HE configuration (not the first-layer
    # default, audit/binding.md B-1).
    honest_ct = he._encrypt_params(_Net(_honest_weights()), secret_ctx, encrypt_layers=list(LAYERS))
    attacker_ct = he._encrypt_params(_Net(_poisoned_weights()), secret_ctx, encrypt_layers=list(LAYERS))  # submits vector B

    results = [
        (SimpleNamespace(cid="honest"), _fit_res(honest_ct, honest_proofs, list(LAYERS))),
        (SimpleNamespace(cid="attacker"), _fit_res(attacker_ct, attacker_proofs, list(LAYERS))),
    ]
    aggregated, _ = mode.aggregate_fit_override(1, results, [], server_ctx, config)

    admitted = mode._last_anchor_data["client_ids"]
    decrypted = _decrypt_aggregate(aggregated, secret_ctx)
    assert "attacker" not in admitted, (
        f"attacker admitted {admitted}; decrypted aggregate model.0.weight = "
        f"{np.round(decrypted[0], 3).tolist()} (honest mean would be "
        f"{_honest_weights()['model.0.weight'].tolist()})"
    )
    np.testing.assert_allclose(decrypted[0], _honest_weights()["model.0.weight"], atol=1e-3)
