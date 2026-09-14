"""Round-trip tests against the real gnark service binary.

Skipped when zkp_gnark_service/gnark_service has not been built. Tensors are
kept tiny so circuit compilation and Groth16 setup take seconds.
"""

import numpy as np
import pytest

from fl.core.zkp_gnark import (
    generate_gnark_proofs,
    verify_gnark_proofs,
    verify_gnark_proofs_light,
)
from tests.conftest import requires_gnark

pytestmark = requires_gnark

BN254_R = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001


@pytest.fixture
def state_dict():
    return {
        "w": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
        "b": np.array([0.05, -0.05], dtype=np.float32),
    }


def test_generate_emits_one_complete_proof_per_layer(gnark_service_url, state_dict):
    proofs, proof_bytes = generate_gnark_proofs(state_dict, service_url=gnark_service_url)

    assert sorted(p["layer"] for p in proofs) == ["b", "w"]
    assert proof_bytes > 0
    for p in proofs:
        assert p["shape"] == list(state_dict[p["layer"]].shape)
        assert p["proof_b64"] and p["hash_hex"]


def test_proofs_verify_against_the_proved_parameters(gnark_service_url, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark_service_url)

    ok, failures = verify_gnark_proofs(
        list(state_dict.values()), list(state_dict), proofs, service_url=gnark_service_url
    )
    assert ok, failures
    ok, failures = verify_gnark_proofs_light(proofs, service_url=gnark_service_url)
    assert ok, failures


def test_full_verification_rejects_tampered_parameters(gnark_service_url, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark_service_url)
    tampered = [state_dict["w"] + 1.0, state_dict["b"]]

    ok, failures = verify_gnark_proofs(
        tampered, list(state_dict), proofs, service_url=gnark_service_url
    )
    assert not ok
    assert failures == ["w"]


def test_light_verification_rejects_a_forged_hash(gnark_service_url, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark_service_url)

    def _flip_last_digit(hex_str):
        return hex_str[:-1] + ("0" if hex_str[-1] != "0" else "1")

    forged = [dict(p, hash_hex=_flip_last_digit(p["hash_hex"])) for p in proofs]

    ok, failures = verify_gnark_proofs_light(forged, service_url=gnark_service_url)
    assert not ok
    assert sorted(failures) == ["b", "w"]


def test_light_verification_rejects_non_canonical_hash(gnark_service_url, state_dict):
    """Fixed on the verifier side (audit/binding.md B-5): hashes ≥ r are rejected before the service reduces them."""
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark_service_url)
    aliased = [dict(p, hash_hex=format(int(p["hash_hex"], 16) + BN254_R, "x")) for p in proofs]

    ok, _ = verify_gnark_proofs_light(aliased, service_url=gnark_service_url)
    assert not ok
