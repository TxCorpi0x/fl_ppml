"""Round trips against real, separate gnark prover and verifier processes.

Skipped when zkp_gnark_service/gnark_service has not been built. The session
uses small pinned test keys (tests/conftest.py), never the committed ones.
"""

import numpy as np
import pytest

from ppflx.core.zkp_gnark import (
    GnarkServiceError,
    generate_gnark_proofs,
    verify_gnark_proofs,
    verify_gnark_proofs_light,
)
from tests.lib.conftest import TEST_NORM_N, gnark_setup, requires_gnark, start_gnark

pytestmark = requires_gnark

BN254_R = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001


@pytest.fixture
def state_dict():
    return {
        "w": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
        "b": np.array([0.05, -0.05], dtype=np.float32),
    }


def test_generate_emits_one_complete_pinned_proof_per_layer(gnark, state_dict):
    from ppflx.core.gnark_keys import NORM_CIRCUIT, pinned_vk_sha256

    proofs, proof_bytes = generate_gnark_proofs(state_dict, service_url=gnark.prover)

    assert sorted(p["layer"] for p in proofs) == ["b", "w"]
    assert proof_bytes > 0
    for p in proofs:
        assert p["shape"] == list(state_dict[p["layer"]].shape)
        assert p["proof_b64"] and p["hash_hex"]
        assert p["vk_sha256"] == pinned_vk_sha256(NORM_CIRCUIT)


def test_proofs_from_the_prover_verify_on_the_separate_verifier(gnark, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)

    ok, failures = verify_gnark_proofs(
        list(state_dict.values()), list(state_dict), proofs, service_url=gnark.verifier
    )
    assert ok, failures
    ok, failures = verify_gnark_proofs_light(proofs, service_url=gnark.verifier)
    assert ok, failures


def test_proofs_still_verify_after_the_verifier_restarts(gnark, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)
    restarted, url = start_gnark("verifier", gnark.keys_dir)
    try:
        assert verify_gnark_proofs_light(proofs, service_url=url)[0]
    finally:
        restarted.terminate()
        restarted.wait(timeout=10)


def test_verifier_with_different_keys_refuses_loudly(gnark, state_dict, tmp_path):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)
    gnark_setup(tmp_path / "keys", tmp_path / "pk")
    other, url = start_gnark("verifier", tmp_path / "keys")
    try:
        with pytest.raises(GnarkServiceError, match="HTTP 503"):
            verify_gnark_proofs_light(proofs, service_url=url)
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_roles_do_not_serve_each_others_endpoints(gnark, state_dict):
    import requests

    assert requests.post(f"{gnark.verifier}/prove", json={}, timeout=5).status_code == 404
    assert requests.post(f"{gnark.prover}/verify", json={}, timeout=5).status_code == 404
    health = requests.get(f"{gnark.verifier}/health", timeout=5).json()
    assert health["role"] == "verifier" and health["circuits"]["norm"]["n"] == TEST_NORM_N


def test_full_verification_rejects_tampered_parameters(gnark, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)
    tampered = [state_dict["w"] + 1.0, state_dict["b"]]

    ok, failures = verify_gnark_proofs(tampered, list(state_dict), proofs, service_url=gnark.verifier)
    assert not ok
    assert failures == ["w"]


def test_light_verification_rejects_a_forged_hash(gnark, state_dict):
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)

    def _flip_last_digit(hex_str):
        return hex_str[:-1] + ("0" if hex_str[-1] != "0" else "1")

    forged = [dict(p, hash_hex=_flip_last_digit(p["hash_hex"])) for p in proofs]

    ok, failures = verify_gnark_proofs_light(forged, service_url=gnark.verifier)
    assert not ok
    assert sorted(failures) == ["b", "w"]


def test_light_verification_rejects_non_canonical_hash(gnark, state_dict):
    """The verifier rejects non-canonical hashes: hashes ≥ r are rejected before the service reduces them."""
    proofs, _ = generate_gnark_proofs(state_dict, service_url=gnark.prover)
    aliased = [dict(p, hash_hex=format(int(p["hash_hex"], 16) + BN254_R, "x")) for p in proofs]

    ok, _ = verify_gnark_proofs_light(aliased, service_url=gnark.verifier)
    assert not ok
