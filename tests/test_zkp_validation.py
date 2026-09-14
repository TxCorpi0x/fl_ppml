"""Tests for fl.compare.validation (ZKP run self-certification)."""

from fl.compare.validation import sampled_coverage_warning, validate_zkp_ledger


def _round(rnd, clients=3, proofs_per_client=7, anchored=None):
    anchored = clients if anchored is None else anchored
    hashes = [f"0x{rnd}{i}" for i in range(anchored * proofs_per_client)]
    return [
        {"type": "ModelCommit", "round": rnd, "num_clients": clients},
        {
            "type": "ProofAnchor",
            "round": rnd,
            "proof_hashes": hashes,
            "client_ids": [str(c) for c in range(anchored)],
            "num_proofs": len(hashes),
        },
    ]


def _ledger(rounds=3, **kwargs):
    return [e for r in range(1, rounds + 1) for e in _round(r, **kwargs)]


def test_complete_ledger_passes():
    report = validate_zkp_ledger(_ledger(), expected_rounds=3)
    assert report["ok"], report["errors"]
    assert report["proofs_per_client"] == [7]
    assert report["rounds_checked"] == 3


def test_partial_participation_round_passes_when_all_aggregated_clients_proved():
    entries = _round(1, clients=2) + _round(2)
    assert validate_zkp_ledger(entries, expected_rounds=2)["ok"]


def test_missing_ledger_fails_closed():
    for entries in (None, []):
        report = validate_zkp_ledger(entries, expected_rounds=3)
        assert not report["ok"]
        assert "cannot be certified" in report["errors"][0]


def test_missing_round_fails():
    report = validate_zkp_ledger(_ledger(rounds=2), expected_rounds=3)
    assert not report["ok"]
    assert any("round 3" in e for e in report["errors"])


def test_aggregated_client_without_proofs_fails():
    entries = _round(1) + _round(2, clients=3, anchored=2)
    report = validate_zkp_ledger(entries)
    assert not report["ok"]
    assert any("3 client(s) aggregated but 2 emitted proofs" in e for e in report["errors"])


def test_empty_anchor_fails():
    entries = _round(1, proofs_per_client=0)
    assert not validate_zkp_ledger(entries)["ok"]


def test_inconsistent_proof_counts_fail():
    entries = _round(1, proofs_per_client=7) + _round(2, proofs_per_client=2)
    report = validate_zkp_ledger(entries)
    assert not report["ok"]
    assert any("inconsistent proofs per client" in e for e in report["errors"])


def test_num_proofs_mismatch_fails():
    entries = _round(1)
    entries[1]["num_proofs"] = 99
    assert not validate_zkp_ledger(entries)["ok"]


def test_verification_outcome_is_reported_as_unchecked():
    warnings = validate_zkp_ledger(_ledger())["warnings"]
    assert any("not persisted" in w for w in warnings)


def test_sampled_coverage_identical_to_full_is_flagged():
    full = validate_zkp_ledger(_ledger(proofs_per_client=7))
    sampled = validate_zkp_ledger(_ledger(proofs_per_client=7))
    assert "zkp_sampled_not_sampling" in sampled_coverage_warning(full, sampled)
    reduced = validate_zkp_ledger(_ledger(proofs_per_client=2))
    assert sampled_coverage_warning(full, reduced) is None
