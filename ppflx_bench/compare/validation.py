"""
ZKP run validation — makes ZKP-family benchmark runs self-certifying.

A ZKP-family run is only usable as evidence if every client that contributed
to every round actually emitted proofs. The persisted record of that is the
per-mode chain ledger: the server writes one ``ModelCommit`` (with one hash per
aggregated client) and one ``ProofAnchor`` (with the proof hashes and the ids
of clients whose proofs were anchored) per round.

What this validates
-------------------
* a ledger exists and is non-empty (no ledger → cannot certify → failure)
* every expected round has both a ModelCommit and a ProofAnchor
* every client aggregated in a round appears in that round's ProofAnchor
* each anchored client contributed at least one proof, the proof count per
  client is the same across clients and rounds, and ``num_proofs`` matches
  the stored hashes

What the ledger does NOT show
-----------------------------
Whether the proofs *verified*. The ledger records what was anchored, not the
verifier's decisions. Those are in the benchmark's ``round_outcomes``, which
``validate_run`` checks: every round must be aggregated (or committed, in
commit–challenge modes) with no rejected clients. Runs without recorded
outcomes can only be validated against the ledger and carry a warning.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Internal mode names (ppflx_bench.compare.registry.ModeConfig.internal_mode) that
# generate and verify gnark proofs.
ZKP_INTERNAL_MODES = frozenset({"zkp", "he_zkp", "he_zkp_dp"})

VERIFICATION_NOT_RECORDED = (
    "no recorded round outcomes; the ledger alone does not show whether the "
    "anchored proofs verified"
)


def load_ledger_entries(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Return ledger entries from a MockChain JSON file, or None if unavailable."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "ledger" in raw:
        raw = raw["ledger"]
    return raw if isinstance(raw, list) else None


def validate_zkp_ledger(
    entries: Optional[List[Dict[str, Any]]],
    expected_rounds: Optional[int] = None,
    required_rounds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Check that every aggregated client emitted proofs in every round.

    Returns a JSON-serialisable report with ``ok``, ``errors``, ``warnings``,
    ``rounds_checked`` and ``proofs_per_client`` (sorted distinct values).
    """
    errors: List[str] = []
    warnings: List[str] = [VERIFICATION_NOT_RECORDED]

    if not entries:
        return {
            "ok": False,
            "errors": ["no ledger entries: run cannot be certified"],
            "warnings": warnings,
            "rounds_checked": 0,
            "proofs_per_client": [],
        }

    commits: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    anchors: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        kind = entry.get("type")
        if kind == "ModelCommit":
            commits[entry.get("round")].append(entry)
        elif kind == "ProofAnchor":
            anchors[entry.get("round")].append(entry)

    rounds = set(commits) | set(anchors)
    if expected_rounds:
        rounds |= set(range(1, expected_rounds + 1))
    if required_rounds:
        rounds |= set(required_rounds)

    proofs_per_client = set()
    for rnd in sorted(r for r in rounds if r is not None):
        if len(commits[rnd]) != 1:
            errors.append(f"round {rnd}: expected 1 ModelCommit, found {len(commits[rnd])}")
        if len(anchors[rnd]) != 1:
            errors.append(f"round {rnd}: expected 1 ProofAnchor, found {len(anchors[rnd])}")
        if len(commits[rnd]) != 1 or len(anchors[rnd]) != 1:
            continue

        commit, anchor = commits[rnd][0], anchors[rnd][0]
        aggregated = int(commit.get("num_clients") or 0)
        client_ids = [str(c) for c in anchor.get("client_ids") or []]
        proof_hashes = anchor.get("proof_hashes") or []

        if aggregated == 0:
            errors.append(f"round {rnd}: no clients aggregated")
            continue
        if len(set(client_ids)) != len(client_ids):
            errors.append(f"round {rnd}: duplicate client ids in ProofAnchor")
        if len(set(client_ids)) != aggregated:
            errors.append(
                f"round {rnd}: {aggregated} client(s) aggregated but "
                f"{len(set(client_ids))} emitted proofs"
            )
        if anchor.get("num_proofs") != len(proof_hashes):
            errors.append(
                f"round {rnd}: num_proofs={anchor.get('num_proofs')} does not "
                f"match {len(proof_hashes)} stored proof hashes"
            )
        if not proof_hashes:
            errors.append(f"round {rnd}: ProofAnchor has no proofs")
        elif client_ids:
            if len(proof_hashes) % len(client_ids):
                errors.append(
                    f"round {rnd}: {len(proof_hashes)} proofs not evenly divisible "
                    f"across {len(client_ids)} clients"
                )
            else:
                proofs_per_client.add(len(proof_hashes) // len(client_ids))

    if len(proofs_per_client) > 1:
        errors.append(
            f"inconsistent proofs per client across rounds: {sorted(proofs_per_client)}"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rounds_checked": len([r for r in rounds if r is not None]),
        "proofs_per_client": sorted(proofs_per_client),
    }


def validate_run(
    entries: Optional[List[Dict[str, Any]]],
    round_outcomes: Optional[List[Dict[str, Any]]],
    expected_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """Ledger validation plus the server's recorded per-round outcomes.

    When outcomes are recorded, every expected round must have exactly one,
    and it must be ``aggregated`` with no rejected clients and no Flower
    failures. Commit–challenge modes record two Flower rounds per federated
    round: odd rounds must be ``committed`` and even rounds ``aggregated``,
    and only even rounds carry ledger entries. A run with aborted or
    partially rejected rounds is not valid benchmark evidence. Without
    recorded outcomes (runs from before they existed), the ledger-only report
    and its warning are returned unchanged.
    """
    if round_outcomes is None:
        return validate_zkp_ledger(entries, expected_rounds)

    two_phase = any(o.get("outcome") == "committed" for o in round_outcomes)
    flower_rounds = (expected_rounds or 0) * (2 if two_phase else 1)
    if two_phase:
        report = validate_zkp_ledger(entries, required_rounds=list(range(2, flower_rounds + 1, 2)))
    else:
        report = validate_zkp_ledger(entries, expected_rounds)

    errors = list(report["errors"])
    by_round: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for outcome in round_outcomes:
        by_round[outcome.get("round")].append(outcome)
    for rnd in range(1, flower_rounds + 1):
        if rnd not in by_round:
            errors.append(f"round {rnd}: no recorded outcome")
    for rnd, outs in sorted(by_round.items(), key=lambda kv: str(kv[0])):
        if len(outs) != 1:
            errors.append(f"round {rnd}: {len(outs)} recorded outcomes")
            continue
        out = outs[0]
        expected = "committed" if two_phase and isinstance(rnd, int) and rnd % 2 == 1 else "aggregated"
        if out.get("outcome") != expected:
            detail = out.get("detail") or out.get("rejected") or ""
            errors.append(f"round {rnd}: outcome {out.get('outcome')} {detail}".rstrip())
        elif out.get("rejected"):
            errors.append(f"round {rnd}: clients rejected {out['rejected']}")
        if out.get("flower_failures"):
            errors.append(f"round {rnd}: {out['flower_failures']} Flower client failure(s)")

    warnings = [w for w in report["warnings"] if w != VERIFICATION_NOT_RECORDED]
    return {**report, "ok": not errors, "errors": errors, "warnings": warnings}


def sampled_coverage_warning(
    zkp_report: Optional[Dict[str, Any]], sampled_report: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Flag a zkp_sampled run that proved exactly as much as full zkp."""
    if not zkp_report or not sampled_report:
        return None
    full, sampled = zkp_report.get("proofs_per_client"), sampled_report.get("proofs_per_client")
    if full and sampled and full == sampled:
        return (
            f"zkp_sampled_not_sampling: zkp_sampled emitted {sampled[0]} proof(s) per "
            f"client per round, identical to full zkp coverage"
        )
    return None
