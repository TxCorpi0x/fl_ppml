# Fail-closed audit: error paths in the ZKP flow

Date: 2026-09-14
Scope: Step 4 Phase 1. Every error path in the ZKP flow on client and server (`zkp`, `zkp_sampled`, `he_*_zkp`, `he_*_zkp_dp`, and the new `he_elgamal_zkp` for comparison), plus the same fail-open shape anywhere else in `fl/`.

**No source files were changed for this report.** Every behaviour marked **tested** was reproduced against the current code by [`audit/evidence/failmodes_evidence.py`](evidence/failmodes_evidence.py), which points the gnark service at a dead port (`http://127.0.0.1:1`). Its output is quoted below. Behaviour marked **code** is read from source and was not executed.

```bash
PYTHONPATH=. python audit/evidence/failmodes_evidence.py
```

## Hypothesis: confirmed

> Both ZKP aggregation paths fail open. Combined with `_generate_proofs` swallowing exceptions, an unreachable gnark service silently degrades the run to unverified FedAvg while still reporting itself as a ZKP mode.

| Evidence | Result |
|---|---|
| (a) `ZKPMode._generate_proofs` with the service unreachable | returns `num_proofs: 0` and logs `[ZKP] gnark proof generation failed …`; no exception |
| (a2) `ZKPMode.send_parameters` then `post_fit_metrics` | returns the plaintext parameters and `post_fit_metrics = {}`, so the client uploads normally with no proof fields |
| (b) plaintext `zkp` server, both clients' proofs fail to verify | `aggregate_fit_override → None`; `pre_aggregate` admits `['c1', 'c2']` to FedAvg; log: `All client proofs failed verification — falling back to unverified FedAvg` |
| (b2) plaintext `zkp` server, no client sent proofs | both admitted to FedAvg |
| (c) `he_tenseal_zkp` server, both clients' proofs fail | anchor `client_ids = ['c1', 'c2']`; log: `all clients excluded — falling back to full result set`; the ciphertexts go to HE aggregation |

Locations: `fl/privacy/zkp.py:184-194` (fallback), `fl/privacy/zkp.py:253-268` (swallowed generation errors), `fl/privacy/he_zkp.py:259-263` (fallback).

## Error-path map: legacy ZKP modes (`zkp`, `zkp_sampled`, `he_*_zkp`, `he_*_zkp_dp`)

| # | Event | Where | What happens now | Evidence | What should happen |
|---|---|---|---|---|---|
| 1 | Proof generation fails: service down, timeout, 5xx, unsatisfied statement, or RSS guard | client, `zkp.py:253-268` | Exception caught; `([], 0)` returned; client uploads parameters with no proof metrics and reports a successful fit | tested (a, a2) | Client `fit` raises, so Flower records a failure and the client doesn't take part in the round |
| 2 | Verification service unreachable, times out, or returns a non-2xx status | server, `zkp_gnark.py:414-423, 437-447, 505-522` | Counted as a verification failure for that client → client excluded → if every client is excluded, row 6 applies | tested (b, c) | **Infrastructure failure: abort the round** (no update, no ModelCommit, no ProofAnchor). Log it as an infrastructure fault, not a client rejection |
| 3a | `/verify_light` returns 2xx with a body that isn't JSON | server, `zkp_gnark.py:524` (`response.json()` outside the `try`) | `ValueError` escapes `aggregate_fit` | code | Infrastructure failure: abort the round |
| 3b | Malformed proof payload, plaintext path: missing `scale`, `bound_sq` or `proof_b64` | server, `zkp_gnark.py:424-427` | **`KeyError: 'scale'` escapes `aggregate_fit_override`.** One malicious client's metadata aborts server aggregation for everyone | tested (f) | Reject that client; never raise |
| 3c | Malformed chunk payload: chunk `shape` larger than the layer | server, `zkp_gnark.py:392-395` | **`ValueError: cannot reshape array of size 2 into shape (5,)` escapes** | tested (f2) | Reject that client |
| 3d | Malformed `zkp_proofs_json` (not JSON, or empty) | server, `zkp.py:143-152`, `he_zkp.py:207-213` | Client excluded | code | Correct as is; keep |
| 4a | Client sends `zkp_layer_names_json = "[]"` | server, `zkp.py:159-175` → `verify_gnark_proofs` | Loop runs zero times → `(True, [])` → **poisoned update admitted** | tested: `tests/test_zkp_binding_attack.py::test_plaintext_zkp_rejects_update_with_empty_layer_names` (strict xfail) | Server-owned schema: reject any upload whose layers or proof coverage don't match the server model |
| 4b | Layer-name list shorter than the parameter list, or a chunked layer missing later chunks | server, `zkp_gnark.py:379` (`zip`), `381-417` | Tensors past the list, or elements past the provided chunks, are aggregated without verification | code (findings.md S1-04, binding.md B-4) | As 4a |
| 4c | Malformed `zkp_layer_names_json` | server, `zkp.py:160-167` | Replaced with index names `"0", "1", …` → proofs don't match → client excluded | code | Reject explicitly; don't guess names |
| 4d | Composite: proofs cover only some layers | server, `he_zkp.py:215-238`, `zkp_gnark.py:485-526` | Any non-empty set of valid proofs admits the whole update | code (binding.md B-6) | As 4a |
| 5 | Some clients fail (excluded, or Flower `failures`) | server, `fl/server.py:148-225` | Remaining clients aggregated; **`failures` is never inspected**; there's no minimum number of admitted clients | code | Configurable quorum (default: at least `min_fit_clients` admitted), otherwise no update. Exclusions and failures reported per round |
| 6 | Every client excluded | server, `zkp.py:184-194`, `he_zkp.py:259-263` | **Full unverified result set aggregated** | tested (b, b2, c) | No aggregation, central model unchanged, no ModelCommit, no ProofAnchor, round recorded as aborted |
| 7 | ZKP backend `pedersen` | server `zkp.py:128-134`; composite `he_zkp.py:211-213, 239-247`; **default of `--zkp_backend` in `main_server.py:60`, `main_client.py:58`, `simulation.py:65`** | Server does no verification and admits every client, while the mode still runs as `zkp`. The harness passes `--zkp_backend gnark`, but anyone running the entry points directly gets the stub silently | tested (d: `pedersen_override_return: None`, `admitted: ['x']`) | Default `gnark`; the stub needs an explicit flag and is labelled as providing no verification |
| 8 | Anchor data can't be built | server, `zkp.py:206-218`, `he_zkp.py:268-280` | Exception printed; round continues with partial or empty anchor data | code | Abort the round: an unanchored "verified" round isn't auditable |
| 9 | Ledger save fails | server, `fl/server.py:406-411` | `logger.warning` only; audit trail silently incomplete | code | Abort the run. `validate_zkp_ledger` already fails closed on a missing ledger for ZKP-family modes |
| 10 | Harness can't start the gnark service (`go` missing, or no health check within 30 s) | harness, `fl/compare/experiment.py:165-170, 213-217` | `skipping gnark service start` / `continuing anyway`; ZKP modes then run into rows 1, 2 and 6 | code | Refuse to run ZKP-family modes and record them as failed |

## Does a fail-open round still write chain anchor data?

**Yes. This is a separate finding: F-1, S1.**

| Case | Anchor written for a round in which nothing verified | Evidence |
|---|---|---|
| Plaintext `zkp`, every proof fails verification | `ProofAnchor` with **4 proof hashes and 2 client ids**, plus a `ModelCommit` of the unverified FedAvg model | tested (b): `anchor: {'round': 1, 'proof_hashes': 4, 'client_ids': 2}`; written by `FedPrivate._chain_commit` (`fl/server.py:340-404`) |
| Plaintext `zkp`, no client sent proofs | `ProofAnchor` with no hashes, plus a `ModelCommit` of the unverified model | tested (b2) |
| `he_tenseal_zkp`, every proof fails | `ProofAnchor` naming both clients with 4 proof hashes | tested (c) |

The ledger is what makes a verified round auditable after the fact, and here it records the round as if the clients' proofs had been accepted. Anyone reading only the ledger can't tell.

## The same shape elsewhere in `fl/`

I searched for the shape, not just the two named lines: fallback/re-admission wording, `list(results)`, broad `except` blocks that continue, `continuing anyway`, and overrides that return `None` to fall through to FedAvg. Results:

| ID | Severity | Location | Pattern | What happens | Evidence |
|---|---|---|---|---|---|
| E-1 | S1 | `fl/privacy/he_tenseal.py:167-182, 296-297` | Server public key file missing → `setup_server_context` returns `None` ("running in simulation mode") even in real mode → `aggregate_fit_override` returns `None` → plain FedAvg over ciphertext bytes | **FedAvg averaged uint8 CKKS ciphertext bytes into float64 "weights"** (e.g. first values `[4.8, 2.4, 10.4]` and, in a second run, `[4.8, 2.4, 7.2]` for real weights `[0.1, -0.2]`; the values change with the ciphertext randomness). The global model is silently corrupted and the round reports success | tested (e) |
| E-2 | S1 | `fl/privacy/he_tenseal.py:256-267` | Client-side decryption failure caught | `receive_parameters failed …; keeping local model weights`: the client trains on its stale local model and uploads, and the round reports success | code |
| E-3 | S1 | `fl/privacy/he_concrete_tfhe.py:234-240` (binding.md B-3); `fl/privacy/he_tenseal.py:490-495` | TFHE encrypted-aggregation exception → FedAvg; `_decompress_cte2_results` swallows parse errors and passes uint8 through | Same FedAvg-over-bytes mechanism shown in E-1 | code (mechanism tested in E-1) |
| E-4 | S1 (privacy) | `fl/privacy/dp.py:62-69` | DP params file missing → config defaults | **ε silently becomes 10.0** (σ = 0.4845), the sentinel meaning "load from file". The harness checks for the file first; direct `main_client.py` / `simulation.py` runs don't | tested (g) |
| E-5 | S1 | `fl/privacy/he_zkp.py:211-213, 239-247` | Non-gnark backend: clients with or without proofs admitted without checks | Same as row 7 for composites | code |
| E-6 | S2 (evidence) | `fl/compare/runner.py:183-191` | Modes whose prerequisites are missing are skipped with a warning, and the run succeeds | Requested modes silently missing from `comparison_report.json`. This happened in the first `he_elgamal_zkp` smoke run (binding.md) | observed in `results/healthcare/20260914_202700` |
| E-7 | S2 | `fl/server.py:193-203, 340-404` | When a mode returns no aggregate, `_chain_commit` still runs | **A `ModelCommit` is written for a round with no model update.** Affects `he_elgamal_zkp`'s fail-closed rounds (no admitted client, infrastructure abort) | tested (h): `[('ModelCommit', 3, 1)]` |

`he_elgamal_zkp`, for comparison: rows 1–6 and 8 already fail closed. The client's prove call raises `ElGamalRejected` or `ElGamalServiceError`. The server separates rejections (bad input, failed verification, coverage mismatch) from infrastructure aborts, enforces the server-owned schema, and on no-admission or abort returns `(None, metrics)` with no anchor (`tests/test_he_elgamal_zkp.py`). What remains for it is E-7, rows 5 (quorum), 9 and 10, and making aborts visible in `comparison_report.json`.

Not tested here: what Flower 1.8 does when a client's `fit` raises (reported as a `failure` vs a dropped connection waiting on a round with no timeout). Phase 2 should cover this with a distributed test.

## Proposed Phase 2

1. **ZKP client (row 1):** proof generation failure raises; no silent `([], 0)`.
2. **Server verification outcome.** Each client is classified as:
   - **admitted**, or
   - **rejected** (missing, malformed or invalid proofs; schema or coverage mismatch; never raises), or
   - **infrastructure failure** (unreachable service, timeout, 5xx, non-JSON), which **aborts the round**.

   This covers rows 2, 3a–d and 4a–d. Legacy modes need a server-owned schema, using the `bind_server_model` hook `he_elgamal_zkp` already uses.
3. **No fallback (rows 6 and 7).** Delete both `list(results)` re-admissions. With no admitted client, or below quorum, the override returns `(None, metrics)`. The plaintext `zkp` mode aggregates its verified subset itself instead of relying on `pre_aggregate` side effects. `--zkp_backend` defaults to `gnark` in all three entry points.
4. **Chain (F-1, E-7, rows 8 and 9).** `_chain_commit` writes nothing for a round without an aggregate. ProofAnchor contains only admitted clients' proofs. Failing to build the anchor or save the ledger aborts.
5. **Visible outcomes.** Record a per-round outcome in the server benchmark, `{round, outcome: aggregated | no_quorum | infrastructure_abort, admitted, rejected: {cid: reason}, flower_failures}`. It is then merged into `comparison_report.json`, and `validate_zkp_ledger` also checks that every recorded `aggregated` round has a matching anchor. This replaces the current "verification outcomes are not persisted" caveat.
6. **Harness (rows 10 and E-6).** A gnark start failure fails ZKP-family modes. Skipped modes appear in the report as failed results with the reason.
7. **Tests** for every row above that changes, plus E-1–E-5 if approved (next paragraph).

**Scope question for approval (CLAUDE.md rule 5).** E-1–E-5 are fail-open paths outside the ZKP flow (TenSEAL and TFHE aggregation and decryption, DP parameters). The Step 4 prompt asks me to report the pattern wherever it appears, but fixing it is outside the named area. I recommend including E-1, E-2, E-3 and E-5 in Phase 2, because each one silently corrupts or skips a security guarantee inside a round. For E-4, missing DP params should raise unless `dp_epsilon` is passed explicitly.

**Stopping here, as Step 4 Phase 1 requires.**
