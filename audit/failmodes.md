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

---

# Phase 2 — fail closed (implemented)

Approved scope: the proposed plan plus E-1 through E-5.

## Changes by error path

| Row / ID | Before | Now | Location | Pinned by |
|---|---|---|---|---|
| 1 client proof failure | swallowed; upload without proofs | raises; no upload. An empty proof set also raises | `fl/privacy/zkp.py::_generate_proofs` | `test_client_proof_generation_failure_raises` |
| 2, 3a service unreachable, 5xx, non-JSON | counted as client failure → fallback; non-JSON crashed | `GnarkServiceError` → **round aborts** (`infrastructure_abort`): no update, no commit, no anchor | `fl/core/zkp_gnark.py::_post_service`, both aggregation paths | `test_verification_service_outage_aborts_round`, `test_composite_service_outage_aborts_round` |
| 3b, 3c malformed proof or chunk payload | `KeyError` / `ValueError` escaped the strategy | counted as failed verification → client rejected | `verify_gnark_proofs`, `verify_gnark_proofs_light` | `test_malformed_payload_counts_as_verification_failure`, `test_bad_uploads_are_rejected_without_raising[missing_scale]` |
| 4a–d layer names, partial coverage, composite coverage | client-supplied names; `"[]"` verified nothing; partial chunks and partial layers admitted | server-owned schema from `bind_server_model`; exact proof coverage, per-proof shape, scale and bound checked against server policy (`check_proof_policy`); upload shapes must match; chunked layers must be fully covered | `zkp_gnark.check_proof_policy`, `expected_proof_layout`, `ZKPMode._check_client`, composite admission | `test_plaintext_zkp_rejects_update_with_empty_layer_names` (**xfail removed; now passes**), `test_bad_uploads_are_rejected_without_raising[drop_layer_proof, off_policy_bound, no_proofs, wrong_shape_upload]` |
| 5 partial failure / quorum | no quorum; `failures` ignored | at least `min_fit_clients` admitted, otherwise `no_quorum`; Flower failures recorded per round | `admission_quorum`, `FedPrivate._record_round` | `test_quorum_below_min_fit_clients_leaves_model_unchanged`, `test_no_model_commit_or_anchor_for_a_round_without_update` |
| 6 all excluded | full unverified set aggregated | **no fallback**: plaintext `zkp` FedAvgs only the admitted subset itself; composites pass only admitted clients to HE aggregation, including the simulation path | `ZKPMode.aggregate_fit_override`, `_HeZKPCompositeMode._aggregate` | `test_all_clients_rejected_leaves_model_unchanged`, `test_composite_never_aggregates_rejected_clients` |
| 7, E-5 pedersen stub | admitted everyone; CLI default | refused unless `FL_ZKP_ALLOW_PEDERSEN_STUB=1`, then recorded as `unverified_stub`; `--zkp_backend` defaults to `gnark` in `main_server.py`, `main_client.py`, `simulation.py` and the registry check | `zkp.resolve_backend` | `test_pedersen_stub_is_refused_unless_explicitly_allowed` |
| 8 anchor build error | printed, round continued | propagates out of the strategy (the run fails) | `zkp.anchor_data` | code |
| 9 ledger save error | warning | raises | `FedPrivate._chain_commit` | `test_ledger_save_failure_raises` |
| 10 gnark start failure | "continuing anyway" | ZKP mode not run; result marked failed | `experiment._ensure_gnark_service`, `run_experiment` | `test_zkp_mode_is_not_run_without_a_healthy_proof_service` |
| F-1, E-7 ledger for non-updating rounds | ModelCommit (and ProofAnchor) written | nothing written without an aggregate; ModelCommit hashes only admitted clients; anchors contain only admitted proofs | `FedPrivate._chain_commit` | `test_no_model_commit_or_anchor_for_a_round_without_update`, `test_model_commit_hashes_only_admitted_clients` |
| B-5 non-canonical hash | `hash + r` verified | hashes and bounds outside [0, r) rejected before the service reduces them | `verify_gnark_proofs_light`, `check_proof_policy` | `test_light_verification_rejects_non_canonical_hash` (**xfail removed; now passes**) |
| E-1 TenSEAL without server context | FedAvg over ciphertext bytes | missing public key raises in real mode; aggregation without a context raises | `he_tenseal.py::setup_server_context`, `aggregate_fit_override` | `test_tenseal_server_without_public_key_refuses_real_mode`, `test_tenseal_aggregation_without_context_raises` |
| E-2 TenSEAL decryption failure | stale local weights kept | raises | `he_tenseal.py::receive_parameters` | `test_tenseal_decryption_failure_raises` |
| E-3 TFHE aggregation failure; undecodable uint8 | plaintext FedAvg fallback; bytes passed through | raises; `_decompress_cte2_results` refuses undecodable uint8 payloads | `he_concrete_tfhe.py`, `he_tenseal.py::_decompress_cte2_results` | `test_tfhe_aggregation_without_context_or_on_error_raises`, `test_undecodable_uint8_payload_is_never_averaged` |
| E-4 DP without params | ε silently 10 | raises unless `dp_epsilon` is passed explicitly; then σ is computed from that ε | `fl/privacy/dp.py` | `test_dp_without_params_file_requires_explicit_epsilon`; `tests/test_fl_package.py::test_dp_context_without_params_file_requires_explicit_epsilon` (replaces a test that asserted the fail-open fallback) |
| E-6 skipped modes | silently absent | recorded as failed results with the reason; the run raises; failed or skipped results never overwrite stored entries in the dataset-level report | `fl/compare/runner.py` | `test_skipped_modes_are_reported_and_fail_the_run`, `test_failed_results_never_replace_stored_dataset_entries` |
| Visible outcomes | verification outcome not persisted | `benchmark.round_outcomes` per round (`round`, `outcome`, `admitted`, `rejected: {cid: reason}`, `flower_failures`), merged into `comparison_report.json`; `validate_run` fails ZKP runs with any non-aggregated round, rejection or Flower failure | `FedPrivate._record_round`, `BenchmarkMetrics.round_outcomes`, `fl/compare/validation.py::validate_run` | `tests/test_zkp_validation.py` (3 new tests) |

`he_elgamal_zkp` gained the same quorum and `last_round_report`, so its rounds appear in `round_outcomes` too.

## Evidence

**Test suite (`flEnv`):** `84 passed, 1 xfailed, 1 failed`.
- The xfail is the CKKS composite ciphertext-binding attack, still vulnerable by design (S1-01).
- The failure is the pre-existing `test_zkp_sampled.py::test_pct_sampling_env` (H2, Step 5).
- Two former strict xfails now pass and their markers are removed: the empty-layer-names attack (S1-04) and the non-canonical hash (B-5).

**`PYTHONPATH=. python audit/evidence/failmodes_evidence.py` on current code:**

```
[a  ZKP client, service unreachable]            -> RAISED RuntimeError: gnark proof generation failed …
[b  zkp server, all proofs fail]                -> override_return (None, {'round_outcome': 'infrastructure_abort'}), admitted [], anchor None
[b2 zkp server, no client sent proofs]          -> (None, {'round_outcome': 'no_quorum', 'admitted': 0, 'rejected': 2}), anchor None
[c  he_tenseal_zkp server, all proofs fail]     -> (None, {'round_outcome': 'infrastructure_abort'}), anchor None
[d  zkp with pedersen backend]                  -> RAISED RuntimeError: ZKP backend 'pedersen' performs no verification …
[e  he_tenseal server, no context, non-sim]     -> RAISED RuntimeError: no server context in non-simulation mode …
[f  verify_gnark_proofs, missing fields]        -> (False, ['w', 'b'])
[f2 verify_gnark_proofs, oversized chunk shape] -> (False, ['w__chunk_0', 'b'])
[g  dp, params file missing]                    -> RAISED FileNotFoundError: DP params not found …
[h  _chain_commit after a no-update round]      -> []
```

Every row that failed open in Phase 1 now either rejects, aborts with no ledger entry, or refuses to run.

**End-to-end run A** (distributed, healthcare, 2 rounds, 2 clients, 1 epoch, output outside `results/`; `compare.py` exit 0). For all three modes, `zkp`, `he_tenseal_zkp` and `he_elgamal_zkp`:
- `round_outcomes` records `aggregated` in both rounds, with both clients admitted, no rejections, and `flower_failures: 0`.
- The ledger has exactly one ModelCommit and one ProofAnchor per round, and each ModelCommit counts the 2 admitted clients.
- `zkp_validation.ok` is true with **no warnings**. The "verification outcomes are not persisted" caveat no longer applies to new runs.

**End-to-end run B** (a `zkp` run, 1 round, 2 clients, with `FL_ZKP_MAX_NORM=0.000001`, so every client's proving fails):
- **The client fails closed:** both clients raised `gnark proof generation failed for layer 'model.0.weight' … 500 Server Error` (the unsatisfied statement) and exited. Neither uploaded an unproven update.
- **Flower 1.8 reports a client whose `fit` raises as a failure:** the server logged `aggregate_fit: received 0 results and 2 failures`. No update was applied. This answers the open question from Phase 1.
- **New finding F-2 (S2, availability and reporting): after every client fails, the server doesn't finish.**
  - It stayed blocked in round-1 evaluation, waiting to sample clients that had exited, from 21:07 until the process was stopped by hand at 22:11. `fl.server.start_server` runs with `ServerConfig(num_rounds=…)` and no `round_timeout`.
  - The harness waits for the server up to `server_timeout = client_timeout + 1800` (`fl/compare/experiment.py:562, 628-651`), which is **9,000 s with the default `FL_CLIENT_TIMEOUT=7200`**.
  - Nothing is accepted wrongly, but the run is only marked failed about 2.5 hours after the round has already failed.
  - **Proposed fix:**
    - In the harness, once every client process has exited, give the server a short grace period (for example 60 s) and then terminate it and mark the mode failed.
    - Pass a `round_timeout` to `ServerConfig` so a standalone server doesn't block forever.

  **Not yet implemented; needs approval, since it's outside Step 5's scope.**
