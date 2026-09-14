# Independent audit: ZKP and HE+ZKP paths

Date: 2026-09-14  
Scope: `zkp`, `he_tenseal_zkp`, and their shared Groth16 service. This is a source audit, not a proof-system audit. No implementation files were changed.

## Executive finding

The claimed Byzantine-update integrity property is not provided by the current code. All five supplied hypotheses are confirmed. More seriously, the current Python and Go proof-response schemas do not match, so a fresh build of the included service cannot emit the proof payload required by the client. The resulting all-client proof-generation failure is then silently aggregated as ordinary FedAvg.

Do not present `zkp`, `zkp_sampled`, `he_*_zkp`, or `he_*_zkp_dp` as providing integrity / Byzantine resistance until the S1 findings are remediated and the benchmark outputs are regenerated.

## End-to-end traces

### `zkp`

1. After local training, `FlowerClient.fit()` calls `mode.send_parameters()` and transmits returned arrays and metrics in Flower `FitRes` ([`fl/client.py:207-229`](../fl/client.py#L207), [`fl/client.py:319-356`](../fl/client.py#L319)).
2. `ZKPMode.send_parameters()` records local `state_dict` keys, calls `_generate_proofs(net.state_dict())`, caches the result, and returns plaintext float32 model-weight arrays ([`fl/privacy/zkp.py:83-95`](../fl/privacy/zkp.py#L83)).
3. The client quantises each selected **model-weight tensor** as `round(float64(w) * scale) -> int64`, sends it and client-derived `bound_sq` to `/prove`, and places the proof, MiMC hash, shape, scale, and bound in client-controlled metrics ([`fl/core/zkp_gnark.py:169-191`](../fl/core/zkp_gnark.py#L169), [`fl/core/zkp_gnark.py:205-270`](../fl/core/zkp_gnark.py#L205)).
4. The server gets plaintext Flower arrays, parses `zkp_proofs_json`, and uses **client-supplied** `zkp_layer_names_json` to pair proofs with parameters ([`fl/privacy/zkp.py:139-175`](../fl/privacy/zkp.py#L139)).
5. `verify_gnark_proofs()` recomputes the quantisation and MiMC hash from each parameter it actually iterates, then calls `/verify` ([`fl/core/zkp_gnark.py:376-445`](../fl/core/zkp_gnark.py#L376)).
6. The strategy takes the mode's accepted subset via `pre_aggregate()` and runs plaintext FedAvg ([`fl/privacy/zkp.py:228-235`](../fl/privacy/zkp.py#L228), [`fl/server.py:206-224`](../fl/server.py#L206)).

The plaintext path binds a proof to a parameter only if the server independently enforces complete canonical parameter coverage. It does not do that now.

### `he_tenseal_zkp`

1. The composite client first calls `ZKPMode.send_parameters()` on plaintext model weights, then independently calls the HE backend to encrypt the same in-memory model ([`fl/privacy/he_zkp.py:130-150`](../fl/privacy/he_zkp.py#L130)).
2. The ZKP payload is client-controlled proof metadata. TenSEAL serialises independent CKKS ciphertext bytes for transport ([`fl/privacy/he_zkp.py:165-176`](../fl/privacy/he_zkp.py#L165), [`fl/privacy/he_tenseal.py:399-409`](../fl/privacy/he_tenseal.py#L399)).
3. The server invokes `verify_gnark_proofs_light()` with only the proof's `shape`, `bound_sq`, `hash_hex`, and `proof_b64`. It does not receive a ciphertext hash, HE encoding, or authenticated proof-to-parameter binding ([`fl/privacy/he_zkp.py:180-238`](../fl/privacy/he_zkp.py#L180), [`fl/core/zkp_gnark.py:448-525`](../fl/core/zkp_gnark.py#L448)).
4. An admitted `FitRes` is forwarded unchanged to `HeTensealMode`, which deserialises and homomorphically averages its ciphertexts ([`fl/privacy/he_zkp.py:249-280`](../fl/privacy/he_zkp.py#L249), [`fl/privacy/he_tenseal.py:307-391`](../fl/privacy/he_tenseal.py#L307)).

The proof establishes a statement about a client-selected hidden vector; the server aggregates an independently supplied ciphertext.

## Hypotheses

| ID | Result | Evidence |
|---|---|---|
| H1 | Confirmed | Plaintext weights are proved before encryption; light verification has no ciphertext input; HE aggregates supplied ciphertexts. |
| H2 | Confirmed | `_select_layers()` defaults `FL_ZKP_NUM_LAYERS` to `"1"`, so its percentage branch is unreachable; default selection is top-1 by size. |
| H3 | Confirmed | Both paths replace an empty accepted subset with the complete unverified result set. |
| H4 | Confirmed | `getCircuit()` runs `groth16.Setup` lazily and caches both proving and verifying keys in the serving process. |
| H5 | Confirmed | The circuit hashes and squares field elements but has no per-witness integer range constraint. |

## Findings

### S1-01 — HE+ZKP proof is not bound to the ciphertext

**Locations:** [`fl/privacy/he_zkp.py:139-150`](../fl/privacy/he_zkp.py#L139), [`fl/privacy/he_zkp.py:215-232`](../fl/privacy/he_zkp.py#L215), [`fl/core/zkp_gnark.py:485-525`](../fl/core/zkp_gnark.py#L485).

The proof and ciphertext are independently generated. `/verify_light` has no public input committing to ciphertext bytes, HE encoding, or encryption of the proved quantised vector. A Byzantine client can prove benign vector A and submit a ciphertext of poisoned vector B; B is admitted and aggregated. This also affects the TFHE and triple modes that inherit the composite base.

Contradicted claim: `README.md:48-51` and `docs/ZKP.md:421-423` describe the composites as providing integrity for the encrypted gradient.

### S1-02 — All-client verification failure silently becomes unverified FedAvg

**Locations:** [`fl/privacy/zkp.py:184-194`](../fl/privacy/zkp.py#L184), [`fl/privacy/he_zkp.py:249-254`](../fl/privacy/he_zkp.py#L249).

After excluding every client for missing, malformed, invalid, or unreachable-service proofs, both paths re-admit `list(results)`. Stop the gnark service: each client produces or supplies no usable proof; the server then averages every unverified update. This contradicts the README threat table and `docs/ZKP.md:220-232`, which state that invalid updates are rejected before aggregation.

### S1-03 — Current `/prove` response cannot satisfy the Python client schema

**Locations:** [`fl/core/zkp_gnark.py:239-260`](../fl/core/zkp_gnark.py#L239), [`zkp_gnark_service/main.go:46-51`](../zkp_gnark_service/main.go#L46), [`zkp_gnark_service/main.go:220-226`](../zkp_gnark_service/main.go#L220), [`fl/privacy/zkp.py:250-275`](../fl/privacy/zkp.py#L250).

Python requires a non-empty `shape` field in `/prove`'s response. Go's `proofResponse` has no `Shape` field and `proveHandler` returns only `proof_b64` and `hash_hex`. Python raises `Missing 'shape'`; `ZKPMode` catches it and returns `([], 0)`, which activates S1-02. A newly compiled service from this source therefore cannot create the expected proof payload.

This makes results generated from this exact source unsuitable as evidence of verified ZKP operation. Whether stored data used a different binary or revision is a Step 2 question and cannot be inferred from documentation.

**Runtime confirmation (added during Step 2 review).** The committed binary `zkp_gnark_service/gnark_service` was started and sent a 2-element `/prove` request; the response keys were exactly `hash_hex`, `proof_b64`, `verified` — no `shape`. The current Python client therefore discards every proof from the current binary.

**Stored results came from a different, unpreserved revision.** All stored runs are dated 2026-03-18 to 2026-03-22; the only commit touching `main.go` and `fl/core/zkp_gnark.py` is `7f52274` (2026-06-04). The stored ledgers contain a `ProofAnchor` for every round with proofs from every aggregated client, and `proof_verification` timings were recorded (a timer that only runs when a client sent proofs). So proofs were produced and transmitted when the results were generated, by code that is not in git history. The stored numbers cannot be reproduced from this repository as committed. See `audit/numbers.md`.

### S1-04 — Plaintext `zkp` accepts client-controlled incomplete coverage

**Locations:** [`fl/privacy/zkp.py:154-175`](../fl/privacy/zkp.py#L154), [`fl/core/zkp_gnark.py:376-445`](../fl/core/zkp_gnark.py#L376).

The server trusts client `zkp_layer_names_json` and uses `zip(layer_names, parameters)`. It never checks that names/count match a server-owned model schema or rejects excess parameters. If a client sends `[]`, the loop runs zero times and returns `(True, [])` when `zkp_proofs_json` is non-empty. Shorter lists leave excess tensors unverified.

Attack: submit poisoned tensors, a syntactically non-empty proof list, and `zkp_layer_names_json="[]"`; the update is treated as verified and enters FedAvg. This contradicts `README.md:46` (“full coverage”).

### S1-05 — The client, rather than the server, chooses the norm bound

**Locations:** [`fl/core/zkp_gnark.py:210-222`](../fl/core/zkp_gnark.py#L210), [`fl/core/zkp_gnark.py:396-425`](../fl/core/zkp_gnark.py#L396), [`fl/core/zkp_gnark.py:497-503`](../fl/core/zkp_gnark.py#L497), [`zkp_gnark_service/main.go:166-177`](../zkp_gnark_service/main.go#L166).

`bound_sq` is created from a client environment variable and forwarded in proof metadata. Both verification paths use it without comparing it to server policy; the service only requires a positive value below the field modulus. An attacker can prove an outlier under a huge valid bound. This contradicts the claimed agreed maximum in `README.md:458` and the common-bound Byzantine argument in `docs/ZKP.md:238`.

### S1-06 — `zkp_sampled` is deterministic and incompatible with full verification

**Locations:** [`fl/privacy/zkp.py:303-347`](../fl/privacy/zkp.py#L303), [`fl/privacy/zkp.py:139-175`](../fl/privacy/zkp.py#L139), [`fl/core/zkp_gnark.py:379-445`](../fl/core/zkp_gnark.py#L379).

`FL_ZKP_NUM_LAYERS` defaults to the literal string `"1"`, so the percentage branch is unreachable. Default `FL_ZKP_SELECT_BY=size` selects the largest layer deterministically. But `send_parameters()` publishes every model-layer name and the verifier requires proofs for all of them. Honest sampled clients are rejected for missing proofs; when all are rejected S1-02 aggregates them unverified. An attacker can also predict the selected layer and poison an unproved layer. This contradicts `README.md:45,452-455` and `docs/ZKP.md:228-246`.

**Correction (added during Step 2 review): the benchmark harness never instantiates `ZKPSampledMode`.** `fl/compare/registry.py` defines `zkp_sampled` with `internal_mode="zkp"`; `_build_mode_flags()` in `fl/compare/experiment.py` emits only `--zkp`; and `_resolve_mode()` in `simulation.py` and `main_server.py` maps `--zkp` to `"zkp"`. The rejection-then-fail-open behaviour described above applies to the class, but it did not occur in stored results. Instead every stored `zkp_sampled` row is a second run of full `ZKPMode`: the ledgers show identical proofs per client per round for `zkp` and `zkp_sampled` on all five datasets (7, 7, 7, 31, 39), matching full chunked coverage. The class is dead in the benchmark; the mode label is wrong.

### S1-07 — The circuit bounds model weights, not the client update

**Locations:** [`fl/privacy/zkp.py:91-95`](../fl/privacy/zkp.py#L91), [`fl/core/zkp_gnark.py:215-222`](../fl/core/zkp_gnark.py#L215), [`zkp_gnark_service/main.go:70-81`](../zkp_gnark_service/main.go#L70).

The witness is the post-training local `state_dict`, not `w_local - w_global`. A bounded final model does not bound the FedAvg update. For a coordinate moving from approximately `-B` to `+B`, the submitted update is almost `2B` while its final value satisfies the proved bound. This contradicts `README.md:58` and `docs/ZKP.md:228`, which call it a gradient or update norm proof.

### S1-08 — Trusted setup is generated and retained by the prove/verify service

**Locations:** [`zkp_gnark_service/main.go:53-103`](../zkp_gnark_service/main.go#L53), [`fl/compare/experiment.py:126-201`](../fl/compare/experiment.py#L126).

The first request for each length compiles the circuit, runs `groth16.Setup`, and caches `pk` and `vk` in one HTTP process. The benchmark runner launches or reuses one local service shared by client and server. The verifier trusts that process and its single-party setup randomness; no verifying-key identity is persisted. After restart a proof made with the old key cannot be verified by the new key. This contradicts `docs/ZKP.md:347`'s stateless/precompiled-service description and makes stored proof hashes non-auditable by themselves.

### S1-09 — Witness values have no range constraints

**Locations:** [`zkp_gnark_service/main.go:64-81`](../zkp_gnark_service/main.go#L64), [`zkp_gnark_service/main.go:106-122`](../zkp_gnark_service/main.go#L106).

Int64 inputs become BN254 field elements. The circuit only hashes them and computes a square sum in the field; it has no bit decomposition or signed integer range constraint. It therefore does not establish the claimed integer L2-norm statement. Exact exploitability for current `n` and scale is deferred to the requested Step 8 arithmetic analysis.

## Error-path matrix

| Event | Current behavior | Required secure behavior |
|---|---|---|
| `/prove` unreachable or invalid response | Exception becomes `([], 0)`; no proof metrics. Composite inherits this. | Client fit fails or explicitly rejects its update. |
| Missing or malformed proof JSON | Client excluded; all-client case falls open under S1-02. | Reject client; abort/no update if quorum is not met. |
| `/verify` or `/verify_light` unreachable | Helper reports failure; all-client case falls open. | Abort/no update and report infrastructure failure. |
| Incomplete proof fields | Some failures are handled; uncaught key/JSON-response failures can escape strategy code. | Parse errors become rejection/abort with explicit cause. |
| No admitted clients but non-empty results | Full unverified result set is aggregated. | No aggregation, central update, ModelCommit, or ProofAnchor. |
| Zero results | Plain `zkp` returns `(None,{})` before chain commit. | Explicit no-quorum/no-admission result. |

The ledger does not repair this. Both modes build anchor data after restoring the full unverified result set, so `_chain_commit()` can write `ModelCommit` and `ProofAnchor` for a round with no verified updates ([`fl/privacy/zkp.py:200-225`](../fl/privacy/zkp.py#L200), [`fl/privacy/he_zkp.py:256-276`](../fl/privacy/he_zkp.py#L256), [`fl/server.py:340-404`](../fl/server.py#L340)).

## Test evidence and limit

`tests/test_zkp_sampled.py` expects percentage sampling even though the default `FL_ZKP_NUM_LAYERS="1"` wins. The original audit could not collect the suite because its Python lacked `torch`.

**Update:** run in the project conda environment (`flEnv`, Python 3.11), `tests/test_zkp_sampled.py::test_pct_sampling_env` fails with `assert 1 == 4` (`FL_ZKP_SAMPLE_PCT=0.5` over 7 layers yields 1 layer). That is runtime confirmation of H2. All other tests pass. This failure is left in place for Step 5.

## Required next action

Do not implement fixes in this audit phase. The highest-value next phase is Step 3 Phase 1, but any design must be paired with fail-closed behavior and a server-owned policy/schema before integrity claims are restored.

