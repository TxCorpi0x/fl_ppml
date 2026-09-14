# Binding the proof to the uploaded update

Date: 2026-09-14
Scope: `fl/privacy/he_zkp.py`, `fl/privacy/he_tenseal.py`, `fl/privacy/he_concrete_tfhe.py`, `fl/privacy/he_zkp_dp.py`, `fl/core/zkp_gnark.py`, `fl/core/concrete_agg.py`, `fl/core/security.py`, `zkp_gnark_service/main.go`. This is Step 3 Phase 1. The only source change is the `/prove` schema fix described below. **No binding fix has been written. A design must be chosen first.**

## Prerequisite fix: `/prove` response schema (findings.md S1-03)

**Change:** `fl/core/zkp_gnark.py::generate_gnark_proofs` now takes `shape` from its own request instead of expecting the service to echo it.
- If a service *does* echo a shape, it must match, or proof generation raises.
- A missing `hash_hex` or an empty tensor also raises.
- `main.go` is unchanged.

**Evidence:** `tests/test_gnark_service_roundtrip.py` runs against the real `zkp_gnark_service/gnark_service` binary on a free port:

| Test | Result |
|---|---|
| one complete proof per layer | pass |
| proofs verify via `/verify` and `/verify_light` | pass |
| `/verify` rejects tampered parameters | pass |
| `/verify_light` rejects a changed hash | pass |
| `/verify_light` rejects a non-canonical hash (`hash + r`) | **xfail (strict)** — see B-5 |

Full suite in `flEnv`: `38 passed, 3 xfailed, 1 failed`. The one failure is the pre-existing `test_zkp_sampled.py::test_pct_sampling_env`, kept as H2 evidence for Step 5.

## Q1 — What the server has, and what it aggregates (composite modes)

```
CLIENT (he_tenseal_zkp)                                    SERVER
─────────────────────────────────────────────              ──────────────────────────────────────────
w = net.state_dict()   (float32, post-training)
│
├─ ZKP path  (he_zkp.py:139-142 → zkp.py:83-95)
│   q = round(float64(w) · FL_ZKP_SCALE)  int64            fit_res.metrics["zkp_proofs_json"]:
│   bound_sq = (FL_ZKP_MAX_NORM·scale)²  ← client env        per proof: layer, shape, scale,
│   /prove(q, bound_sq) → proof, hash=MiMC(q)                bound_sq, hash_hex, proof_b64
│                                                            │
│                                                            ▼
│                                               verify_gnark_proofs_light(proofs)  (he_zkp.py:215-222)
│                                                 /verify_light: Groth16.Verify(vk_n, proof,
│                                                   public = (bound_sq, hash_hex))
│                                                 n = ∏shape  ← from client metadata
│                                                            │ admitted / excluded
│                                                            ▼
└─ HE path  (he_zkp.py:143-150 → he_tenseal.py:399-439)   fit_res.parameters:
    crypte(w, ctx, encrypt_layers)                           per layer: zlib(CVEC(shape) ‖ ckks_vector)
    selected layers → ts.ckks_vector(float list)             or zlib(.npy plaintext) for other layers
    other layers    → np.save (plaintext)                    │
                                                             ▼
                                               HeTensealMode.aggregate_fit_override
                                                 Σ αᵢ · ctᵢ   per layer   (he_tenseal.py:307-391)
```

**The server's inputs at verification time:** `(shape, bound_sq, hash_hex, proof_b64)` per proof, all written by the client. `bound_sq` and `shape` are never compared against server policy or the model schema, and nothing requires the proofs to cover every layer.

**What it aggregates:** the uint8 arrays in `fit_res.parameters`, which are CKKS ciphertexts for selected layers and plaintext `.npy` for the rest.

**No value appears on both sides.** Nothing links the ciphertext (or plaintext) arrays to `hash_hex`: no digest, commitment or shared randomness.

## Q2 — Hypothesis S1-01 is confirmed

I found no binding anywhere in the path traced above. The attack runs end to end with real CKKS and real Groth16.

**Test:** `tests/test_zkp_binding_attack.py::test_he_tenseal_zkp_rejects_ciphertext_that_does_not_match_proof`

**Setup:**
- Two clients. Both prove the honest vector A with the real gnark service.
- The honest client encrypts A. The attacker encrypts the poisoned vector B (every element 1000.0; ‖B‖ ≈ 2236 per layer, far above the bound of 100).
- Every layer is encrypted, which is the strongest HE configuration, not the default (see B-1).

**Control:** `test_control_poisoned_vector_cannot_be_proved` passes. The service refuses to prove B directly, so the norm bound does bite and the attacker's only way in is the missing binding.

**Result with `--runxfail`:**

```
AssertionError: attacker admitted ['honest', 'attacker']; decrypted aggregate model.0.weight =
  [[500.05, 499.9], [500.15, 499.8]]
  (honest mean would be [[0.1, -0.2], [0.3, -0.4]])
```

The decrypted aggregate is exactly (A + B) / 2. The attacker's out-of-bound update was averaged into the global model under a "verified" ZKP. The test asserts the secure outcome and is marked `xfail(strict=True)`, so it fails today. Phase 2 must make it pass by removing the marker, not by editing the assertion.

**The same gap exists in every composite:**
- `he_concrete_tfhe_zkp` inherits `_HeZKPCompositeMode.aggregate_fit_override` unchanged.
- `he_tenseal_zkp_dp` and `he_concrete_tfhe_zkp_dp` subclass it, and `he_zkp_dp.py` overrides only setup and metrics.

## Q3 — Tests written

| Test | Asserts | Status today |
|---|---|---|
| `test_he_tenseal_zkp_rejects_ciphertext_that_does_not_match_proof` | attacker not admitted; aggregate ≈ A | **xfail strict** — attack succeeds |
| `test_plaintext_zkp_rejects_update_with_empty_layer_names` | attacker not admitted | **xfail strict** — attack succeeds (S1-04) |
| `test_control_plaintext_zkp_rejects_mismatched_params_with_full_layer_names` | attacker excluded | pass |
| `test_control_poisoned_vector_cannot_be_proved` | `/prove` refuses B | pass |

## Q4 — Plaintext `zkp` mode

**Plaintext mode does bind proof to parameters, but only for the layers the server chooses to check, and the client chooses that list.**

| | `verify_gnark_proofs` (plaintext `zkp`) | `verify_gnark_proofs_light` (composites) |
|---|---|---|
| Inputs | received parameters, client layer names, proofs | proofs only |
| Hash public input | **recomputed by the server** from the received parameters (`zkp_gnark.py:424-427` → `/verify` runs `computeHash(weights)`) | taken from the client's `hash_hex` |
| Circuit size | from the received parameter length | from the client's `shape` |
| Binding to the aggregated vector | yes, per verified layer | none |
| Coverage | the pairs in `zip(layer_names, parameters)`, where `layer_names` comes from the client | only the proofs the client chose to send |

**The control test passes:** with honest, complete layer names, an attacker whose parameters differ from the proved vector is excluded.

**The attack test fails:** sending `zkp_layer_names_json = "[]"` makes the loop run zero times, `(True, [])` is returned, and the poisoned update is averaged in.

**B-4 (code-level, not yet tested): partial coverage inside one chunked layer.** For layers split into chunks, `verify_gnark_proofs` checks only the chunk keys that are present (`zkp_gnark.py:381-417`). A client that sends `layer__chunk_0` but leaves out later chunks gets only the first 2000 elements checked, and the rest of the layer is aggregated unverified.

## Q5 — Quantization: what the circuit hashes vs what HE encrypts

| Path | Plaintext representation | Where it's computed | Lossy? |
|---|---|---|---|
| Circuit witness | `q = round(float64(w) · 10⁶)` as int64; negative values become field elements `r − |q|`; MiMC hashes field elements; the norm is `Σq²` in the field | Python client (`zkp_gnark.py:159-160`), **outside** the circuit; the circuit trusts the witness integers | Rounds at 10⁻⁶; no range check (S1-09) |
| CKKS (TenSEAL) | `ts.ckks_vector(w.flatten().tolist())`: float64 values encoded through the canonical embedding at scale 2⁴⁰; randomized RLWE encryption, N = 8192, moduli [60, 40, 40, 60] bits | `security.py:227-234` | Approximate: decryption returns w plus noise; ciphertexts are randomized |
| TFHE (Concrete), tabular | `q14 = round((clip(w, −5, 5) + 5)/10 · 16383) − 8192`, then `prescaled = q14 // num_clients`; `bit_width`, `shape`, `scale`, `quant_min/max` sent in a clear header | `concrete_agg.py:902-931`, `784` | 14-bit (step ≈ 6.1·10⁻⁴), clipped at ±5, floor-divided by client count |
| TFHE, image datasets | the same `q14` int32 array with `is_simulated=True`, **not encrypted** (B-2) | `concrete_agg.py:753-765` | as above |

**These representations do not match, and neither HE vector is the one the circuit hashes:**
1. **CKKS has no integer plaintext.** Tying it to the circuit means proving the encoding relation (canonical embedding plus rounding) and the RLWE encryption relation. A ciphertext digest alone says nothing about content.
2. **TFHE encrypts `q14 // n`, a different vector.** It's 14-bit, clipped at ±5, and pre-divided. The circuit hashes 10⁶-scaled weights and allows a norm of 100, so a coordinate the circuit accepts (e.g. 50.0) is silently clipped to 5.0 by TFHE. Any binding must first make the circuit hash and bound **exactly the integers that are encrypted**, including the clipping, the division and the client count. The clear quantization header must also be server policy, not client input.
3. **In both cases the norm has to be proved in the encrypted domain:** on `q14 // n` for TFHE, and on the encoded plaintext with a stated error margin for CKKS. Otherwise the proved bound doesn't limit what gets aggregated.

## New findings from this investigation

| ID | Severity | Location | Finding | Evidence |
|---|---|---|---|---|
| B-1 | S1 | `he_tenseal.py:414-419`, `config.py:134-139`, `client.py:151,212` | **TenSEAL encrypts only `model.0.weight` and `model.0.bias` by default.** The config default `encrypt_layers="ALL"` becomes `None`, and `_encrypt_params` treats `None` as "read `FL_ENCRYPT_LAYERS`", whose default is `"model.0.weight,model.0.bias"`. On tabular models 2 of 6 tensors are encrypted (768 of 2914 healthcare parameters) and the rest go as plaintext `.npy`. On CNNs (`conv1.*`, `fc*.*`) no names match, so **nothing is encrypted**. Nothing in the repo or harness sets `FL_ENCRYPT_LAYERS`. The claim "server never sees plaintext weights" (`he_zkp.py:17`, README) is false for every stored run. | Code path above. Stored uploads match: tabular `he_tenseal` 673,619 B ≈ two ciphertexts plus about 9 KB of plaintext; MNIST/CIFAR `he_tenseal` uploads are *smaller* than the plaintext model (numbers.md N-5) |
| B-2 | S1 | `he_concrete_tfhe.py:30-44,60-79`, `concrete_agg.py:753-765` | **Real TFHE is turned off on MNIST/CIFAR** unless `FL_CONCRETE_TFHE_FORCE_REAL=1`, which nothing in the repo sets. Clients then send `EncryptedTensor(is_simulated=True, ciphertext=<plaintext int32 q14>)`. Every stored image-dataset TFHE row therefore sent quantized plaintext while being reported as encrypted. | Code path. Stored image TFHE uploads (MNIST 71,038 B; CIFAR 94,744 B) are consistent with zlib-compressed int32 quantized weights, not TFHE ciphertexts |
| B-3 | S1 (code-level, not tested) | `he_concrete_tfhe.py:234-240` | **TFHE aggregation fails open:** any exception during encrypted aggregation prints a warning and returns `None`, dropping into the plaintext FedAvg path. What that path does with `CFH2` envelopes (which `_decompress_cte2_results` doesn't recognize) was not tested. | Code |
| B-4 | S1 (code-level, not tested) | `zkp_gnark.py:381-417` | Partial chunk coverage in plaintext verification (Q4) | Code |
| B-5 | S3 | `main.go:340-345`, `zkp_gnark.py:485-503` | **Public inputs are not canonical.** `/verify_light` parses `hash_hex` as any big integer and gnark reduces it mod r, so `hash` and `hash + r` both verify. Two distinct payloads verify for one proof, so ledger proof hashes (`hash_proof_payload`) aren't unique per proof, and any design that compares public inputs byte-for-byte must canonicalize first. `bound_sq` goes through the same `parseBigInt` path in `/verify_light`, which lacks `/prove`'s `< r` check. | `test_light_verification_rejects_non_canonical_hash` (strict xfail): `hash + r` verified for both layers |
| B-6 | S1 | `he_zkp.py:195-247`, `zkp_gnark.py:485-526` | **The composite verifier enforces nothing about coverage.** Any non-empty proof list with valid proofs is accepted: one proof for a 2-element bias admits the whole update. The server has no model schema to check `shape` or proof count against. **This holds even if S1-01 is fixed:** binding each proof to a ciphertext still requires the server to demand a bound proof for every layer and chunk. | Code; same structure as S1-04 |

## Binding designs

These apply to all four composites. Every design also needs the following, or the binding means little:
- a server-owned schema and required coverage (S1-04, B-4, B-6)
- a server-owned bound (S1-05)
- a bound on the update, not the weights (S1-07)
- per-element range constraints (S1-09)
- canonical public inputs (B-5)
- all layers actually encrypted (B-1, B-2)
- fail-closed aggregation (S1-02, B-3)
- a pinned verifying key (S1-08)

A public input that isn't tied to server context should also carry `(round, client_id)`, to stop proofs being replayed across rounds and clients.

The constraint counts below are **estimates to be measured**, not results. The only current figure is the code comment of ~332 constraints/element (`zkp_gnark.py:37-40`), which is itself unmeasured. Phase 2 should record `cs.GetNbConstraints()` for each design.

### Design A — Commit to the ciphertext bytes as a public input

- **Circuit:** unchanged witness. Add a public input `d = H(ct_bytes ‖ round ‖ client_id)` computed outside the circuit and passed to both prover and verifier. The server recomputes `d` from what it received.
- **Constraints:** about +1 if `d` is just a public input with no constraint on it. Hashing the ciphertext *inside* the circuit would be ~10⁷ constraints or more for ~330 KB (SHA-256 is ~25k constraints per 64-byte block), and still wouldn't help.
- **Guarantees:** the proof and ciphertext were submitted together, and the proof can't be moved to another round, client or ciphertext.
- **Does NOT guarantee:** that `ct` encrypts the vector that was hashed and bounded. The circuit never relates `d` to the witness. **The attack in Q2 still works unchanged:** prove A, bind the proof to the digest of `Enc(B)`, submit.
- **Verdict:** needed hygiene against replay, and cheap. **Not a binding.** Include it with any other design; don't ship it alone.

### Design B — Verifiable additive encryption that is cheap inside the circuit (exponential ElGamal on BabyJubJub)

- **Replace the upload encryption in composite modes** with lifted ElGamal over the BN254 twisted-Edwards curve (BabyJubJub), whose arithmetic is native to the Groth16 field.
- **Circuit (per coordinate i, quantized update vᵢ):**
  - range-check `vᵢ ∈ [−2ᵏ, 2ᵏ)`
  - `C1ᵢ = rᵢ·G` and `C2ᵢ = vᵢ·G + rᵢ·PK`
  - `Σ vᵢ² ≤ B`
- **Public inputs:** `PK`, `B` (server policy), every `(C1ᵢ, C2ᵢ)` (or one digest of them computed in-circuit with Poseidon/MiMC), `round`, `client_id`. The server aggregates by adding points. The key holders decrypt the sum with a bounded discrete log (the sum ≤ n_clients·2ᵏ per coordinate, solvable with baby-step giant-step).
- **Constraints:** per coordinate, two scalar multiplications on a native twisted-Edwards curve (on the order of several thousand constraints each, one of them fixed-base), plus a k-bit range check (~k) and one square. Estimate: **roughly 10–30× the current per-element cost.** At the healthcare model size (2914 parameters) that is proving on the order of 10⁷ constraints per client per round, split across chunks as today.
- **Guarantees:** the aggregated ciphertext encrypts exactly the vector whose range and norm were proved. The server learns only the public inputs. **Q2's attack becomes impossible,** because a ciphertext of B has no valid proof.
- **Does NOT guarantee:**
  - honest training or clean data (only a norm-bounded update)
  - confidentiality against anyone holding the shared decryption key (the same trust model as today, where every client loads one shared CKKS secret)
  - post-quantum security (discrete log)
  - float precision beyond the fixed quantization
- **Costs:**
  - decrypting a large model with discrete logs is slow (per-coordinate BSGS; manageable for small ranges and tabular models, expensive for CNNs)
  - it replaces CKKS/TFHE in the composites, which changes the thesis comparison: these become a third backend, not "CKKS + ZKP"
  - ciphertexts are 2 curve points per coordinate
- **Verdict:** the only design here that fixes S1-01 soundly and fits in Groth16/gnark within an MSc-scale engineering effort. **Recommended for Phase 2,** as a new composite mode (e.g. `he_elgamal_zkp`). The existing CKKS/TFHE composites would be relabelled as confidentiality-only with an unbound proof.

### Design C — Verifiable RLWE encryption for CKKS, inside the SNARK

- **Circuit:** witness `(m, u, e0, e1)` with `ct0 = pk0·u + e0 + m`, `ct1 = pk1·u + e1` in `R_q = Z_q[X]/(X^N+1)`, for each RNS limb q ∈ {60, 40, 40, 60}-bit.
  - Prove `u` ternary and `e0, e1` small (per-coefficient range checks).
  - Check polynomial products at one random point (Schwartz–Zippel) instead of coefficient-wise.
  - Prove `m` is the CKKS encoding of the quantized vector `v`: `decode(m) = v ± ε`, which needs the canonical-embedding transform.
  - Prove `‖v‖² ≤ B` with `ε` slack.
- **Public inputs:** `ct0, ct1` (or an in-circuit digest), `pk`, `B`, `round`, `client_id`.
- **Constraints:** per ciphertext, 2 components × 4 limbs × 8192 coefficients of modular reduction and range checks (~40–60 bits each) is already ~10⁶–10⁷, before the encoding transform. Plus a proving key of several GB, and one or more ciphertexts per layer. **Estimate: minutes to tens of minutes of Groth16 proving per layer per client,** on top of today's cost.
- **What's specifically hard for CKKS:**
  1. **Encoding** uses the canonical embedding (a complex FFT with rounding), not coefficient encoding, so proving `decode(m) ≈ v` needs a non-native, approximate transform.
  2. **The norm bound is in slot space** while the noise bounds are in coefficient space. They relate through an embedding-norm factor (up to √N), which loosens the proved bound.
  3. **Correctness is approximate:** the proof can only claim `‖decode(Dec(ct))‖ ≤ √B + ε`, and ε depends on noise that grows through aggregation.
  4. **Non-native moduli** need per-coefficient reduction proofs.
  5. **TFHE has the same problem,** as LWE ciphertexts of dimension ≥ 1024 over a 64-bit torus per coordinate.
- **Guarantees:** keeps CKKS and binds its ciphertext to a bounded plaintext, up to ε.
- **Does NOT guarantee:** exact bounds (only up to ε). Groth16 is also the wrong tool here: lattice-native ZK proofs are the realistic route, and gnark has no ready gadgets for them.
- **Verdict:** correct in principle, but research-grade. **Not feasible as a Phase-2 implementation.** Worth a paragraph in the paper as the path that keeps CKKS.

### Design D (non-cryptographic fallback) — Decrypt-and-check audit

- **Mechanism:** no circuit change. For a random fraction p of (client, round) submissions, chosen *after* upload by a server-committed seed, a party holding the decryption key decrypts that client's ciphertext. It re-quantizes exactly as the circuit does and checks `MiMC(q) == hash_hex` and `Σq² ≤ B`. A failure excludes the client and can revoke it.
- **Constraints:** +0.
- **Guarantees:** a cheating client is caught with probability p per submission (1 − (1 − p)^t over t rounds). Deters persistent attackers.
- **Does NOT guarantee:**
  - that any single round is clean (a one-shot poison slips through with probability 1 − p)
  - confidentiality of audited updates against the auditor, which breaks the model where the server never sees plaintext unless the auditor is separate and trusted
  - that CKKS decryption noise won't break exact hash equality (it will, so the check must allow a tolerance, which weakens it)
- **Verdict:** a fallback for deployments that must keep CKKS now. **Not a cryptographic binding,** and it can't be described as one.

## Summary of options for you to choose from

| | Fixes Q2 attack | Keeps CKKS/TFHE | Extra prover cost (estimate) | Feasible for Phase 2 |
|---|---|---|---|---|
| A: ciphertext digest as public input | No | Yes | ≈ 0 | Yes, but insufficient alone |
| B: ElGamal-on-BabyJubJub verifiable encryption | **Yes** | No (new backend) | ~10–30× per element | **Yes (recommended)** |
| C: RLWE verifiable encryption | Yes, up to ε | Yes | orders of magnitude | No (research) |
| D: decrypt-and-check audit | Probabilistically | Yes | 0 | Yes, but not cryptographic |

**Recommendation:** B + A, together with the prerequisites listed above. Relabel the existing CKKS/TFHE composites as confidentiality-only with no integrity claim, and fix B-1 and B-2 so their confidentiality claim holds. The Phase-2 cost benchmark would compare `he_elgamal_zkp` against `he_tenseal_zkp` on healthcare after rebaselining, because the stored unbound numbers can't be reproduced from committed code (findings.md S1-03).

**Stopping here, as Step 3 requires.** No binding code is written until a design is chosen.

---

# Phase 2 — Design B + A implemented

Chosen: B + A, with the CKKS/TFHE composites relabelled confidentiality-only, and B-1 and B-2 fixed.

## What was built

| Component | Location |
|---|---|
| Circuit: range-check `v ∈ [0, 2¹⁸)`, `C1 = r·G`, `C2 = v·G + r·PK`, `Σ(v − 2¹⁷)² ≤ Bound`; public inputs PK, Bound, Context and every ciphertext coordinate; Context bound by an explicit constraint | `zkp_gnark_service/elgamal.go` |
| Native keygen, encrypt+prove, verify with canonical point and field checks, weighted homomorphic aggregation, BSGS decryption; endpoints `/elgamal/{prove,verify,aggregate,decrypt,info}`; `gnark_service elgamal-keygen` | same file, `main.go` |
| Python transport, policy, schema, chunking, per-chunk bounds, context, quantization; infrastructure errors kept separate from client rejections | `fl/core/elgamal_gnark.py` |
| Mode `he_elgamal_zkp` | `fl/privacy/he_elgamal_zkp.py` |
| Key files and `python -m fl.keys generate he_elgamal` | `fl/keys/he_elgamal.py`, `fl/keys/cli.py` |
| Server-owned schema: new `PrivacyMode.bind_server_model` hook called by `make_strategy`; round reaches the client through a new `on_fit_config` hook called at the start of `fit` | `fl/privacy/base.py`, `fl/server.py`, `fl/client.py` |
| Harness and CLI wiring (`--he --he_backend elgamal --zkp`; `elgamal` with `--dp` is refused) | `fl/compare/registry.py`, `main_client.py`, `main_server.py`, `simulation.py`, `fl/config.py` |
| Relabel: composite docstring, registry display names "(unbound)", README mode table and triple-mode text | `fl/privacy/he_zkp.py`, `fl/compare/registry.py`, `README.md` |
| B-1 fix: TenSEAL encrypts every layer unless `FL_ENCRYPT_LAYERS` is set explicitly; unknown names are an error. The decrypt path already decides per layer from content (`deserialized_layer`). **Correction:** the first version of this fix did not reach harness runs, because `fl/compare/experiment.py::_grpc_env` injected `FL_ENCRYPT_LAYERS=model.0.weight,model.0.bias` into every subprocess. The 2-round healthcare run `results/healthcare/20260914_202700` shows it: only `model.0.*` recorded encryption time, and the server received 1.29 MB for 2 clients (≈665 KB each = two ciphertexts; all six layers is ≈2.0 MB). `_grpc_env` no longer injects the variable; `test_harness_does_not_force_partial_encryption` pins this | `fl/privacy/he_tenseal.py`, `fl/compare/experiment.py` |
| B-2 fix: TFHE on image datasets refuses to run unless `FL_CONCRETE_TFHE_FORCE_REAL=1` or `FL_CONCRETE_TFHE_ALLOW_SIMULATED=1`, and the simulated path logs that weights are plaintext | `fl/privacy/he_concrete_tfhe.py` |

**Server-side rules** in `HeElGamalZKPMode.aggregate_fit_override`:
- **Schema, chunking, bounds, context:** derived from the server's model and its `Policy`.
- **Coverage:** every expected (layer, chunk) needs exactly one proof. Missing, duplicate or extra proofs reject the client.
- **Verification:** each proof is checked against the ciphertext bytes the server received.
- **Rejected clients:** excluded and logged with the reason.
- **No admitted client:** the global model is unchanged and no proof anchor is written.
- **Proof service unreachable or failing:** the round aborts, reported as an infrastructure failure rather than a client fault.

## Test evidence

**Go (`go test ./...`, 8 tests, all pass):**

| Test | Shows |
|---|---|
| `TestElgamalHonestRoundTrip` | Weighted aggregate of two proved uploads decrypts to exactly `2a + 3b` |
| `TestElgamalRejectsCiphertextOfAnotherVector` | Proof over A fails against a ciphertext of B |
| `TestElgamalProofIsBoundToPublicInputs` | Proof fails under a changed context, bound, public key, or ciphertext order |
| `TestElgamalRefusesOverBoundNorm` | Norm one above the bound can't be proved; exactly at the bound can |
| `TestElgamalCircuitRejectsOutOfRangeRawWitness` | With the service pre-check bypassed (a client using its own prover), the circuit itself refuses v = 2¹⁸ and v = −1 |
| `TestElgamalCircuitAcceptsInRangeRawWitness`, `TestElgamalRefusesOutOfRangeValues`, `TestDecodeRejectsNonCanonicalPoints` | In-range edge values prove; out-of-range values are refused before proving; altered point encodings are rejected |

**Python (`tests/test_he_elgamal_zkp.py`, 13 tests, all pass, real keys and real service):**

| Test | Shows |
|---|---|
| `test_honest_clients_aggregate_to_weighted_mean` | Full client → server → client path decrypts to the num_examples-weighted mean (within 1/scale) |
| `test_client_cannot_prove_a_poisoned_vector` | The poisoned vector can't be proved under the server bound |
| `test_proof_over_honest_vector_does_not_admit_poisoned_ciphertext` | **The Q2 attack, rejected:** valid proofs over A plus ciphertexts of B (made under an inflated bound). Attacker excluded, aggregate equals the honest client alone, anchor lists only the honest client |
| `test_proof_replayed_from_an_earlier_round_is_rejected` | Round-1 upload rejected in round 2; model unchanged, no anchor |
| `test_incomplete_or_rearranged_proof_coverage_is_rejected` ×3 | Dropped, duplicated or swapped chunk proofs are rejected |
| `test_unreachable_service_aborts_round_without_aggregating` | Infrastructure failure aborts the round; no aggregate, no anchor |
| `test_server_without_bound_schema_refuses_to_aggregate`, `test_server_key_file_must_not_contain_secret`, `test_simulation_mode_is_refused` | Fail-closed configuration checks |
| `test_tenseal_encrypts_every_layer_by_default`, `test_tenseal_rejects_encrypt_layer_names_missing_from_model` | B-1 regression, including a CNN (every layer arrives as CVEC ciphertext) |
| `test_tfhe_on_image_dataset_refuses_silent_plaintext` | B-2 regression |

**Full suite in `flEnv`:** `52 passed, 3 xfailed, 1 failed`.
- The failure is the pre-existing `test_zkp_sampled.py::test_pct_sampling_env` (H2, Step 5).
- The three strict xfails are the known open defects: S1-04, B-5, and the relabelled CKKS composite.
- `tests/test_zkp_binding_attack.py::test_he_tenseal_zkp_rejects_ciphertext_that_does_not_match_proof` stays a strict xfail on purpose. That mode is not being fixed; it is relabelled. Its secure counterpart is the Q2 attack test above, which passes as a rejection.

## Measured cost: circuit level

Apple M3 Pro (11 cores), gnark v0.10.0, Groth16/BN254, one 128-coordinate chunk of N(0, 0.1) weights. Times are steady state (best/median of 3 after a warm-up call that absorbs circuit compile and setup), measured through the HTTP service from Python.

| | Unbound norm circuit (`he_*_zkp`) | Bound ElGamal circuit (`he_elgamal_zkp`) | Ratio |
|---|---|---|---|
| R1CS constraints, n = 1 | 2,209 | 8,157 | 3.7× |
| R1CS constraints, n = 32 | 12,470 | 202,682 | 16.3× |
| R1CS constraints, n = 128 | 44,246 (≈346/coord) | 805,082 (≈6,290/coord) | **18.2×** |
| Public inputs, n = 128 | 3 | 517 | — |
| Prove time, n = 128 (min / median) | 0.383 / 0.389 s | 2.035 / 2.062 s | **5.3×** |
| Verify time, n = 128 (min / median) | 1.8 / 1.8 ms | 5.3 / 5.6 ms | 3.1× |
| Proof size (base64) | 220 B | 220 B | 1× |
| Upload per coordinate | 4 B plaintext (+ separate CKKS/TFHE ciphertext) | 64 B ciphertext | — |

**Reading:**
- The constraint ratio (18×) matches the 10–30× estimate in Design B.
- Prove time grows less (5.3×) because the gnark prover parallelizes the extra scalar-multiplication constraints.
- Verification grows with the 517 public inputs but stays in milliseconds.
- The unbound column is the old prover under its new, working schema, so it's comparable. Its stored benchmark numbers came from an unpreserved revision (findings.md S1-03).

**Projection, not measured:** the healthcare model has 2,914 parameters → 25 chunks at size 128 (22 full chunks plus layer remainders) → roughly 50 s of proving per client per round at 2 s per full chunk. Remainder chunks add one setup each for their sizes.

## End-to-end smoke run (healthcare, 2 rounds, 2 clients, 1 epoch)

**Run:** `results/healthcare/20260914_203124`, from:

```bash
python compare.py --dataset healthcare --modes baseline,he_tenseal_zkp,he_elgamal_zkp --rounds 2 --num-clients 2 --max-epochs 1
```

Distributed gRPC, seed 42, Apple M3 Pro, one gnark service shared by both clients and the server. It was a service left running from the previous run (PID 9383, started 20:27), so its log is `results/healthcare/20260914_202700/he_tenseal_zkp/gnark_service.log`.

**This is a pipeline check with one seed and 2 rounds.** The cost figures below are real measurements of this configuration, but they are not publication results.

### Correctness evidence

| Check | `he_elgamal_zkp` | `he_tenseal_zkp` |
|---|---|---|
| Server log per round | `aggregated 2 verified client(s), rejected 0` (rounds 1 and 2) | `7 ZKP proof(s) verified [OK]` per client (unbound: says nothing about integrity) |
| Proofs per client per round (ledger) | 26 = 21×128 + 3×64 + 1×32 + 1×2 chunks, the full server schema | 7 |
| `zkp_validation.ok` | true, 2 rounds | true, 2 rounds |
| Client errors / failures | none / 0 | none / 0 |
| Round-1 download | 11,656 B: server's plaintext initial model, as designed | CKKS from one client |
| Encrypted upload / download | 192,240 B up; 186,520 B = 2,914 × 64 B aggregate down | 3.81 MB received per round, all 6 layers aggregated as CKKS ciphertext (**B-1 fix confirmed**; the previous run received 1.29 MB, first layer only) |
| Clients decrypt and continue training | yes ("Updated model", loss falls in round 2) | yes |

### Measured cost

**Per-chunk ElGamal proving, from the service log** (104 proofs = 26 × 2 clients × 2 rounds, both clients proving at once):

| Chunk size | Proofs | First call (includes compile + setup) | Median | Min |
|---|---|---|---|---|
| 128 | 84 | 31.3 s | 3.51 s | 2.28 s |
| 64 | 12 | 16.8 s | 1.81 s | 1.14 s |
| 32 | 4 | 12.2 s | — (setup-dominated) | 0.66 s |
| 2 | 4 | 1.45 s | 0.31 s | 0.05 s |

Verification medians: 4.6 ms per 128-chunk, 2.9 ms per 64-chunk.

**Contention matters.** Median 128-chunk proving was 3.51 s with two clients proving at once, vs 2.06 s with a single caller (circuit-level table above). Both clients share one service and 11 cores.

**Per round:**

| | `he_tenseal_zkp` (unbound, all layers CKKS) | `he_elgamal_zkp` (bound) | Ratio |
|---|---|---|---|
| Measured elapsed, round-1 commit → round-2 commit | 9.3 s | 84.9 s | 9.1× |
| Proof generation per client per round, mean (includes round-1 setup) | 8.50 s | 108.9 s | 12.8× |
| Proof generation per client per round, steady state (from medians above) | ≈ 8.5 s | ≈ 80 s | ≈ 9.4× |
| Server verification per client | 0.01 s | 0.13 s | 13× |
| Server aggregation per round | 0.10 s | 0.26 s | 2.6× |
| Client decryption per download | 0.01 s | 0.12 s | 12× |
| Upload per client per round | 1,997,781 B | 192,240 B | **0.10×** (bound mode is 10× smaller) |

**Reading:**
- On this model, binding the proof to the ciphertext costs about 9× in wall-clock round time.
- It cuts upload about 10× compared with CKKS over all layers.
- One-time Groth16 setup (~61 s summed over the four chunk sizes) is paid inside the first round's `proof_generation`, which inflates round-1 means (numbers.md N-2).
- The ~50 s per client per round projection above assumed a single prover. Two clients sharing one service took ≈80 s each in steady state.

### Not valid from this run

- **Quality:** 2 rounds, 1 epoch, one seed. The summary table's `Acc(%)` column is `fl/compare/report.py::_best_accuracy()`, which returns round-mean **AUPRC** when present, so 84.0 / 74.8 / 81.4 are AUPRC. Round-mean test accuracy was 74.5 / 63.7 / 72.4. Neither supports a comparison.
- **Publication cost:** needs the standard configuration (`--rounds 20 --num-clients 3`, dataset default epochs), ≥3 seeds, setup excluded or reported separately, and a freshly started gnark service per run.

## Still open after Phase 2

| Item | Status |
|---|---|
| S1-07 bound on weights, not update | Unchanged in `he_elgamal_zkp`; Step 7 |
| S1-08 trusted setup in the proving process | Unchanged: the ElGamal circuit is also set up lazily per chunk size in the service; Step 6 |
| Client-identity binding | Not a public input: Flower's server-side client ids differ from client-side ids. Replaying another client's whole (ciphertext, proof) set in the same round still verifies. It only duplicates a norm-bounded update |
| Shared client decryption key | Same trust model as the TenSEAL composites: any key holder can decrypt any client's upload |
| Per-chunk proportional bound | Stricter than the global bound; an honest update with concentrated energy can be rejected |
| Round-1 download | Server's plaintext initial model (the server's own random initialization) |
| ModelCommit on a no-update round | `FedPrivate._chain_commit` still writes a ModelCommit when the mode returns no aggregate; Step 4 |
| S1-02/S1-04/B-3/B-4/B-5/B-6 in the old modes | Unchanged; Steps 4 and 8 |
