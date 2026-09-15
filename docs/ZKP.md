# Zero-Knowledge Proofs: Reference

This document describes what the framework proves with zero-knowledge proofs, how the proofs are produced and checked, what they guarantee, and what they do not. Every statement about behaviour refers to the current code; every number is a measurement with its setting stated.

1. [Summary](#1-summary)
2. [Background](#2-background)
3. [Threat model](#3-threat-model)
4. [The update bound](#4-the-update-bound)
5. [Circuits](#5-circuits)
6. [Modes and protocols](#6-modes-and-protocols)
7. [Keys and trusted setup](#7-keys-and-trusted-setup)
8. [The proof service](#8-the-proof-service)
9. [Python API](#9-python-api)
10. [Results, outcomes and the ledger](#10-results-outcomes-and-the-ledger)
11. [Performance](#11-performance)
12. [Configuration](#12-configuration)
13. [Limitations](#13-limitations)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Summary

Clients prove that their model update is small: ‖w_local − w_global‖₂ ≤ B, where B is a bound the server sets. The proofs are Groth16 zk-SNARKs over BN254, produced by a Go service built on [gnark](https://github.com/consensys/gnark).

| Mode | What is proven | Bound to what the server aggregates? | Integrity guarantee |
|---|---|---|---|
| `zkp` | Norm bound and MiMC hash of the quantized update, every coordinate | Yes: the server recomputes the update from the upload and its own global model | Each admitted update satisfies the bound |
| `zkp_sampled` | The same, for server-sampled coordinates after a commitment | Yes, for sampled coordinates | None beyond `zkp`: the server already sees plaintext. Benchmark of sampled proving cost only |
| `he_elgamal_zkp` | Every ciphertext encrypts an in-range value, and the encrypted update against the encrypted global model satisfies the bound | Yes: the ciphertexts and the global aggregate are public inputs | Each admitted update satisfies the bound; the server sees neither model nor update |
| `he_elgamal_zkp_sampled` | The same, for server-sampled committed coordinates | Yes, for sampled coordinates | Probabilistic: a coordinate that breaks the bound is caught if sampled |
| `he_tenseal_zkp`, `he_concrete_tfhe_zkp`, and their `_dp` variants | The plaintext statement over a client-chosen vector | **No** | **None.** Confidentiality only (see [6.5](#65-cksstfhe-composites)) |
| `pedersen` backend | Nothing is verified | — | **None.** Refused unless explicitly allowed (see [6.6](#66-pedersen-stub)) |

A bounded update can still be malicious. The bound limits how far one admitted client moves the global model per round, not in which direction (see [13](#13-limitations)).

---

## 2. Background

**Zero-knowledge proof.** A prover convinces a verifier that a statement is true without revealing anything beyond its truth. The properties used here:

- **Completeness:** an honest prover with a true statement always convinces the verifier.
- **Soundness** (knowledge soundness for SNARKs): a prover without a valid witness convinces the verifier only with negligible probability, under the scheme's assumptions and an honest trusted setup.
- **Zero knowledge:** the proof reveals nothing about the witness beyond the statement.

**Groth16.** A pairing-based zk-SNARK with constant-size proofs and constant-time verification (three pairings). Statements are arithmetic circuits compiled to rank-1 constraint systems (R1CS) over the scalar field of BN254, a prime r ≈ 2²⁵⁴. Each circuit needs its own trusted setup (see [7](#7-keys-and-trusted-setup)).

**Field arithmetic.** Circuit values are elements of the field, so an unconstrained sum of squares can wrap around r. Both circuits constrain every witness to a small integer range, which rules this out.

**MiMC.** A hash function designed for arithmetic circuits (cheap in constraints). The norm circuit publishes the MiMC hash of the vector it bounds, binding the proof to specific values.

**Exponential ElGamal on BabyJubJub.** An additively homomorphic encryption over the twisted Edwards curve defined over the BN254 scalar field, so curve arithmetic is native to the circuit. A value v encrypts as (r·G, v·G + r·PK); ciphertexts add coordinate-wise, and decryption recovers v·G, from which small v is found by a discrete-log search.

---

## 3. Threat model

- **Clients** may be malicious: they can upload arbitrary parameters, ciphertexts, proofs and metadata, run their own modified prover, and replay earlier messages.
- **The server** is honest in following the protocol (it verifies and aggregates correctly). In `he_elgamal_zkp` it is curious: it must not learn client updates or the global model.
- **All clients share one ElGamal secret key.** Confidentiality in the ElGamal modes is against the server, not between clients.
- **The trusted setup** was performed by a party that did not keep the setup randomness (see [7](#7-keys-and-trusted-setup)). Whoever ran setup could forge proofs.
- **Out of scope:** colluding clients coordinating bounded updates, the correctness of local training, availability attacks against the proof service, side channels.

---

## 4. The update bound

### 4.1 What is bounded

For every ZKP mode the statement is about the update Δ = w_local − w_global, over all model tensors (weights and buffers) flattened:

‖Δ‖₂ ≤ B (plus a rounding slack stated per circuit below).

The server chooses B and sends it to clients in the fit configuration (`zkp_max_update_norm`). A client that receives no bound refuses to prove.

The CKKS/TFHE + DP composites (`he_tenseal_zkp_dp`, `he_concrete_tfhe_zkp_dp`) do not enforce the bound: their proofs are not bound to the aggregated ciphertext ([6.5](#65-cksstfhe-composites)), so a bound would add no integrity, and DP noise makes honest updates several times larger than the non-DP calibration, so every update would be clipped. They still prove the update with per-proof declared bounds, without a model-wide total and without clipping.

### 4.2 Choosing B

```
B = PER_STEP_UPDATE_NORM[dataset] × local_epochs × max_client_batches
```

- `max_client_batches` is the largest number of batches any client has per epoch. The server computes it from the same data partition clients use (same seed, client count, validation split and Dirichlet parameter), so it knows the number of local optimiser steps without trusting clients.
- `PER_STEP_UPDATE_NORM` is calibrated per dataset with `scripts/calibrate_update_norm.py`: plain FedAvg with the harness's training settings, for several client counts and several initial models, recording ‖Δ‖ / steps for every client and round. The table stores KAPPA = 1.5 times the maximum.
- `FL_ZKP_MAX_NORM` overrides the table with an explicit update norm for one run.

Why steps and initial models: an honest update grows with the number of optimiser steps (fewer clients → larger shards → more steps per epoch), and the first rounds' updates depend strongly on the initial model. `fl.server.make_strategy` seeds the initial model with the run's seed so runs are reproducible.

**Calibrated values** (client counts 2, 3, 5; initial-model seeds 0, 1, 2, 3, 42; 3 rounds; partition seed 42; one local epoch; harness batch sizes; learning rate 0.001):

| Dataset | Per-step bound | Largest honest ‖Δ‖ (clients: 2 / 3 / 5) | Headroom B ÷ largest (2 / 3 / 5) |
|---|---|---|---|
| healthcare | 0.00379517 | 0.0683 / 0.0399 / 0.0218 | 1.50 / 1.71 / 1.91 |
| stock | 0.00270624 | 0.1002 / 0.0854 / 0.0523 | 1.95 / 1.52 / 1.50 |
| creditcard | 0.00212879 | 2.078 / 1.970 / 1.819 | 3.28 / 2.31 / 1.50 |
| mnist | 0.0129091 | 1.891 / 1.872 / 1.454 | 2.88 / 1.95 / 1.50 |
| cifar10 | 0.00389718 | 0.856 / 0.611 / 0.334 | 1.60 / 1.50 / 1.65 |

Datasets without an entry (including the legacy `cifar` key) make ZKP modes refuse to start unless `FL_ZKP_MAX_NORM` is set.

**Update norm grows sublinearly with steps.** The per-step norm is largest for configurations with few steps, so taking the maximum keeps honest clients unclipped in every calibrated configuration but leaves the bound loose where clients take many steps (up to 3.3× the largest honest update for creditcard with 2 clients). A configuration with fewer steps than any calibrated one can exceed the calibrated per-step maximum; clipping is then visible in the round outcomes. Recalibrate after changing the model, optimiser, learning rate or batch size, and calibrate DP runs separately (DP noise enlarges honest updates).

### 4.3 Clipping

Honest clients clip before proving, so a correct client is never rejected by the bound:

- `zkp`, `zkp_sampled` (`fl/privacy/zkp.py::clip_update_in_place`): Δ is scaled to 0.999·B, and the exact integer statement the server will check is tested before proving, shrinking further if rounding would push it over. The clipped weights are what the client uploads.
- `he_elgamal_zkp`, `he_elgamal_zkp_sampled` (`fl/core/elgamal_gnark.py::quantize_update`): the quantized values are rounded around the global model, so the clipped update always fits the circuit's slack.

Each client reports `zkp_update_norm` (before clipping) and `zkp_update_clipped`; the server records them per round.

### 4.4 Declared per-proof bounds

A model is proven in chunks, each with its own public bound. Clients declare each chunk's bound as that chunk's exact energy (at least 1); the circuit enforces each declared bound, and the server rejects a client whose declared bounds sum to more than the model-wide total. This bounds the whole update without rejecting honest updates whose energy is concentrated in a few chunks. In `he_elgamal_zkp` the declared bounds are visible to the server (see [13](#13-limitations)).

---

## 5. Circuits

Both circuits have a fixed size fixed by the pinned keys. Shorter inputs are padded with values that contribute nothing to the bound.

### 5.1 Norm circuit (`zkp_gnark_service/main.go`, n = 256)

Public inputs: `Bound`, `Hash`. Witness: `Weights[256]`.

Constraints:
1. each `w` is a signed 64-bit integer (range check of w + 2⁶³ in 64 bits);
2. `MiMC(Weights) = Hash`;
3. Σ w² ≤ `Bound`.

The range check means Σ w² ≤ 256 · 2¹²⁶ < r, so the sum cannot wrap the field. Without it, a prover whose hash is not recomputed by the verifier could choose two large field elements whose squares sum to a small value modulo r.

What is proven in each mode: the witness is the quantized update of one chunk, Δq = round(w·s) − round(g·s) with s = `FL_ZKP_SCALE` (10⁶), for up to 256 consecutive values of one tensor. Padding is zeros, and the hash covers the padded vector.

Model-wide bound checked by the server: Σ declared bounds ≤ ⌈B·s + √n⌉², where n is the number of coordinates proven. The √n term absorbs rounding (each Δq coordinate is within 1 of s·Δ), so any ‖Δ‖ ≤ B fits. The admitted update therefore satisfies ‖Δ‖ ≤ B + √n / s.

Size: **103,365 constraints**.

### 5.2 ElGamal update circuit (`zkp_gnark_service/elgamal.go`, n = 128)

Public inputs: `PK`, `Bound`, `Context`, `Weight` (W), the client's ciphertexts `C1[128], C2[128]`, and the global model at the same slots `G1[128], G2[128]`.
Witness: `Values[128]` (v = q + 2¹⁷), `Rand[128]`, `Agg[128]` (T), `SK`.

Constraints:
1. `SK·G = PK`, so the witness key is the key PK belongs to;
2. W < 2¹⁴ (range check);
3. for each slot: v < 2¹⁸ (range check that doubles as the scalar decomposition), `C1 = r·G`, `C2 = v·G + r·PK`;
4. for each slot: T < 2³² and `G2 = T·G + SK·G1`, so T is the unique plaintext of the global slot;
5. Σ (W·v − T)² ≤ `Bound`, which equals Σ (W·q − S)² because the offsets cancel (T = S + W·2¹⁷, S = Σ nₖ·qₖ).

`Context` is (round << 32) | (layer << 16) | chunk and is bound by a constraint, so a proof cannot be replayed at another round or position.

The global model is the server's previous aggregate: G = Σ nₖ·Cₖ with W = Σ nₖ. Before the first aggregation it is the server's public initial model, encoded as G = (identity, (q + 2¹⁷)·G) with W = 1. The server passes the ciphertexts it holds; only clients, which hold the shared secret key, know S.

Padding: client slots are (identity, 2¹⁷·G) with v = 2¹⁷, r = 0; global slots are (identity, W·2¹⁷·G) with T = W·2¹⁷. Both sides rebuild them, and their difference term is 0.

Values: q = round(w·s) with s = `FL_ELGAMAL_SCALE` (10⁴) and |q| < 2¹⁷, so |w| < 13.1.

Model-wide bound checked by the server: Σ declared bounds ≤ W²·⌈B·s + √n/2⌉². The admitted update satisfies ‖q/s − S/(W·s)‖ ≤ B + √n / (2s).

Size: **1,274,949 constraints**.

---

## 6. Modes and protocols

### 6.1 `zkp`

**Client, per round**
1. Receive the global model g (it is kept as the base of the update).
2. Train locally.
3. Clip Δ to B ([4.3](#43-clipping)); compute Δq per tensor.
4. Split each tensor's Δq into chunks of 256, declare each chunk's bound, and request a proof from the prover service. The client checks that the prover answered under the pinned verifying key.
5. Upload the clipped weights with the proofs in the fit metrics (`zkp_proofs_json`).

**Server, per round**, for each client:
1. Check proof coverage against its **own** model schema: exactly one proof per expected tensor or chunk, correct shapes and scale, every proof under the pinned verifying key, declared bounds positive and summing to at most the total.
2. Recompute Δq from the uploaded parameters and its own global model.
3. Ask the verifier service to verify each proof with the hash recomputed from Δq.

Admitted clients are averaged (FedAvg); the aggregate becomes the global model. If fewer clients are admitted than `min_fit_clients`, the global model is unchanged (`no_quorum`). If the verifier is unreachable or fails, the round is aborted (`infrastructure_abort`), and no client is blamed.

### 6.2 `zkp_sampled`

Two Flower rounds per federated round (commit–challenge):

- **Commit (odd rounds):** clients train, clip, and upload their full weights without proofs. The server stores each commitment together with the global model at that time.
- **Challenge (even rounds):** only now does the server draw a fresh 256-bit seed and send it. Clients prove the sampled coordinates of the Δq they committed; the server recomputes those coordinates from the commitment and verifies. Clients that pass are aggregated.

Sample size is s = ⌈rate · n⌉ with rate `FL_ZKP_SAMPLE_PCT` (default 0.1). Indices come from a partial Fisher–Yates shuffle driven by SHA-256 of the seed, so anyone with the seed can reproduce them; the seed is recorded in the round outcome. The total bound uses the sample size.

This mode has no security value: the server receives plaintext weights and can check the update norm directly. It exists to compare sampled against full proving cost.

A client that has no commitment for a challenge (for example one that connected late) answers with no proofs and is rejected; a client that committed but does not answer is rejected too.

### 6.3 `he_elgamal_zkp`

**Client, per round**
1. Download the global model. In round 1 this is the server's plaintext initial model; afterwards it is the encrypted aggregate, which the client decrypts with the shared secret key, keeping the sums S and weight W.
2. Train locally.
3. Quantize and clip the update around the global model ([4.3](#43-clipping)).
4. For each chunk of 128 coordinates of each tensor, declare the bound and request `/elgamal/prove` with the values, the global slots and their sums. The service encrypts with fresh randomness and proves the chunk statement.
5. Upload the ciphertexts (64 bytes per coordinate) with a header and the proofs.

**Server, per round**, for each client:
1. Check the header and ciphertext sizes against its own schema, proof coverage, the pinned verifying key, and the declared bounds against W²·⌈B·s + √n/2⌉².
2. Verify each chunk proof against the received ciphertexts, the global aggregate it holds at those slots, and W.

Admitted ciphertexts are aggregated homomorphically with weights nₖ (the clients' reported batch counts); the result becomes the next global model. The total weight must stay below 2¹⁴; otherwise the round is aborted, because no client could prove against that aggregate. The same no-quorum and abort rules as `zkp` apply.

The server never holds the secret key, the global model or any update in plaintext.

### 6.4 `he_elgamal_zkp_sampled`

Commit–challenge as in [6.2](#62-zkp_sampled), over ElGamal:

- **Commit:** clients clip, quantize and encrypt every coordinate, keeping the values, the encryption randomness and the global model they measured against. They upload ciphertexts only.
- **Challenge:** after the seed, clients prove each chunk of sampled coordinates with `/elgamal/prove_with`, using the stored randomness, so the proof only verifies against the committed ciphertexts. The server verifies against the commitments and the global aggregate it held at commit time.

A coordinate that makes the bound impossible, or whose committed ciphertext differs from what the client later proves, is detected if sampled. With m such coordinates out of n and s sampled, the detection probability is 1 − C(n−m, s)/C(n, s) ≥ 1 − (1 − s/n)^m (`fl.core.sampling.detection_probability`). Unsampled coordinates carry no proof. An out-of-range unsampled ciphertext makes aggregate decryption fail for every client: a denial of service that is only detected after aggregation.

### 6.5 CKKS/TFHE composites

`he_tenseal_zkp`, `he_concrete_tfhe_zkp` and their `_dp` variants run the `zkp` client over the plaintext update and then encrypt with CKKS or TFHE. The server cannot recompute the update from ciphertexts, so it uses **light verification**: it checks the proofs against the client-supplied hash (`/verify_light`) plus coverage and the total bound.

Nothing ties that hash to the ciphertext the server aggregates. A client can prove an honest update and upload the encryption of a different one. These modes provide confidentiality only; their ZKP adds cost and no integrity. The `_dp` variants do not enforce the update bound at all ([4.1](#41-what-is-bounded)).

### 6.6 Pedersen stub

`--zkp_backend pedersen` computes Pedersen commitments locally and sends nothing to verify. It is refused unless `FL_ZKP_ALLOW_PEDERSEN_STUB=1`; when allowed, every round is recorded with outcome `unverified_stub`, and the run's `zkp_backend` is recorded as `pedersen`.

---

## 7. Keys and trusted setup

### 7.1 Setup

```bash
cd zkp_gnark_service && go build -o gnark_service . && cd ..
zkp_gnark_service/gnark_service setup --keys-dir zkp_gnark_service/keys --pk-dir ~/.cache/fl_ppml/gnark_pk [--norm-n 256] [--elgamal-n 128] [--force]
```

`setup` compiles both circuits, runs Groth16 setup once for each, and writes:

- `keys/<circuit>-<n>.vk`: verifying keys, committed to the repository;
- `keys/manifest.json`: for each circuit its size, constraint count, SHA-256 of the verifying and proving keys, the gnark version, the date and a statement of the setup model;
- `<pk-dir>/<circuit>-<n>.pk`: proving keys (hundreds of MB), written with mode 0600 and not committed.

It refuses to overwrite existing keys without `--force`. Regenerating keys re-pins the verifying keys: proofs made under old keys no longer verify, and the new `keys/` directory must be committed.

### 7.2 Roles and pinning

- The **prover** loads proving keys (hash-checked against the manifest) and compiles the circuits.
- The **verifier** loads only verifying keys (hash-checked) and never receives `--pk-dir`.

Either role refuses to start if a file is missing or its hash differs from the manifest, and neither runs setup. Every proof payload carries the SHA-256 of its verifying key. The server rejects a proof made under any other key, and sends the pinned hash (never the client's claim) with every verification request; a verifier holding different keys answers HTTP 503, which aborts the round. Each round outcome records the manifest hash.

The harness (`compare.py`) starts both roles, and reuses a running service only if its `/health` reports the expected role and manifest hash.

### 7.3 What a single-party setup does and does not give

Guaranteed:
- every prover and verifier uses the same published verifying key, identified by hash;
- proofs remain verifiable after restarts and by independent verifiers; anyone with the committed `.vk` can re-check a stored proof;
- the verifier process never holds proving keys or setup randomness.

Not guaranteed: that the setup randomness (τ, α, β, γ, δ) was destroyed. It existed in the memory of the process that ran `setup`. Whoever controlled that process could forge proofs for false statements that every verifier accepts. Soundness holds against provers who did not run setup.

A multi-party ceremony would change this: a universal Powers-of-Tau phase followed by a per-circuit phase in which several independent parties contribute makes forgery require all contributors to collude. gnark provides MPC setup support (`backend/groth16/bn254/mpcsetup`); this framework does not use it.

---

## 8. The proof service

### 8.1 Running it

```bash
zkp_gnark_service/gnark_service serve --role prover   --keys-dir zkp_gnark_service/keys --pk-dir ~/.cache/fl_ppml/gnark_pk --port 9000
zkp_gnark_service/gnark_service serve --role verifier --keys-dir zkp_gnark_service/keys --port 9001
zkp_gnark_service/gnark_service elgamal-keygen keys/he_elgamal/secret_key.json keys/he_elgamal/public_key.json
```

`compare.py` starts both roles automatically. `python -m fl.keys generate he_elgamal` wraps `elgamal-keygen`.

### 8.2 Endpoints

| Endpoint | Role | Purpose |
|---|---|---|
| `GET /health` | both | `status`, `service`, `role`, `manifest_sha256`, and for each circuit `n` and `vk_sha256` |
| `POST /prove` | prover | Norm proof |
| `POST /verify` | verifier | Norm proof, hash recomputed from the values sent |
| `POST /verify_light` | verifier | Norm proof against a supplied hash |
| `POST /elgamal/prove` | prover | Encrypt with fresh randomness and prove a chunk |
| `POST /elgamal/prove_with` | prover | Prove a chunk for given values and randomness (commit–challenge) |
| `POST /elgamal/encrypt` | prover | Encrypt values, returning ciphertexts and randomness |
| `POST /elgamal/decrypt` | prover | Decrypt aggregate sums |
| `GET /elgamal/info?n=` | prover | Constraint counts for a chunk size (compiles the circuit) |
| `POST /elgamal/verify` | verifier | Verify a chunk proof |
| `POST /elgamal/aggregate` | verifier | Weighted homomorphic sum of client ciphertexts |

Integers are little-endian int64, base64-encoded. Field integers (`bound_sq`, `context`, `sk`) are decimal strings. Points are 32-byte compressed encodings; public keys are hex. Every verification request must include `vk_sha256`.

**`/prove`** request: `layer_name`, `weights_b64`, `shape`, `scale`, `bound_sq`. Response: `proof_b64`, `hash_hex`, `vk_sha256`, `circuit_n`.

**`/verify`** request: the same plus `proof_b64` and `vk_sha256`. Response: `verified`.

**`/verify_light`** request: `layer_name`, `shape`, `bound_sq`, `hash_hex` (canonical field element), `proof_b64`, `vk_sha256`. Response: `verified`.

**`/elgamal/prove`** request: `pk`, `sk`, `values_b64`, `bound_sq`, `context`, and the global model at the same slots: either `global_ct_b64` + `global_weight` + `global_sums_b64` (aggregate), or `global_plain_b64` (initial model, weight 1). `/elgamal/prove_with` adds `rand_b64` (32-byte big-endian scalars). Response: `ct_b64`, `proof_b64`, `vk_sha256`.

**`/elgamal/verify`** request: `pk`, `ct_b64`, `bound_sq`, `context`, `proof_b64`, `vk_sha256`, and `global_ct_b64` + `global_weight` or `global_plain_b64` (no sums). Response: `verified`.

**`/elgamal/aggregate`** request: `cts_b64` (one per client), `weights`. Refuses a total weight ≥ 2¹⁴. Response: `ct_b64`.

**`/elgamal/decrypt`** request: `sk`, `ct_b64`, `offset_total`, `max_abs`. Response: `values`.

### 8.3 Status codes and how the framework treats them

| Status | Meaning | Framework behaviour |
|---|---|---|
| 200 | Request processed; for verification, see `verified` | Admit only if `verified` is true |
| 400 | Malformed request, wrong size, value out of range | During verification: the client is rejected |
| 422 | Statement not satisfied (prove), or refused value | During proving: the client's upload fails |
| 503 | The service's keys differ from the requested `vk_sha256` | Round aborted (`infrastructure_abort`) |
| other 5xx, timeout, connection error | Service failure | Round aborted (`infrastructure_abort`) |

---

## 9. Python API

**`fl.core.zkp_gnark`** (norm circuit)
- `generate_gnark_proofs(state_dict, layers=None, service_url=None, scale=None, total_bound_sq=None, timeout=None) -> (proofs, proof_bytes)`: integer arrays are proven as given; float arrays are quantized. Each proof declares its own energy as its bound; with `total_bound_sq`, their sum must fit.
- `verify_gnark_proofs(parameters, layer_names, proofs) -> (ok, failures)`: recomputes each hash from `parameters`.
- `verify_gnark_proofs_light(proofs) -> (ok, failures)`: verifies against the proofs' own hashes (not bound to anything the caller holds).
- `check_proof_policy(proofs, schema, *, require_hash, total_bound_sq, scale=None) -> reason | None`
- `policy_bound_sq(max_update_norm, n, scale=None)`, `quantize(values)`, `energy(q)`, `expected_proof_layout(schema)`
- `GnarkServiceError`: raised for infrastructure failures, never for client faults.

**`fl.core.elgamal_gnark`** (ElGamal circuit)
- `GlobalModel.initial(arrays, scale)`, `GlobalModel.aggregate(weight, layers, sums=None)`, `GlobalModel.request(indices, prover=...)`
- `quantize_update(glob, local_flat, max_update_norm, scale) -> (q, norm, clipped)`
- `total_bound_sq(max_update_norm, scale, weight, n)`, `update_energy(q, sums, weight)`
- `prove_chunk`, `prove_with`, `verify_chunk`, `encrypt_values`, `aggregate`, `decrypt`
- `Policy.from_env()`, `chunks_for(schema, chunk_size)`, `chunk_indices(schema, chunk)`, `context_value(round, chunk)`
- `ElGamalServiceError` (infrastructure), `ElGamalRejected` (the service refused the contents)

**`fl.core.update_bound`**: `max_update_norm(config)`, `bound_from_fit_config(fit_config)`, `clip_update(global, local, bound, fits)`, `split_bound(energies, total)`, `PER_STEP_UPDATE_NORM`, `KAPPA`.

**`fl.core.gnark_keys`**: `load_manifest()`, `manifest_sha256()`, `circuit_size(circuit)`, `pinned_vk_sha256(circuit)`, `missing_proving_keys()`.

**`fl.core.sampling`**: `sample_rate_from_env()`, `sample_size(n, rate)`, `new_round_seed()`, `sample_indices(seed_hex, n, s)`, `detection_probability(n, s, m)`.

**Modes** (`fl.privacy`): `ZKPMode`, `ZKPSampledMode`, `HeElGamalZKPMode`, `HeElGamalZKPSampledMode`; the commit–challenge protocol is `fl.privacy.commit_challenge.CommitChallengeMixin`.

---

## 10. Results, outcomes and the ledger

Each run's `benchmark.json` records:

- `transport`: `network` (real server and client processes) or `simulated` (in-process Flower simulation, where HE modes transport plaintext; such results are marked `[SIM]` in reports and never replace a networked result in the dataset report);
- `zkp_backend`: `gnark`, `pedersen`, or `null` for modes without ZKP;
- `round_outcomes`, one entry per Flower round:

| Field | Meaning |
|---|---|
| `outcome` | `aggregated`, `committed` (commit round), `no_quorum`, `infrastructure_abort`, `unverified_stub` |
| `admitted`, `rejected` | client ids; rejection reasons per client |
| `flower_failures` | clients whose fit call failed |
| `key_manifest_sha256` | the pinned keys the proofs were checked under |
| `update_norms`, `clipped` | each client's update norm before clipping, and which clients were clipped |
| `sample_seed`, `sampled_coordinates` | sampled modes: the challenge seed and sample size |

`compare.py` validates ZKP runs: a run with rejected clients, aborted rounds, Flower failures or missing outcomes is reported as failed.

The audit ledger (`fl/chain.py`) writes a `ModelCommit` for every round that updated the model and a `ProofAnchor` with SHA-256 hashes of the admitted proof payloads. Payloads include the verifying-key hash and declared bounds, so an anchored proof can be re-verified with the committed verifying key. Nothing is written for rounds that did not update the model.

---

## 11. Performance

### 11.1 Single proofs

Apple M3 Pro, 18 GB RAM, services warm, one proof at a time, production keys (2026-09-15):

| Circuit | Constraints | Prove (median) | Verify (median) | Proof size |
|---|---|---|---|---|
| Norm, n = 256 | 103,365 | 0.65 s | 2.3 ms (light), 3.6 ms (hash recomputed) | 220 base64 characters |
| ElGamal, n = 128 | 1,274,949 | 3.13 s | 7.4 ms | 220 base64 characters, plus 64 bytes of ciphertext per coordinate |

Setup (same machine): norm 5.0 s, proving key 36.1 MB, verifying key 520 B; ElGamal 50.5 s, proving key 420.2 MB, verifying key 33.4 KB, peak memory about 2.4 GB.

### 11.2 Proofs per client per round

- `zkp`: Σ over tensors of ⌈numel / 256⌉. Healthcare model (2,914 parameters): 15 proofs.
- `he_elgamal_zkp`: Σ over tensors of ⌈numel / 128⌉. Healthcare: 26 proofs.
- Sampled modes: ⌈s / 256⌉ or ⌈s / 128⌉ for s = ⌈rate · n⌉ sampled coordinates.

Proofs are generated one at a time per client by default (`FL_ZKP_PARALLELISM=1`); each proof already uses several cores inside the prover.

### 11.3 End to end

Healthcare, 3 clients proving concurrently on one machine (the M3 Pro above), 3 rounds, 1 local epoch, measured with the norm circuit before its int64 range check was added (86,725 constraints; per-proof prove time then 0.64 s, now 0.65 s):

| Mode | Proof generation per client per round | Verification per client per round |
|---|---|---|
| `zkp` | 15.4 s | 0.06 s |
| `zkp_sampled` (10 %) | 2.1 s | 0.01 s |
| `he_elgamal_zkp` | 227 s | 0.21 s |
| `he_elgamal_zkp_sampled` (10 %) | 24.0 s | 0.02 s |

Concurrent proving on shared cores is slower per proof than the single-proof figures (about 7.4 s per ElGamal proof here).

---

## 12. Configuration

| Variable | Default | Description |
|---|---|---|
| `FL_ZKP_BACKEND` | `gnark` | `gnark`, or `pedersen` (unverified stub) |
| `FL_ZKP_ALLOW_PEDERSEN_STUB` | unset | `1` allows the Pedersen stub |
| `FL_ZKP_MAX_NORM` | calibrated | Update-norm bound B, overriding the per-dataset calibration |
| `FL_ZKP_SCALE` | `1000000` | Norm-circuit quantization scale |
| `FL_ELGAMAL_SCALE` | `10000` | ElGamal quantization scale; \|w\| · scale < 2¹⁷ |
| `FL_ZKP_SAMPLE_PCT` | `0.1` | Sampled modes: fraction of coordinates proven, in (0, 1] |
| `FL_ZKP_PARALLELISM` | `1` | Proofs generated concurrently per client |
| `FL_ZKP_LAYERS` | `ALL` | Tensors to prove. Anything but `ALL` fails the server's coverage check |
| `FL_ZKP_PROVER_URL` | `http://127.0.0.1:9000` | Prover service |
| `FL_ZKP_VERIFIER_URL` | `http://127.0.0.1:9001` | Verifier service |
| `FL_ZKP_KEYS_DIR` | `zkp_gnark_service/keys` | Manifest and verifying keys |
| `FL_ZKP_PK_DIR` | `~/.cache/fl_ppml/gnark_pk` | Proving keys (prover only) |
| `FL_ZKP_TIMEOUT` | `600` | Fallback HTTP timeout, seconds |
| `FL_ZKP_PROVE_TIMEOUT` | `1800` | Prove request timeout |
| `FL_ZKP_VERIFY_TIMEOUT`, `FL_ZKP_VERIFY_LIGHT_TIMEOUT` | `900` | Verify request timeouts |
| `FL_ZKP_MAX_RSS_MB` | 70 % of RAM | Client memory guard during proof generation |
| `FL_ZKP_AGGRESSIVE_GC`, `FL_ZKP_GC_SLEEP_MS` | `1`, `0` | Garbage collection between proofs |
| `FL_CLIENT_WAIT_TIMEOUT` | `600` | Seconds the server waits for `min_avail_clients` before stopping the run |
| `FL_SERVER_GRACE` | `600` | Harness: seconds a server may run after all clients exited (60 s if any client failed) |

---

## 13. Limitations

- **Direction is not bounded.** An admitted client can move the global model by up to (nₖ / N) · (B + slack) per round in any direction. A clipped malicious update is admitted; robust aggregation is not part of this framework.
- **The bound is calibrated, not derived.** It is loose where clients take many steps, and can clip honest clients in configurations with fewer steps than any calibrated one ([4.2](#42-choosing-b)).
- **Single-party trusted setup** ([7.3](#73-what-a-single-party-setup-does-and-does-not-give)).
- **Shared client key.** In the ElGamal modes every client can decrypt every aggregate.
- **Per-chunk leakage in `he_elgamal_zkp`.** Declared chunk bounds are public, so the server learns the squared norm of the update restricted to each chunk of 128 coordinates.
- **Sampled modes** prove nothing about unsampled coordinates.
- **CKKS/TFHE composites** provide no integrity ([6.5](#65-cksstfhe-composites)).
- **Client identity** is not a public input: proofs are bound to a round, position, ciphertext and global model, not to a client.
- **Not post-quantum.** BN254 pairings and BabyJubJub discrete logarithms fall to a large quantum computer.
- **Honest training is not proven.** A bounded update may come from any data or procedure.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No pinned ZKP key manifest` | Keys never generated | Run `gnark_service setup` ([7.1](#71-setup)) |
| `Proving keys [...] not in ...` | Fresh checkout: proving keys are not committed | Run `setup --force` and commit the new `keys/` |
| Service exits at start with a hash mismatch | Key files differ from the manifest | Restore the committed `keys/` or regenerate with `setup --force` |
| HTTP 503 on verification, round `infrastructure_abort` | Verifier started from other keys | Restart the services from the committed `keys/` |
| `no calibrated ZKP update-norm bound for dataset` | Dataset missing from `PER_STEP_UPDATE_NORM` | Run `scripts/calibrate_update_norm.py` and add the value, or set `FL_ZKP_MAX_NORM` |
| Many clients listed in `clipped` | Configuration takes fewer steps than calibrated, or DP noise | Recalibrate for the configuration, or set `FL_ZKP_MAX_NORM` |
| `declared update bounds sum to ... above the server's bound` | The client did not clip (not the framework client), or client and server use different bounds | Clients must use the bound from the server's fit configuration |
| `|w|·scale must be < 131072` | ElGamal: a weight too large for the scale | Lower `FL_ELGAMAL_SCALE` |
| `total weight ... ≥ 16384` | ElGamal: clients' batch counts sum too high | Fewer clients or larger batches |
| `ZKP memory guard triggered` | Client RSS above the guard | `FL_ZKP_PARALLELISM=1`, or raise `FL_ZKP_MAX_RSS_MB` |
| `round N fit: only k of m required clients available` | Clients did not connect within `FL_CLIENT_WAIT_TIMEOUT` | Check client logs, or raise the timeout |
