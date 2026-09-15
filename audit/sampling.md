# `zkp_sampled`: what it actually does

Date: 2026-09-14
Scope: Step 5 Phase 1. `ZKPSampledMode._select_layers` and `_generate_proofs` in `fl/privacy/zkp.py`, how the harness and server treat the mode, and the claims made about it.

**No source files were changed for this report.** Measurements come from [`audit/evidence/sampling_evidence.py`](evidence/sampling_evidence.py), run against the current code:

```bash
PYTHONPATH=. python audit/evidence/sampling_evidence.py
```

## Hypothesis: confirmed

> `ZKPSampledMode` does not do what its name and docstring say.

The docstring describes a configurable fraction of layers, selected randomly with an optional reproducible seed that "in production should be derived from a verifiable round seed". The code does something else:
- Under the defaults it deterministically proves **exactly one layer: the largest**.
- The fraction setting has no effect.
- The seed and RNG never run.
- The selection is made by the client, from data the client controls.

## Q1 — The `k` computation: `FL_ZKP_SAMPLE_PCT` is unreachable

`fl/privacy/zkp.py` (`ZKPSampledMode._select_layers`):

```python
pct = os.environ.get("FL_ZKP_SAMPLE_PCT")
num = os.environ.get("FL_ZKP_NUM_LAYERS", "1")   # default is the string "1", never None
if num is not None:                               # always true
    k = max(1, int(num))
elif pct is not None:                             # dead
    ...
else:                                             # dead: the "20% default" never happens
    ...
```

Measured on a 7-layer state dict:

| Environment | Layers selected |
|---|---|
| none | 1 of 7 |
| `FL_ZKP_SAMPLE_PCT=0.5` | 1 of 7 |
| `FL_ZKP_SAMPLE_PCT=1.0` | 1 of 7 |
| `FL_ZKP_NUM_LAYERS=3` | 3 of 7 |

`tests/test_zkp_sampled.py::test_pct_sampling_env` fails for this reason (`assert 1 == 4`).

**Does a sweep over sampling percentage exist?** No. None of the sampling variables (`FL_ZKP_SAMPLE_PCT`, `FL_ZKP_NUM_LAYERS`, `FL_ZKP_SAMPLE_SEED`, `FL_ZKP_SELECT_BY`) is set anywhere in the harness, scripts, or stored results; they appear only in code, tests, README and docs. A sweep would have produced identical runs, but none was run. More fundamentally, **the benchmark never instantiated `ZKPSampledMode`**: the harness maps `zkp_sampled` to `--zkp`, which resolves to `ZKPMode`. Every stored `zkp_sampled` row is a second run of full ZKP with identical proof counts (audit/numbers.md item 1, findings.md S1-06 correction).

## Q2 — `select_by` and the RNG

`FL_ZKP_SELECT_BY` defaults to `"size"`, which sorts layers by element count and takes the top `k`. The RNG built from `FL_ZKP_SAMPLE_SEED` is used only on the `"random"` branch.

| Measurement (20-layer state dict, 20 calls each) | Result |
|---|---|
| Default | **1 distinct selection** (`layer19`, the largest), every call |
| Default with `FL_ZKP_SAMPLE_SEED` = 1, 2, 3 | `layer19` for every seed: the seed has no effect |
| `FL_ZKP_SELECT_BY=random` | 14 distinct selections, but seeded by nothing unless the client sets the seed |

**Additional finding (S3): `tests/test_zkp_sampled.py::test_seed_determinism` is vacuous.** It sets a percentage and a seed and asserts two calls agree. Under the default `size` strategy, neither variable is read, so the test passes whether or not seeded sampling works.

Even the `random` branch provides no security:
- The RNG runs on the client.
- It's seeded by a client environment variable, or by nothing at all.
- The server is never told which layers were selected. It only sees the proofs the client chose to send.

## Q3 — What fraction the default selection covers, per dataset

Model sizes are reconstructed from the stored baseline upload bytes (4 B per parameter). The script checks that each rebuilt model has exactly that many parameters.

| Dataset | Parameters | Layer proven by default | Share of parameters proven | Proofs (sampled / full, 2000-element chunks) | Share of proof work skipped |
|---|---:|---|---:|---:|---:|
| healthcare | 2,914 | `model.2.weight` | 70.3% | 2 / 7 | 29.7% |
| creditcard | 4,130 | `model.2.weight` | 49.6% | 2 / 7 | 50.4% |
| stock | 2,914 | `model.2.weight` | 70.3% | 2 / 7 | 29.7% |
| mnist | 44,426 | `fc1.weight` | 69.1% | 16 / 31 | 30.9% |
| cifar | 62,006 | `fc1.weight` | 77.4% | 24 / 39 | 22.6% |

The circuit cost is linear in proven elements: ≈346 constraints per element, measured in audit/binding.md. So proof work falls roughly in proportion to the "share of proof work skipped" column. **Sampling could save 23–50% of proving**, not the order-of-magnitude reduction the name suggests. In return it leaves 23–50% of every model unproven.

## Q4 — Can a client predict the proven layer? Yes, completely.

The selection depends only on tensor shapes, which are fixed by the architecture and identical for every client in every round. Measured on the healthcare model:
- selected before training: `['model.2.weight']`
- after multiplying **every unselected layer by 1000**: `['model.2.weight']`

So the selection is known before any update exists and can't be changed by the update. A client can put an arbitrarily large or crafted update into `model.0.*`, `model.2.bias` and `model.4.*` (29.7% of healthcare parameters, 50.4% on creditcard) and still produce a valid proof set for the one proven layer.

**What this does to the claim that the mode "prevents gradient poisoning (Bagdasaryan et al. 2020)."**

That sentence is in the thesis text, which isn't in this repository; the closest repo claims are listed below. The claim is contradicted on four independent grounds:
1. **Unproven layers are unconstrained.** The attacker knows which layers they are before training.
2. **The bound is on the weights, not the update** (findings.md S1-07). Even the proven layer can carry a large update while its final values stay inside the bound.
3. **The selection is the client's.** In the random branch the client chooses the seed and the server never learns the selection, so a client can simply exclude the layer it poisons.
4. **Model replacement doesn't need every layer.** The Bagdasaryan et al. attack scales a backdoored update to override the aggregate; a backdoor planted in unproven layers passes this check as is.

**Detection probability against an adversary who poisons unselected layers, under the current default: 0.**

**Current state after Step 4:** the server now requires proofs covering every layer of its own schema (`check_proof_policy`). A real `ZKPSampledMode` upload is therefore *rejected*, not silently admitted. The mode is unusable, but it no longer weakens a guarantee silently.

## Claims contradicted

| Location | Claim | Contradicted by |
|---|---|---|
| `fl/privacy/zkp.py` `ZKPSampledMode` docstring | fraction of layers via `FL_ZKP_SAMPLE_PCT`; seeded sampling | Q1, Q2 |
| `README.md:45`, `docs/README.md:33` | "Groth16 zk-SNARK, sampled layers — Gradient integrity (Byzantine clients)" | Q4 |
| `docs/README.md:356` | `zkp_sampled` "✅ Working … sampled layers" | never instantiated by the harness; now rejected by the server |
| `README.md:458-459`, `docs/README.md:406-407` | `FL_ZKP_SAMPLE_PCT` "fraction of layers to prove"; `FL_ZKP_SAMPLE_SEED` "omit to vary across rounds" | Q1: the percentage has no effect; Q2: the seed has no effect under the default, and nothing varies across rounds |
| `docs/FL.md:349` | "ZKP sampled coordinates = 100" | sampling is per layer, not per coordinate; there is no 100-coordinate setting |
| `docs/FL.md:569` | sampled mode gives soundness against Byzantine clients with probability 1 − 2⁻¹²⁸ | Groth16 soundness covers only what is proven; unproven layers have detection probability 0 (Q4) |
| `docs/FL.md:575` | "sampling is seeded by round and client ID (deterministically but unpredictably to the clients)" | no round or client seed exists; the selection is fully predictable (Q2, Q4) |
| `docs/README.md:329`, `README.md:428`, `docs/FL.md:376` | timing and accuracy figures for `zkp_sampled` | every stored `zkp_sampled` row is full ZKP (numbers.md) |
| thesis text (not in repo) | "prevents gradient poisoning (Bagdasaryan et al. 2020)" | Q4 grounds 1–4 |

## Design question before Phase 2 (needs a decision)

The Phase 2 requirements are:
- a server-generated round seed pushed via `configure_fit`
- coordinate-level sampling across the whole model
- a working percentage setting
- reproducible server-side selection

They can be implemented as written, but **which mode sampling belongs to** changes whether the result means anything:

1. **Plaintext `zkp`: sampling adds no security value.**
   - The server already receives every plaintext weight and recomputes each proof's hash from them (`verify_gnark_proofs`).
   - So it can compute the exact norm of the full vector itself, at no cost and without any proof.
   - A sampled proof in this mode costs the client prover time to show the server something it can check directly.
   - Sampling would be correct here, but the resulting detection-probability curve would describe a mechanism nobody should deploy.
2. **`he_elgamal_zkp`: the mode where sampling is meaningful.**
   - The server can't see the values, so proofs are its only evidence, and the proofs are bound to the ciphertexts.
   - Sampling there proves range, encryption and norm for a server-chosen, unpredictable subset of ciphertexts.
   - Unsampled ciphertexts are unconstrained, which is exactly the adversary Step 5 Phase 3 asks to analyse.
   - Two consequences need design:
     - **An out-of-range value in an unsampled ciphertext** would make the clients' bounded discrete-log decryption fail for the whole aggregate. That's a denial of service, detected only after aggregation.
     - **Per-chunk norm bounds** need rethinking when only sampled coordinates are proven.
3. **The CKKS/TFHE composites: sampling there measures nothing.** Their proofs aren't bound to the ciphertext at all (S1-01).

**Recommendation:** implement server-seeded coordinate sampling for `he_elgamal_zkp` (sampled proofs over the ciphertexts at server-chosen indices), and do the detection-probability analysis for that mode. Give plaintext `zkp_sampled` the same selection mechanism only for completeness of the benchmark, labelled as offering no security benefit over the server checking the norm itself. Alternatively, remove `zkp_sampled` and state why.

**Stopping here, as Step 5 Phase 1 requires.**

---

# Phase 2 — commit–challenge sampling (implemented)

**Decisions:** sampling is implemented for `he_elgamal_zkp_sampled`, and mirrored in plaintext `zkp_sampled`, labelled benchmark-only. The protocol is **design C, commit then challenge**.

The requirement to push a seed in `configure_fit` conflicts with the requirement that clients can't predict the selection before computing their update: Flower delivers the fit config together with the model, before training. A seed sent that way lets a client see the selection first, giving detection probability 0 against an adaptive client. Design C sends the seed only after the update is committed.

## Protocol

Each federated round takes two Flower rounds (`fl/privacy/commit_challenge.py`):

| Flower round | Server | Client |
|---|---|---|
| odd: **commit** | sends the global model with `zkp_phase=commit`; stores each well-formed upload; no aggregation, no ledger entry; outcome `committed` | downloads, trains, uploads the full update with no proofs (ElGamal: ciphertexts of every coordinate, randomness kept locally) |
| even: **challenge** | draws a fresh 256-bit seed in `configure_fit` (after the commits are stored) and sends `zkp_phase=challenge`, the seed, and the rate; verifies each response against the stored commitment; aggregates admitted commitments; ledger entry; outcome `aggregated` with the seed | no download, no training; selects s = ⌈rate·n⌉ coordinates from the seed and proves them over the committed update |

**Requirements from the Step 5 prompt:**

| Requirement | Implementation |
|---|---|
| Seed generated server-side per round and pushed to clients; clients can't predict the selection before computing their update | `CommitChallengeMixin.fit_config` creates the seed in the challenge round's `configure_fit`, which Flower calls only after the commit round's `aggregate_fit`. The update is already stored by then |
| Coordinate-level sampling across the whole model | `fl/core/sampling.py::sample_indices`: s distinct coordinates of the flattened model in schema order |
| `FL_ZKP_SAMPLE_PCT` works; dead branches removed | the server reads `FL_ZKP_SAMPLE_PCT` (default 0.1, must be in (0, 1]) and sends it to clients; `FL_ZKP_NUM_LAYERS`, `FL_ZKP_SELECT_BY`, `FL_ZKP_SAMPLE_SEED` and `_select_layers` are gone |
| Selection reproducible server-side for audit | partial Fisher–Yates driven by SHA-256(seed ‖ i ‖ counter) with rejection sampling, independent of Python and numpy versions; the seed and sample size are recorded in each round's `round_outcomes` entry |

**How a proof is tied to the commitment.**
- **ElGamal:** the gnark service's new `/elgamal/prove_with` rebuilds each ciphertext from the client's stored value and randomness, and the circuit's public inputs are those ciphertexts. The server verifies against the committed ciphertexts at the sampled indices, so a proof made from any other value or randomness fails (`TestElgamalProofWithWrongRandomnessFailsAgainstCommitment`).
- **Plaintext:** the server recomputes the MiMC hash from the committed weights at the sampled indices.

**Rejected with a reason:**
- a challenge response without a commitment
- a commitment with no challenge response
- incomplete or duplicate chunk coverage
- a chunk that fails verification

Quorum, infrastructure aborts and ledger rules are as in Step 4.

## Evidence

| Test | Shows |
|---|---|
| `tests/test_sampling.py` (8) | seed reproducibility, distinct sorted in-range indices, per-coordinate sampling frequency at the nominal rate over 3,000 seeds, rate validation, exact detection probability within 0.03 of a 4,000-trial Monte Carlo |
| `tests/test_he_elgamal_zkp_sampled.py` | honest commit → challenge → aggregate decrypts to the weighted mean; **adversary test** (below); challenge verification outage aborts the round |
| `tests/test_zkp_sampled.py` (8) | plaintext protocol: no seed in commit rounds, fresh seed per challenge round, a response that doesn't match the commitment is rejected, missing commitment or missing response rejected, a client refuses a challenge it didn't commit for, simulation refused |
| `zkp_gnark_service/elgamal_test.go` (+3) | commit then prove the sampled coordinates verifies; wrong randomness fails against the commitment; out-of-range randomness refused |

**Full suite in `flEnv`: 100 passed, 1 xfailed** (the unbound CKKS composite, by design). The old `test_pct_sampling_env` failure is gone with the code it tested.

**Adversary test** (`test_poisoned_commitment_is_caught_only_if_sampled`): the attacker replaces one committed ciphertext with an encryption of q = 90,000 (in range, far above any chunk bound) and answers the challenge using its honest values and randomness. The seed is fixed, so the test covers both cases:
- **Poisoned coordinate sampled:** the attacker is rejected and only the honest client is aggregated.
- **Not sampled:** both clients are admitted and the poisoned value reaches the decrypted aggregate. **This is undetected by design.**

# Phase 3 — detection probability and operating point

## Model

- **n** coordinates in the model (for example 2,914 for healthcare), **s = ⌈p·n⌉** sampled per client per round.
- **Commitment.** The client's upload is fixed before the seed exists. ElGamal under a fixed public key is perfectly binding: C₂ − sk·C₁ = v·G determines v. The client can't open a committed ciphertext to a different value.
- **Challenge.** The seed is 256 bits of server randomness (`secrets.token_hex`) drawn after the commit round. SHA-256 is treated as a random oracle, so the selected set is a uniformly random s-subset of the n coordinates, independent of the commitment.
- **Bad coordinate.** Coordinate i is bad if no valid proof can include it: its value is out of range (|q| ≥ 2¹⁷), or q_i² exceeds the bound share of every chunk it could land in (q_i² > B²·c/n for chunk length c, B = FL_ZKP_MAX_NORM·scale). Any chunk containing a bad coordinate has no satisfying witness, so by Groth16 soundness (standard assumptions, trusted setup; see S1-08) the client can't produce a verifying proof for it and is rejected.
- **Adversary.** An adaptive client that commits an update with m bad coordinates at positions of its choosing, then answers the challenge as best it can.

## Derivation

The client evades detection only if none of its m bad coordinates is sampled. The selection is a uniform s-subset independent of those positions, so:

  P(evade) = C(n−m, s) / C(n, s) = ∏_{i=0}^{m−1} (n−s−i)/(n−i)

Each factor is at most (n−s)/n = 1−p, so

  P(detect) = 1 − P(evade) ≥ 1 − (1−p)^m ≥ 1 − e^(−p·m).

To be detected with probability at least 1−δ, it's therefore enough that **m ≥ ln(1/δ) / p**. The exact values used below come from `fl.core.sampling.detection_probability`.

**Scope of the guarantee:**
- **No grinding.** A client can't retry its commitment against a known seed within a round, because the seed is drawn after the commitment is stored and a second upload in the same round is not accepted.
- **Repeated rounds.** A client that keeps attacking is detected in each round independently. Over T rounds, P(never detected) = P(evade)^T: for m = 10 at p = 0.1, 0.347 in one round, 0.0050 over 5 rounds. **A detected client is only excluded from that round.** No persistent exclusion is implemented, so an attacker's poison is still admitted in the rounds where it evades.
- **Sub-threshold poisoning isn't covered.** A coordinate whose value is below its chunk's bound share passes even when sampled; the mechanism is a norm and range bound, not a correctness check. With the current policy (FL_ZKP_MAX_NORM = 100, scale 1000, chunk 128) on healthcare, a single coordinate can be as large as |w| ≈ 21 before it counts as bad, far above honest weights of about 0.1–1. **The bound has to be tightened, and applied to the update rather than the weights (S1-07), for detection to mean much.**
- **Unsampled coordinates carry nothing.** An out-of-range unsampled ciphertext makes aggregate decryption fail on every client (a denial of service, fail-closed). It's avoided with the same probability as poisoning is.
- **Not covered:** a server that leaks the seed early or colludes with a client; trusted setup (S1-08).

## Trade-off (measured cost × exact detection)

Source: [`audit/tables/sampling_tradeoff.md`](tables/sampling_tradeoff.md) and `.csv`, from [`audit/evidence/sampling_tradeoff.py`](evidence/sampling_tradeoff.py). Plot: [`audit/figures/sampling_tradeoff.png`](figures/sampling_tradeoff.png).

**Cost basis.** Median `/elgamal/prove_with` time per chunk, one caller on an 11-core Apple Silicon machine: 0.303 s / 0.999 s / 1.977 s for 16 / 64 / 128 coordinates. The fit is t(k) = 0.055 s + 15.0 ms·k, so client proving cost is almost exactly proportional to the rate. This is a cost measurement, not a security result. With several clients sharing one proof service, per-chunk times are higher (3.5 s median at 128 with two clients, binding.md smoke run).

| Rate p | Share of full proving | healthcare prove s / client / round | CIFAR prove s | P(detect) m=1 | m=10 | m=50 | smallest m with P ≥ 0.99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.1% | 0.5 | 9.6 | 0.010 | 0.099 | 0.407 | 413 |
| 0.05 | 5.1% | 2.3 | 47.8 | 0.050 | 0.402 | 0.925 | 89 |
| **0.10** | **10.1%** | **4.5** | **95.5** | **0.100** | **0.653** | **0.995** | **44** |
| 0.20 | 20.1% | 9.0 | 191.0 | 0.200 | 0.893 | 1.000 | 21 |
| 0.30 | 30.0% | 13.5 | 286.5 | 0.300 | 0.972 | 1.000 | 13 |
| 0.50 | 50.1% | 22.5 | 477.5 | 0.500 | 0.999 | 1.000 | 7 |
| 1.00 | 100% | 44.9 | 954.9 | 1.000 | 1.000 | 1.000 | 1 |

Detection depends mostly on p and m, only weakly on model size: the "smallest m with P ≥ 0.99" column is identical across all four datasets from p = 0.1 up, and differs by at most 43 at p = 0.01 (413 for healthcare vs 456 for MNIST). Cost scales with n.

## Recommended operating point

**Default p = 0.1** (`FL_ZKP_SAMPLE_PCT=0.1`), with this justification and these limits:

1. **Cost:** about 10% of full proving (healthcare 4.5 s instead of 44.9 s per client per round; CIFAR 95 s instead of 955 s), plus one extra network round trip per round.
2. **Coverage of attacks that move a bounded model a lot:** any update with at least 44 bad coordinates is detected with probability ≥ 0.99 in a single round. Model-replacement style attacks (Bagdasaryan et al. 2020) scale a whole update, so m is on the order of n and detection is certain at any rate ≥ 1/n, **provided** the scaled values exceed the bound, which depends on tightening it (see the scope above).
3. **Not sufficient for sparse attacks:** an attack that makes fewer than about 10 coordinates bad is detected with probability ≤ 0.65 per round at p = 0.1. If the threat model includes sparse out-of-bound manipulation:
   - use **p = 0.3**: m ≥ 13 at 0.99, ≈30% of the cost
   - or full proving with `he_elgamal_zkp`
   - or add persistent client exclusion so the per-round probability compounds: m = 10 at p = 0.1 is caught within 5 rounds with probability 0.995
4. **Plaintext `zkp_sampled`:** the same numbers apply, but the mode has no security value (the server sees plaintext); it exists for the cost comparison.

## End-to-end distributed check

Run: `compare.py --dataset healthcare --modes zkp_sampled,he_elgamal_zkp_sampled --rounds 2 --num-clients 2 --max-epochs 1`, with output outside `results/`.

**`zkp_sampled`:**
- The harness selected the mode through `--privacy_mode`.
- The server ran 4 Flower rounds, recorded as `committed`, `aggregated`, `committed`, `aggregated`, with both clients admitted every time.
- Each challenge round logged `aggregated 2 client(s) after proving 292/2914 sampled coordinates`, which is s = ⌈0.1 · 2914⌉.

**`he_elgamal_zkp_sampled`:** still running when this was committed; its result will be recorded separately.
