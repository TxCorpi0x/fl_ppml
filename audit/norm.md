# Norm bound: weights, not updates

Date: 2026-09-15
Scope: Step 7 Phase 1. Which vector every ZKP mode proves a norm bound on, how the bound is chosen, and what an attacker can do inside it. **No source files were changed for this report.**

Runtime numbers come from [`audit/evidence/norm_evidence.py`](evidence/norm_evidence.py):
- Healthcare, 2 clients, seed 42, harness hyperparameters (lr 0.001, momentum 0.9, batch 16).
- Production pinned keys (Step 6).
- It starts the real prover and verifier and runs uploads through the servers' real admission code.

```bash
PYTHONPATH=. python audit/evidence/norm_evidence.py
```

```
model: 2914 parameters; norm circuit chunk n=256; FL_ZKP_MAX_NORM=100.0; plaintext scale 1e+06; ElGamal scale 1000, chunk 128
units bounded separately by the plaintext circuit: 15

N1  honest norms
    global (init, seed 42): ||w||=5.932, max unit 2.786
    client 0, 1 epoch(s) from init: ||w||=5.932, max unit ||w||=2.786, ||Δ||=0.0344, max|Δ_i|=0.0078
    client 1, 1 epoch(s) from init: ||w||=5.932, max unit ||w||=2.786, ||Δ||=0.0354, max|Δ_i|=0.0071
    client 0, 5 epoch(s) from init: ||w||=5.943, max unit ||w||=2.788, ||Δ||=0.2363, max|Δ_i|=0.0553
    client 1, 5 epoch(s) from init: ||w||=5.943, max unit ||w||=2.788, ||Δ||=0.2429, max|Δ_i|=0.0525
    client 0, 1 epoch from a 10-epoch model: ||Δ||=0.0524
    client 1, 1 epoch from a 10-epoch model: ||Δ||=0.0507

N2  what the policy admits
    plaintext zkp: every unit ||w_unit|| ≤ 100.0 → whole model ||w|| ≤ 100.0·√units = 387.3
    he_elgamal_zkp: shares sum to ||w|| ≤ 100.0; per coordinate |w| < 131.1
    largest admissible ||Δ|| = ||w_global|| + bound (Δ = −g·(1 + bound/||g||)): elgamal 105.9

A1 boosted replacement: ||w||=5.97, max unit 2.80, ||Δ||=0.52 (15× honest 1-epoch), plaintext-admissible=True, elgamal-admissible=True
A2 max plaintext: ||w||=383.43, max unit 99.00, ||Δ||=388.60 (11288× honest 1-epoch), plaintext-admissible=True, elgamal-admissible=False
A2 max elgamal: ||w||=99.00, max unit 29.34, ||Δ||=103.73 (3013× honest 1-epoch), plaintext-admissible=True, elgamal-admissible=True

Reference: FedAvg(honest, honest)  acc/AUPRC = 51.7 / 71.9
           attacker's model alone   acc/AUPRC = 25.2 / 36.3

zkp             A1: admitted=['honest', 'attacker'] rejected=[] → acc/AUPRC 25.7 / 36.4
zkp             A2: admitted=['honest', 'attacker'] rejected=[] → acc/AUPRC 46.2 / 51.5
he_elgamal_zkp  A1: admitted=['honest', 'attacker'] rejected=[] → acc/AUPRC 25.7 / 36.4
he_elgamal_zkp  A2: admitted=['honest', 'attacker'] rejected=[] → acc/AUPRC 44.6 / 48.3
```

(The honest reference model is weak because it's one epoch from initialization at lr 0.001. What matters is the comparison between rows, not the absolute accuracy.)

## Hypothesis: confirmed

> The circuit enforces Σw² ≤ bound_sq on a layer's weight tensor, not on ‖w_local − w_global‖.

### What each mode proves

In every mode, the vector proved is the **post-training local weights**:

| Mode | Proof input | Evidence |
|---|---|---|
| `zkp` | `net.state_dict()` after training | `fl/client.py:148-210`: `receive_parameters` loads the global model into `self.net`, `train` updates it in place, then `send_parameters` is called. `fl/privacy/zkp.py:154` → `_generate_proofs` → `generate_gnark_proofs(net.state_dict())` (`zkp.py:260`) |
| `he_tenseal_zkp`, `he_concrete_tfhe_zkp` (+`_dp`) | the same plaintext `state_dict`, via the inner `ZKPMode` | `fl/privacy/he_zkp.py:162`. These proofs aren't bound to the ciphertext anyway (S1-01, relabelled confidentiality-only in Step 3) |
| `zkp_sampled` | sampled coordinates of the committed local weights | `fl/privacy/zkp_sampled.py:61-64` commits `_plain_params(net)`; `:73` proves `values[indices]` |
| `he_elgamal_zkp` | quantized local weights, encrypted | `fl/privacy/he_elgamal_zkp.py:102-113`: `quantize(state[name])` → `prove_chunk` |
| `he_elgamal_zkp_sampled` | quantized local weights, committed then sampled | `fl/privacy/he_elgamal_zkp_sampled.py:66-70, 89-93` |

The circuits contain no reference to a global model:
- Norm circuit: `Σ w_i² ≤ Bound` plus a MiMC hash of w (`zkp_gnark_service/main.go:61-72`).
- ElGamal circuit: `Σ (v_i − 2¹⁷)² ≤ Bound` over the encrypted values (`zkp_gnark_service/elgamal.go:118-122`).
- The server's admission checks (`zkp.py:230-248`, `he_elgamal_zkp.py:232-265`) hand the service only the uploaded parameters or ciphertexts.

`findings.md` S1-07 is **confirmed**.

**Does the client have the global model at proof time?** Yes, in every mode, but it isn't kept:
- `fit()` receives `parameters`, and `receive_parameters` decrypts them for HE modes. Clients hold the shared secret key (`he_elgamal_zkp.py:143-162`); training then overwrites `self.net`.
- Commit–challenge modes train only in the commit round, so the global model from that round would have to be stored in `context["commitment"]` alongside the committed values.

## Can a client submit an arbitrarily large update while its final weights stay inside the bound?

**Yes.** The admissible set is a ball around 0, not around w_global. An admitted update can have norm up to ‖w_global‖ + B, and in any direction. Two concrete scenarios are measured above, both admitted by the real server code under the production keys.

**A1: boosted model replacement.** This is the Bagdasaryan et al. (2020) construction.
- **Construction.** The attacker trains its own target model, here 5 epochs on flipped labels (AUPRC 36.3). It then uploads `w_att = g + N·(w_bad − g)` for N = 2 clients, so FedAvg with one honest client lands on w_bad.
- **Why the bound doesn't see it.** Every weight stays tiny: ‖w_att‖ = 5.97, largest unit 2.80, against a bound of 100. The update norm is 0.52, 15× an honest one-epoch update.
- **Result.** Both `zkp` and `he_elgamal_zkp` admit it, and **the aggregate becomes the attacker's model** (AUPRC 71.9 → 36.4). One malicious client replaces the global model in one round.

**A2: the largest admissible update.** Each bounded unit is pushed to 0.99·B in the direction opposite to the global weights.
- **Size.** For `zkp` the update norm is 388.6, **11,288×** an honest one-epoch update. For `he_elgamal_zkp`, whose bound is split across the model, it's 103.7, **3,013×**.
- **Result.** Both are admitted. The aggregate is damaged (AUPRC 71.9 → 51.5 and 48.3). The damage is smaller than A1 only because this direction is crude, not because the bound limits it.

### How much this weakens the guarantee

For its stated purpose, which is Byzantine robustness and poisoning resistance (README, `docs/ZKP.md:986`), **this is a total break:**
- **Nothing an attacker would want is excluded.** The bound only rules out models whose own weights are enormous. Honest models on healthcare have ‖w‖ ≈ 5.9 (a trained stored baseline is 5.76 to 5.84), which is 17× below the bound. Any backdoored or degraded model of the same architecture that keeps normal-looking weights is admissible.
- **One client is enough.** Boosting lets a single client install such a model in one round (A1).

What survives is narrow:
- the proof shows the uploaded values are the ones hashed or encrypted (binding, Step 3);
- no single coordinate can be absurdly large (range check, ElGamal only);
- whole-model weight energy is capped.

These are integrity-of-encoding properties, not robustness properties.

**Moving the bound to the update is a partial fix, not a complete one.** With ‖Δ‖ ≤ B and FedAvg, m admitted attackers out of N clients can shift the global model by at most about (m/N)·B per round, but **in any direction**:
- Unboosted A1 (w_bad itself, ‖Δ‖ ≈ 0.26) is indistinguishable in norm from an honest 5-epoch client (0.24), so any bound that admits honest clients admits it.
- An update bound therefore caps per-round influence and removes one-shot boosting. It doesn't make FedAvg Byzantine-robust; that needs robust aggregation on top (for example, median or trimmed mean over admitted updates), which is outside this step.

## How the bound is chosen

**It's a fixed constant, the same for every dataset:**
- `FL_ZKP_MAX_NORM`, default `100.0`, read at import in `fl/core/zkp_gnark.py:20` and in `fl/core/elgamal_gnark.py:92`.
- Since Step 4 the server's own value is authoritative (`check_proof_policy`, `zkp_gnark.py:471-485`), so clients can't choose it.
- It isn't derived from the DP clipping norm, which is a separate per-step *gradient* clip (`dp_max_grad_norm = 1.0`, `fl/config.py:79`, applied in `fl/core/engine.py:145-161`).
- It isn't configured per dataset.

**It's loose enough to be vacuous:**
- On weights, 100 is 17× the whole honest model and 36× its largest bounded unit.
- On updates it would be 2,900× (one epoch) to 420× (five epochs) the honest norm.
- Either way, no plausible honest or malicious client is ever near it.

**The two circuits also compose the bound differently** (new finding **N-2**):
- **Plaintext `zkp` and `zkp_sampled`:** *every* layer or 256-value chunk gets the full bound B (`zkp_gnark.py:238`, `check_proof_policy:484`). The whole-model bound is therefore B·√units, which is 387 for healthcare's 15 units. It grows with model size and with the Step 6 chunk size (256).
- **`he_elgamal_zkp` (sampled too):** proportional shares that sum to B model-wide (`elgamal_gnark.py:130-141`).
- Either way the same `FL_ZKP_MAX_NORM` means different guarantees in different modes.

**The documentation is wrong in three places:**

| Location | Claim | Status |
|---|---|---|
| `README.md:472`; `docs/ZKP.md:1516, 1828` | `FL_ZKP_MAX_NORM` is the "max ℓ₂ gradient norm" | **Contradicted.** It bounds weights |
| `docs/ZKP.md:986` | "A poisoned update that violates the proven constraints will cause proof verification to fail" | **Misleading.** Poisoned updates satisfy the constraints (A1, A2) |
| `docs/ZKP.md:1678-1682`; `README.md:472` | set it equal to the DP clipping norm (e.g. 1.0) | **Contradicted, and breaks honest clients.** The DP clip is per-step on gradients. On weights, B = 1 is below the honest largest unit (2.786, N1), so every honest `zkp` client fails to prove. This is derived from the N1 numbers, not run |

## Phase 2 plan (for approval)

1. **Prove the update.** The client stores the global model it received (for commit–challenge modes, in the commitment context) and proves on Δ = w_local − w_global.
2. **Model-wide, proportional bound in every mode.** Replace the per-unit full bound in the plaintext circuit with the ElGamal-style shares, so ‖Δ‖ ≤ B holds model-wide in every mode (fixes N-2).
3. **Honest clients always pass: clip the update.** The client clips Δ to B before proving and uploading, as in norm-clipping defences and DP-FedAvg. An honest client can then never be rejected by the bound, and B becomes a stated protocol parameter rather than a guess.
4. **Choosing B.** Set it per dataset from a calibration of honest update norms, recorded in the dataset config and documented. For healthcare at 1 epoch that's ‖Δ‖ ≈ 0.035; at 5 epochs, ≈ 0.24. For example, the 95th percentile times a small margin, with the clipping rate reported per round so over-tight bounds show up in results. When DP is on, the bound should come from the DP accounting of the update, not from the per-step clip.
5. **Re-run the healthcare benchmark** and report prover cost and admission rates, per the step prompt. The norm-evidence attacks become regression tests: A2 must be rejected, and boosted A1 must be clipped or rejected.

### Decisions needed before Phase 2

**D1. How the *plaintext* circuit learns w_global.** The server knows it in plaintext.
- **(a)** Add a MiMC hash of the global chunk as a second public input; the circuit hashes both vectors and bounds Σ(w_i − g_i)². The server computes the hash of its own global model. Estimated at about 2× norm-circuit constraints (86,725 today), and it keeps two public inputs. **Recommended.**
- **(b)** Pass g as n public inputs. There's no extra hashing, but verifier cost grows linearly with n, and so does the verifying key.

**D2. How the *ElGamal* circuit learns w_global without revealing it to the server.** This is the hard part.
- **(a)** Clients prove against the server's aggregate ciphertext from the previous round. The circuit takes those ciphertexts as public inputs (the server holds them) and the plaintext sums S as witness. It checks `C2 − sk·C1 = S·G` with the shared client secret bound by `sk·G = PK`, then bounds Σ(W·v − S)² ≤ W²·share. Round 1 uses the public initial model.
  - **Cost:** one extra variable-base scalar multiplication per coordinate, a wider range check on S (about 26 bits instead of 18), and twice as many public inputs. Estimated 1.5–2× today's 805,082 constraints and proving key. This is an estimate, to be measured.
  - **Recommended:** it keeps the global model hidden from the server.
- **(b)** Give the server g in plaintext as a public input. It's cheap, but it **breaks the confidentiality the mode exists for**. Not recommended.
- **(c)** Keep ElGamal on weights with a documented limitation. It's the smallest change, but it leaves the flagship mode vulnerable to A1.

**D3. Bound calibration.** Clip the update client-side with B from a per-dataset calibration run (**recommended**), or use a fixed B without clipping, which risks rejecting honest clients.

Out of scope and not proposed: robust aggregation (see "How much this weakens the guarantee").

**Stopping here, as Step 7 Phase 1 requires.**

---

## Phase 2: proofs bound the update (implemented)

Decisions taken: D1 (plaintext), D2 option (a) (ElGamal against the encrypted aggregate), D3 (client-side clipping with a per-dataset calibrated bound). One deviation from the D1 recommendation, explained below.

### What each mode now proves

| Mode | Statement | What the server checks it against |
|---|---|---|
| `zkp` | MiMC(Δq) = H and ΣΔq² ≤ b_j for each 256-value chunk j, with Δq = round(w·10⁶) − round(g·10⁶) | Δq recomputed from the upload and the server's **own** global model g (`fl/privacy/zkp.py::quantized_update`), and Σ_j b_j ≤ ⌈B·10⁶ + √n⌉² |
| `zkp_sampled` | the same, over the sampled coordinates of the committed Δq | Δq from the commitment and the global model at commit time; Σ_j b_j ≤ ⌈B·10⁶ + √s⌉² for s sampled coordinates |
| `he_elgamal_zkp` | per 128-slot chunk: each C encrypts an in-range q; SK·G = PK; each global slot (G1, G2) decrypts under SK to T < 2³² with G2 = T·G + SK·G1; Σ(W·v − T)² = Σ(W·q − S)² ≤ b_j | the ciphertexts it received, **its previous aggregate** and its total weight W as public inputs (the initial model, W = 1, before any aggregation); Σ_j b_j ≤ W²·⌈B·scale + √n/2⌉² |
| `he_elgamal_zkp_sampled` | the same, over the sampled committed coordinates | the committed ciphertexts and the aggregate held at commit time |
| `he_tenseal_zkp`, `he_concrete_tfhe_zkp` (+`_dp`) | the plaintext statement over the client's own Δq | only light verification, as before: **still not bound to the ciphertext** (S1-01, confidentiality-only) |

Across all modes:
- **Per-proof bounds are declared by the client.** They are public inputs, so the circuit enforces each one, and the server enforces their sum. Proportional shares were dropped: they could reject an honest update whose energy is concentrated in a few chunks.
- **The rounding slack is explicit.** √n (plaintext: difference of two rounded values) or √n/2 (ElGamal: one rounding around the global model). By the triangle inequality, any ‖Δ‖ ≤ B then always fits. The admitted update is ‖Δ‖ ≤ B + √n/scale (plaintext) or B + √n/(2·scale) (ElGamal).
- **N-2 is fixed.** The plaintext circuit no longer gives each chunk the full bound.

### D1 deviation: hash of the update instead of a second hash of g

The approved plan added MiMC(g) as a second public input. Instead the client proves over Δq directly, and the server recomputes Δq from the upload and its own g before checking the hash.
- **Equivalent for the verifier:** the server holds g in plaintext, so a proof over Δq with a server-derived hash binds the proof to the update exactly as a (w, g) pair would.
- **Cheaper:** the norm circuit, its keys and its prover cost are unchanged, where D1 was estimated at 2× constraints.
- **The only difference is for a third-party auditor.** They need w and g to recompute the hash either way.
- **The composites don't gain binding under either design**, because their server holds neither w nor g in plaintext.

### D2 (a): the ElGamal circuit

`zkp_gnark_service/elgamal.go`:
- **The global model's plaintext is a witness,** decrypted inside the proof: SK is constrained by SK·G = PK, and T by G2 = T·G + SK·G1 with a 32-bit range check. That makes T unique: the group order is about 2²⁵¹, and G1 is a sum of client C1 values, which lie in the prime-order subgroup.
- **The server learns neither the global model nor the update.**
- **W < 2¹⁴ is range-checked,** so W·v and T stay below 2³² and nothing wraps. The server aborts a round whose total weight would exceed it (`check_aggregate_weight`), rather than aggregate something no client can prove against.
- **Padding slots** carry v = offset, r = 0 against the global slot (identity, W·offset·G), so their difference term is 0.

Measured (`gnark_service setup`, n = 128): **1,274,949 constraints**, up from 805,082 (×1.58, within the 1.5–2× estimate). The proving key is 420.2 MB (was 245.7 MB), the verifying key 33.4 KB, and setup took 48.0 s. The keys were re-pinned: `zkp_gnark_service/keys/manifest.json`, recorded in `audit/setup.md`.

New circuit tests (`elgamal_circuit_test.go`), using raw witnesses that bypass the Go pre-checks:
- a false global plaintext (claiming a zero update);
- T aliased by the group order;
- a secret key that doesn't match PK;
- W = 2¹⁴.

Each is refused by the circuit itself. `elgamal_test.go` covers:
- two rounds, the second proved against an encrypted aggregate;
- the bound applying to the update rather than the weights;
- a proof failing against a different global model, a re-randomised encryption of the same model, or a different weight.

### D3: the bound, clipping and calibration

- **The server sets B and sends it to clients** in the fit config (`zkp_max_update_norm`). A client refuses to prove without it.
- **B = PER_EPOCH_UPDATE_NORM[dataset] × local_epochs** (`fl/core/update_bound.py`). Scaling with epochs follows from the triangle inequality over per-epoch updates. `FL_ZKP_MAX_NORM` overrides it and is now an update norm.
- **Clients clip before proving.**
  - Plaintext (`clip_update_in_place`): the clip is checked against the exact integer statement, including one unit per zero-energy proof. Unclipped tensors are proved as uploaded.
  - ElGamal (`quantize_update`): q is rounded around the global model, so the clip always fits the slack.
  - Clip status is reported per round (`zkp_update_norm`, `zkp_update_clipped`).
- **Calibration** (`scripts/calibrate_update_norm.py`): plain FedAvg from the seeded initial model with harness hyperparameters, one epoch per round.
  - Healthcare, 3 clients, 5 rounds: largest honest update 0.021748, so the per-epoch bound is 1.5 × 0.021748 = **0.032623**.
  - Other datasets were not calibrated at this point; see the N-3 fix below, which calibrates all five.
- **ElGamal default scale 1000 → 10⁴.** At 1000, the rounding slack for healthcare (√2914/2 ≈ 27 units) was comparable to B·scale ≈ 33 units, so the proven bound would have been almost twice B. At 10⁴ the slack is 2.7 units against 326, i.e. B + 0.0027. The value range still allows |w| < 13.1.

### Found and fixed while re-running the benchmark

The pre-change "before" run failed both sampled modes (`zkp_sampled`, `he_elgamal_zkp_sampled`) with `challenge received without a commitment from the previous round`. Two defects combined:
1. **The strategy started round 1 without every client.** `FedPrivate.configure_fit` sized its sample from the clients connected at that moment, and only afterwards waited for `min_available_clients`. With `--num-clients 3`, round 1 sampled 2 of 3 whenever a client was still starting (visible in the logs, `sampled 2 clients (out of 3)`).
   - This affected every mode's first round, not only the sampled ones.
   - Fixed: `configure_fit` now waits before sizing the sample, and the harness passes `--min_avail_clients` equal to the number of clients.
   - Setting the flag alone did not help; a partial run with only the flag still sampled 2 of 3.
2. **A late client crashed the challenge round.** Such a client raised an exception in the challenge round, which Flower recorded as a client failure and which disconnected it.
   - Now it answers with no proofs and the server rejects it as `challenge response without a commitment`, which fails closed without a crash.

Tests: `test_first_round_samples_every_client_that_connects_while_waiting`, `test_client_without_a_commitment_answers_with_no_proofs`.

### Tests

Go: `ok`. Python: **131 passed, 1 xfailed** (the xfail is the documented S1-01 CKKS composite).

Attack regressions:

| Test | Asserts |
|---|---|
| `test_zkp_binding_attack.py::test_unclipped_large_update_with_valid_proofs_is_rejected_by_the_total_bound` | a plaintext A1/A2-style update with valid proofs and truthful bounds is rejected, and the aggregate is the honest update |
| `…::test_understated_bound_does_not_verify` | declaring small bounds for a large update fails verification |
| `…::test_update_measured_against_the_wrong_global_model_is_rejected` | a proof against a stale or different g is rejected |
| `…::test_oversized_honest_update_is_clipped_and_admitted` | the client library clips, and ‖aggregate − g‖ ≤ B |
| `test_he_elgamal_zkp.py::test_unclipped_large_update_with_valid_proofs_is_rejected_by_the_total_bound` | the same for ElGamal |
| `…::test_large_update_cannot_be_proved_under_bounds_that_fit_the_total` | the service refuses to prove (`statement not satisfied`) |
| `…::test_update_proved_against_a_stale_global_model_is_rejected` | a round-2 proof against the initial model is rejected |
| `…::test_honest_updates_aggregate_across_two_rounds` | round 2 proves against the encrypted aggregate (W = 40) and decrypts to the expected mean |
| `test_he_elgamal_zkp_sampled.py::test_poisoned_commitment_is_caught_only_if_sampled[True-over_bound_commitment]` and `[False-over_bound_commitment]`, `test_over_bound_sampled_update_with_valid_proofs_is_rejected` | an over-bound committed coordinate is rejected when sampled and admitted when not (by design) |
| `test_update_bound.py` | a clipped update always fits the server's integer bound (plaintext, and ElGamal for W ∈ {1, 3, 40}), and an unclipped one of the same size does not |

### What this does and does not give

**Now guaranteed:**
- An admitted client moves the global model by at most (n_k/N)·(B + slack) per round.
- In `he_elgamal_zkp` this holds with the proof bound to the aggregated ciphertexts and to the encrypted global model, without revealing either to the server.

**Still not guaranteed:**
- **Direction.** An unboosted poisoned update of honest size is admitted (Phase 1, "partial fix"). One-shot boosted replacement is no longer possible, but a sustained attack over many rounds or by several clients is not prevented. That needs robust aggregation, which is out of scope.
- **Sampled modes:** unsampled coordinates carry no proof.
- **Composites:** still unbound (S1-01).

**New leakage in `he_elgamal_zkp`:** each chunk's declared bound is public, so the server learns the squared norm of the update restricted to each 128-coordinate chunk. Proportional shares would avoid this but can reject honest clients. Coarse bucketing of the declared bounds is a possible mitigation, not implemented.

### Healthcare benchmark: before vs after

Same command on both code versions, so the numbers are directly comparable in their settings:
`python compare.py --dataset healthcare --modes baseline,zkp,zkp_sampled,he_elgamal_zkp,he_elgamal_zkp_sampled --rounds 3 --num-clients 3 --max-epochs 1`.

- **Before:** commit 09ee0dc, weight bound, run 20260915_102121. It ran from a scratch snapshot that was later lost, so only its summary table survives (recorded in this session).
- **After:** commit 6a32193 plus the docs in this commit, run `results/healthcare/20260915_142030`.

| Mode | Before: status, time, crypto/round | After: status, time, crypto/round | After: rounds (admitted / rejected per aggregating round) |
|---|---|---|---|
| baseline | OK, 1.1 s | OK, 0.1 s | 3/0 ×3 |
| zkp | OK, 96.7 s, 36.2 s | OK, 46.6 s, 15.5 s | 3/0 ×3 |
| zkp_sampled | **FAIL** (late client, round 2) | OK, 6.4 s, 2.1 s | commit 3, challenge 3/0 ×3 |
| he_elgamal_zkp | OK, 330.6 s, 124.8 s | OK, 684.5 s, 228.2 s | 3/0 ×3 |
| he_elgamal_zkp_sampled | **FAIL** (late client, round 2) | OK, 74.1 s, 24.7 s | commit 3, challenge 3/0 ×3 |

Proof cost in the after run (`benchmark.json`, per client per round, three clients proving concurrently with 4 workers each):

| Mode | Proof generation (mean) | Verification (mean) | Per-proof prove time (prover log) |
|---|---|---|---|
| zkp | 15.4 s | 0.06 s | 0.69 s mean, 1.24 s max over 273 norm proofs (n = 256) |
| zkp_sampled | 2.1 s | 0.01 s | same circuit |
| he_elgamal_zkp | 227.4 s | 0.21 s | 7.36 s mean, 12.7 s max over 385 proofs (n = 128) |
| he_elgamal_zkp_sampled | 24.0 s | 0.02 s | 7.98 s mean over 27 proofs |

**What changed and why:**
- **ElGamal prover cost rose about 1.8×** (crypto per round 124.8 s → 228.2 s). The circuit grew 1.58× (805,082 → 1,274,949 constraints). The rest is consistent with more memory and CPU contention from 12 concurrent proofs, but the lost before-logs don't allow a per-proof decomposition. The ElGamal statement now also needs the shared secret key and the decrypted global sums on the client, which were already available.
- **Verification cost is unchanged in order of magnitude** (0.2 s per client, mostly HTTP and ciphertext transfer).
- **The plaintext norm circuit is unchanged,** so any change in `zkp` cost does not come from the update proof. The after run's `zkp` crypto per round is lower (36.2 s → 15.5 s), but I can't attribute it with the before logs gone. One visible difference: the before run's round 1 used 2 of 3 clients, so its per-round averages mix client counts. Treat this as unexplained rather than as a speed-up from Step 7.
- **Admission:** every client in every aggregating round was admitted in the after run, in all four ZKP modes, so the calibrated bound rejected no honest client in this run. What the before run's full modes admitted is not recoverable from its summary.
- **The sampled modes now complete.** That comes from the strategy fix, not from the update bound.
- **Accuracy columns are not comparable** between runs, or meaningful at 3 rounds × 1 epoch (baseline AUPRC was 67.6 before and 35.2 after on identical code). They vary with round-1 client participation and run-to-run nondeterminism. Nothing here measures the accuracy effect of clipping.

**Gap: clipping rate is not recorded.** Clients send `zkp_update_norm` and `zkp_update_clipped` in their fit metrics, but `FedPrivate.aggregate_fit` keeps only a fixed set of metric keys, so neither reaches `benchmark.json`. Honest updates at the calibrated 1.5× margin should rarely clip. This run can't confirm that, and recording them is a small follow-up in the strategy.

### Attack evidence after the change

[`audit/evidence/norm_phase2_evidence.py`](evidence/norm_phase2_evidence.py) reproduces the Phase 1 setting: healthcare, 2 clients, seed 42, one epoch, production keys, real admission code. The bound is B = 0.032623 (`PER_EPOCH_UPDATE_NORM["healthcare"]`). Each case runs one honest client (client library: clip, prove the update) plus the second client:

```
reference: FedAvg(honest 0, honest 1) unclipped acc/AUPRC = 51.7 / 71.9
           attacker's model alone         acc/AUPRC = 26.5 / 36.3

zkp             honest  admitted=['honest', 'attacker'] rejected={} → acc/AUPRC 51.7 / 71.3, ‖agg − g‖ = 0.0323
zkp             A1 raw  admitted=['honest'] rejected={'attacker': 'declared update bounds sum to 271145984383, above the server'} → acc/AUPRC 51.7 / 71.3, ‖agg − g‖ = 0.0326
zkp             A2 raw  admitted=['honest'] rejected={'attacker': 'declared update bounds sum to 10760141412971126, above the s'} → acc/AUPRC 51.7 / 71.3, ‖agg − g‖ = 0.0326
zkp             A1 lib  admitted=['honest', 'attacker'] rejected={} → acc/AUPRC 51.3 / 64.4, ‖agg − g‖ = 0.0072
he_elgamal_zkp  honest  admitted=['honest', 'attacker'] rejected={} → acc/AUPRC 51.7 / 71.3, ‖agg − g‖ = 0.0323
he_elgamal_zkp  A1 raw  admitted=['honest'] rejected={'attacker': "declared update bounds sum to 27121039, above the server's b"} → acc/AUPRC 51.7 / 71.4, ‖agg − g‖ = 0.0327
he_elgamal_zkp  A2 raw  admitted=['honest'] rejected={'attacker': 'declared update bounds sum to 1076057086140, above the serve'} → acc/AUPRC 51.7 / 71.4, ‖agg − g‖ = 0.0327
he_elgamal_zkp  A1 lib  admitted=['honest', 'attacker'] rejected={} → acc/AUPRC 51.3 / 64.5, ‖agg − g‖ = 0.0073
```

(The script's header lines, the update norms, were lost with the terminal capture. The setting is deterministic and identical to Phase 1, where the honest one-epoch updates had ‖Δ‖ = 0.0344 and 0.0354, A1 0.52 and A2 103.7.)

| Case | Phase 1 (weight bound) | Phase 2 (update bound) |
|---|---|---|
| A1 boosted replacement, proved without clipping | admitted; aggregate = attacker's model (AUPRC 36.4) | **rejected** in both modes (declared bounds above the total); aggregate = honest update only (71.3 / 71.4) |
| A2 largest weight-admissible update | admitted (AUPRC 51.5 / 48.3) | **rejected** in both modes |
| A1 through the client library (clipped to B, admitted) | n/a | admitted; AUPRC 71.3 → 64.4 / 64.5 |

**Confirmed:**
- **One-shot model replacement is gone.** An unclipped large update can't be admitted in either mode, and the aggregate stays exactly the honest client's (clipped) update.
- **Both rejections come from the server's own total-bound check** on proofs that individually verify. The regression tests also cover the dishonest alternative (understated bounds), which fails verification.

**Confirmed limitation:**
- **A bounded malicious update still does damage.** The boosted attacker, clipped to B, is admitted, and one round costs about 7 AUPRC points (71.3 → 64.4). Its clipped direction nearly cancels the honest update (‖agg − g‖ = 0.0072 against 0.0323 for two honest clients).
- This is the partial fix predicted in Phase 1: the bound caps per-round influence, not direction.

### New finding N-3: the calibrated bound clips honest clients when the client count differs

In the honest case above both clients were **clipped**: ‖agg − g‖ = 0.0323 ≈ B, while their true updates were 0.0344 and 0.0354. They were admitted, not rejected, but their updates were shrunk by about 7%.
- **Cause.** The healthcare bound was calibrated with 3 clients. Each client then holds a third of the data, so one epoch has fewer optimizer steps than with 2 clients, which hold half. The calibration's largest update was 0.0217, and the 2-client updates are about 1.6× that, roughly the ratio of steps per epoch (1/2 ÷ 1/3 = 1.5).
- **Consequence.** `B = per_epoch × local_epochs` is under-specified: honest update size scales with the number of **local steps** (epochs × batches per client), which depends on the client count, the partition and the batch size.
- **The benchmark above is consistent with it.** It used 3 clients, matching the calibration, and admitted every client. Clipping status is not recorded (see the gap above), so this can't be confirmed.
- **Not fixed in this commit.** Proposed fix: calibrate a per-step bound and set B = per_step × local steps. The server knows the local epochs and batch size; the per-client batch count would need to be reported or bounded by partition size. Alternatively, recalibrate per deployment configuration.

## N-3 fix: the bound scales with local steps

**Change** (`fl/core/update_bound.py`): B = `PER_STEP_UPDATE_NORM[dataset] × local_epochs × max_client_batches`.
- The server computes `max_client_batches`, the largest per-client batch count, from the same partition clients use: same seed, client count, validation split and Dirichlet parameter. `main_server.py` and `fl/runner.py` pass it to `make_strategy(..., client_batches=...)`. A strategy built without it refuses to set a bound unless `FL_ZKP_MAX_NORM` is given.
- **Calibration** (`scripts/calibrate_update_norm.py --clients 2,3,5 --rounds 3`) records ‖Δ‖ / steps for every client and round in each client count, and sets the per-step bound to KAPPA (1.5) × the maximum. It prints the headroom this leaves in each measured configuration.
- **Clipping is now recorded.** Each `round_outcomes` entry carries `update_norms` (per client) and `clipped` (client ids). Sampled modes report it from the commit round, where the update is clipped (`fl/server.py::_record_round`).

### Calibration results

`--clients 2,3,5 --init-seeds 0,1,2,3,42 --rounds 3`, partition seed 42, one local epoch, harness batch size, lr 0.001. For each client count the table shows the largest honest update over all five initial models.

| Dataset | Clients | Batches/client | Largest honest ‖Δ‖ | B at this config | Headroom |
|---|---|---|---|---|---|
| healthcare | 2 | 27 | 0.068313 | 0.102470 | 1.50 |
| healthcare | 3 | 18 | 0.039871 | 0.068313 | 1.71 |
| healthcare | 5 | 11 | 0.021834 | 0.041747 | 1.91 |
| stock | 2 | 72 | 0.100194 | 0.194849 | 1.95 |
| stock | 3 | 48 | 0.085385 | 0.129899 | 1.52 |
| stock | 5 | 29 | 0.052321 | 0.078481 | 1.50 |
| creditcard | 2 | 3205 | 2.078326 | 6.822758 | 3.28 |
| creditcard | 3 | 2137 | 1.970154 | 4.549215 | 2.31 |
| creditcard | 5 | 1282 | 1.819402 | 2.729103 | 1.50 |
| mnist | 2 | 422 | 1.891221 | 5.447625 | 2.88 |
| mnist | 3 | 282 | 1.872020 | 3.640356 | 1.95 |
| mnist | 5 | 169 | 1.454421 | 2.181632 | 1.50 |
| cifar10 | 2 | 352 | 0.856292 | 1.371808 | 1.60 |
| cifar10 | 3 | 235 | 0.610558 | 0.915838 | 1.50 |
| cifar10 | 5 | 141 | 0.333994 | 0.549503 | 1.65 |

Per-step bounds: **healthcare 0.00379517, stock 0.00270624, creditcard 0.00212879, mnist 0.0129091, cifar10 0.00389718**. The legacy `cifar` dataset key has no entry, so ZKP modes refuse to start on it; use `cifar10`. Calibration read only the local `dataset/` directory. An earlier attempt with the default `./data/` path made torchvision download MNIST and part of CIFAR-10 into `./data/`, which is not used by the harness.

### The initial model mattered as much as the step count

A first calibration used only seed 42 and gave a healthcare per-step bound of 0.00208. A 2-client `zkp` smoke run then **clipped both honest clients** (‖Δ‖ = 0.0563 and 0.0587 against B = 0.0561). Nothing in the server or clients seeded the initial model, so every run started from a different random model, and the first update's size depends strongly on it: across five initial models the 2-client healthcare maximum is 0.0683, against 0.0354 for seed 42 alone.

Two changes:
- **`fl/server.py::make_strategy` seeds the initial model** with `config.seed`, so runs are reproducible.
- **Calibration takes the maximum over several initial models**, so a run with a different seed isn't clipped.

**Smoke test after the change** (healthcare, `zkp`, 2 clients, 2 rounds, harness):

| Round | Admitted | Clipped | Update norms |
|---|---|---|---|
| 1 | 2 | none | 0.0698, 0.0699 |
| 2 | 2 | none | 0.0578, 0.0591 |

B = 0.00379517 × 27 = 0.1025, so neither round clips.

**Not exactly reproducible.** Round 1's norms are about 2% above the largest the calibration observed for this configuration (0.0683). The calibration takes its sample batch after seeding, while `make_strategy` takes it before, and harness clients shuffle from unseeded generators. So the harness doesn't reproduce exactly the models and shuffles the calibration sampled. The five-seed sweep covers that variation, and KAPPA = 1.5 absorbed the remaining 2%. This is also why KAPPA should not be dropped to 1.

### New observation N-4: update norm grows sublinearly with steps

Per-step norms are not constant. On creditcard, with seed 42, they rise from 0.00062 (2 clients, 3205 steps) through 0.00086 (3 clients) to 0.00131 (5 clients, 1282 steps). On stock they go 0.00046 → 0.00071 → 0.00092. The multi-seed maxima show the same pattern. Gradients shrink as a client's model moves within an epoch, so doubling the steps doesn't double the update. Healthcare, with only 11–27 steps, stays close to constant (0.00119–0.00139).

**Consequences of taking the maximum:**
- **Honest clients are safe** in every calibrated configuration: headroom is at least 1.5 = KAPPA, reached at 5 clients.
- **The bound is loose at many steps.** creditcard with 2 clients admits 3.28× the largest honest update, so an attacker there can move the model about 3× further per round than an honest client.
- **Configurations with fewer steps than any calibrated one** (more clients, larger batches, smaller shards) can exceed the measured per-step maximum and clip honest clients. Clipping is now recorded, so this shows up in `round_outcomes`.
- **Not addressed here.** A tighter fit, norm ∝ steps^α per dataset or calibrating the exact deployment configuration, would tighten the bound without clipping honest clients.
