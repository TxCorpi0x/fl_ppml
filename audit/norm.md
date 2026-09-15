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
