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
