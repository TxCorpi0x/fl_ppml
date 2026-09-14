# Benchmark integrity and number regeneration

Date: 2026-09-14
Scope: every stored `comparison_report.json` under `results/`, the per-mode chain ledgers stored next to them, and the performance figures published in `README.md` and `docs/`. The raw results are the only source of truth used here (CLAUDE.md rule 6).

## Executive summary

1. **`zkp_sampled` was never run.** In every stored `zkp_sampled` row, the harness actually ran full `ZKPMode`. Any published difference between "sampled" and "full" ZKP is between two runs of the same configuration.
2. **Proofs were emitted in all 30 stored ZKP-family runs.** Every aggregated client emitted proofs in every round, with full coverage. **Whether they verified is not recorded anywhere,** so the runs do not show that any update was *admitted* on the strength of a proof.
3. **The stored results cannot be reproduced from the committed code** (see findings.md S1-03 update).
4. **Every stored number is a single run (seed 42, n = 1).** Run-to-run noise between identical configurations reaches 12.5% on prover time and 3.1% on elapsed time, and single-run HE overhead comes out negative (−8.1%) on creditcard. Overhead differences below roughly 10% are not supported.
5. **The README/docs performance table is not supported by the raw data:** its round count, upload sizes, timings and accuracies all differ.

## Reproduce

```bash
conda activate flEnv
python scripts/regenerate_tables.py        # -> audit/tables/<dataset>.{csv,md}
python scripts/validate_zkp_runs.py        # exit 1 if any ZKP run cannot be certified
python -m pytest -q tests/test_zkp_validation.py
```

`regenerate_tables.py` output:

```
cifar: 10 modes -> audit/tables/cifar.csv
creditcard: 10 modes -> audit/tables/creditcard.csv
healthcare: 10 modes -> audit/tables/healthcare.csv
mnist: 10 modes -> audit/tables/mnist.csv
stock: 10 modes -> audit/tables/stock.csv
```

Each dataset has exactly one unique report payload. `results/<dataset>/comparison_report.json` is a byte-identical copy of the `keep_*` run and is excluded. Every row is therefore n = 1, seed 42, 3 clients, 20 rounds, distributed gRPC (not simulation; per-client benchmark files are present).

### Changes made to the Step 1 table script

| Change | Why |
|---|---|
| Added **measured elapsed** (first→last `ModelCommit` timestamp in the ledger) and computed overhead vs baseline from it | The previous wall-clock column was an estimate built from component timers. It is off by up to 11% from the measurement (CIFAR baseline: 462.5 s estimated vs 520.6 s measured). |
| Server-side timers (`proof_verification`, `server_aggregate`) are no longer divided by client count | They run sequentially on the server, not in parallel across clients |
| Added a proofs/client/round column from the ledger | Makes each ZKP row carry its own coverage evidence |
| Quality columns relabelled "round mean" | `model_quality.*.mean` averages all 20 evaluated rounds. It is not the final model's score; the final-round value is not stored. |
| Deduplication prefers run directories that have ledgers | The merged dataset-level copy has no ledger to cross-check |
| `zkp_sampled` footnote on every table | See item 1 |
| `.gitignore` now keeps `audit/tables/*.csv` | `*.csv` was silently excluding the CSV deliverable |

## Item 1 — `zkp` vs `zkp_sampled`

| Dataset | Proof gen total s (zkp / sampled) | sampled ÷ zkp | Proof gen max s (zkp / sampled) | Proofs/client/round (zkp / sampled) | Upload B (zkp / sampled) | Measured elapsed s (zkp / sampled) |
|---|---|---:|---|---|---|---|
| healthcare | 2046.1 / 2049.2 | 1.002 | 52.4 / 36.4 | 7 / 7 | 13196 / 13196 | 660.6 / 680.8 |
| creditcard | 3230.9 / 2826.8 | 0.875 | 227.3 / 52.1 | 7 / 7 | 18060 / 18060 | 2256.9 / 2242.1 |
| stock | 1991.2 / 2046.0 | 1.027 | 35.2 / 36.3 | 7 / 7 | 13196 / 13196 | 677.0 / 693.7 |
| mnist | 31244.7 / 30407.0 | 0.973 | 549.1 / 526.3 | 31 / 31 | 184524 / 184524 | 11274.9 / 10981.3 |
| cifar | 44422.7 / 44010.4 | 0.991 | 769.7 / 765.3 | 39 / 39 | 256604 / 256604 | 15126.7 / 14993.9 |

"Sampled" is slower on stock (+2.7%) and healthcare (+0.2%) and faster on the other three. It never saves the ~30% that proving only the largest layer would save: that layer holds 67% (healthcare), 50% (creditcard), 69% (MNIST) and 77% (CIFAR) of parameters.

**Mechanism: the sampled mode never ran.**
- `fl/compare/registry.py` defines `zkp_sampled` with `internal_mode="zkp"`.
- `_build_mode_flags()` (`fl/compare/experiment.py`) emits only `--zkp`.
- `_resolve_mode()` (`simulation.py:140`, `main_server.py:124`) maps `--zkp` to `"zkp"`.
- `ZKPSampledMode` is never instantiated by the benchmark.

Two independent stored records confirm this:
- **Ledgers:** proofs per client per round are identical for both labels. They equal full chunked coverage of the model: tabular 6 tensors with the 2048-element layer split into 2 chunks = 7; MNIST CNN = 31; CIFAR CNN = 39. The layer sizes were computed from `fl/models`.
- **Upload bytes:** `fl/client.py:219-227` adds proof bytes to the upload tally. zkp minus baseline is exactly 220 B × proof count on every dataset (healthcare 1540 = 7×220; CIFAR 8580 = 39×220), with std 0 across all fits.

Upload is byte-identical because the model, the proof count and the proof size are all identical.

**What the differences actually are:** run-to-run noise between identical configurations, plus one systematic effect.

**Systematic effect — lazy Groth16 setup is billed to whichever ZKP mode runs first (S2).** `getCircuit()` compiles the circuit and runs `groth16.Setup` inside the first `/prove` call for each circuit size. That call sits inside the client's `proof_generation` timer. The harness reuses one healthy gnark service across modes (`_ensure_gnark_service`). Ledger timestamps show `zkp` always runs first among ZKP modes, and its proof-gen maximum is the outlier: creditcard 227.3 s vs ~52 s for every later ZKP mode, healthcare 52.4 s vs ~36–38 s. On creditcard, this one-off cost plus noise is the whole 12.5% "saving".

The zkp/sampled pairs are the only same-configuration replicates in the data. They put the noise floor at **0.2–12.5% on prover totals and 0.7–3.1% on measured elapsed time.**

## Item 2 — ordering `he_*_zkp_dp ≥ he_*_zkp ≥ he_*`

Checked on measured elapsed time and on summed crypto timers (`encryption + decryption + proof_generation + proof_verification + dp_noise_addition`).

| Dataset | Backend | Measured elapsed s (he / +zkp / +zkp+dp) | Elapsed ordering | Crypto-timer ordering |
|---|---|---|---|---|
| healthcare | tenseal | 13.0 / 690.6 / 696.3 | holds | holds |
| healthcare | concrete_tfhe | 176.8 / 888.5 / 891.2 | holds | holds |
| stock | tenseal | 26.0 / 703.6 / 718.8 | holds | holds |
| stock | concrete_tfhe | 193.0 / 908.9 / 913.8 | holds | holds |
| creditcard | tenseal | 1274.6 / 2232.4 / 2436.4 | holds | **inverted** (2935.8 → 2906.4 s, −1.0%) |
| creditcard | concrete_tfhe | 1554.5 / 2617.6 / 2839.1 | holds | **inverted** (3429.7 → 3386.6 s, −1.3%) |
| mnist | tenseal | 982.8 / 10871.4 / 10741.4 | **inverted** (−1.2%) | **inverted** (31312.4 → 30761.4 s, −1.8%) |
| mnist | concrete_tfhe | 1016.0 / 10741.4 / 10954.5 | holds | holds |
| cifar | tenseal | 494.0 / 14846.5 / 15083.5 | holds | holds |
| cifar | concrete_tfhe | 523.3 / 14598.1 / 15257.6 | holds | holds |

**Mechanism.** The DP step adds almost nothing to crypto cost: `dp_noise_addition` totals 0.02–0.24 s per run. Prover time dominates, and it depends on circuit size, not on whether weights were noised. Every inversion is 1.0–1.8%, inside the noise floor measured in item 1. The triple and double modes are therefore indistinguishable in crypto cost at n = 1.

Where the triple mode is clearly slower on elapsed time (creditcard +204 s / +222 s, about 9%), the extra time comes from DP-SGD *training*, not crypto. That matches `dp` vs `baseline` on creditcard (+164 s, +12%).

Two related points:
- **HE overhead is negative on two datasets:** `he_tenseal` on creditcard (−8.1%) and CIFAR (−5.1%). Small overheads cannot be measured from single runs.
- **Round 1 often runs short a client.** On every dataset, round 1 aggregated only 2 of 3 clients in `baseline`, `dp`, `he_tenseal`, `zkp` and `zkp_sampled`, and in `he_concrete_tfhe` on CIFAR. All four HE+ZKP composites, and `he_concrete_tfhe` on the other four datasets, aggregated 3 of 3 in every round. So one mode family systematically trains round 1 on less data than the other, and the pattern tracks the mode rather than any failure. This contradicts `docs/FL.md:637` ("all clients participate in every round"). The cause (likely a client-availability race at round 1) is not investigated here.

## Item 3 — proof generation without proofs

**The field the prompt names is not stored.** `gnark_num_proofs` exists only in the in-flight Flower fit metrics (`fl/privacy/zkp.py:113`). It is not written to any benchmark or report file, so "proof_generation > 0 but gnark_num_proofs absent" cannot be checked directly.

**Substitute evidence used instead:**
- the ledger `ProofAnchor` per round (proof hashes plus the ids of clients whose proofs were anchored), and
- the upload-byte identity from item 1.

**Result:** all 30 ZKP-family runs pass. Every round has a `ProofAnchor`, every aggregated client is in it, and the proof count per client is constant across clients and rounds. `scripts/validate_zkp_runs.py`:

```
30 run(s) checked, 0 failed validation
WARN  … zkp_sampled_not_sampling: … identical to full zkp coverage      (all 5 datasets)
Not checked: proof verification outcomes are not persisted; a fail-open round is indistinguishable from a verified one (audit/findings.md S1-02)
```

**What this does and does not establish:**
- **Established:** no stored run fell through the proof-*generation* fail-open path (`_generate_proofs` returning `([], 0)`). If a client had sent no proofs, it would be missing from the anchor while still counted in the `ModelCommit`.
- **Not established:** that any proof *verified*. When no client verifies, both aggregation paths re-admit every client (S1-02) and build the anchor from that set. A fully fail-open round leaves exactly the same ledger as a fully verified round, and no log of verification results was kept.
- **Not reproducible:** these runs used a gnark/Python revision that returned a usable `/prove` response. The committed revision does not (S1-03 update). Whether proofs would verify under that revision cannot be re-checked.

## Validation mode added to the harness

- **`fl/compare/validation.py`** contains `validate_zkp_ledger()` and `sampled_coverage_warning()`. The module docstring spells out what is and is not checked.
- **`fl/compare/runner.py`:** after each ZKP-family mode (`internal_mode` in `zkp`, `he_zkp`, `he_zkp_dp`), the runner validates the mode's ledger against the configured round count and stores the result as `zkp_validation` in `comparison_report.json`. A failing mode gets `success=False` and a loud `[ZKP-VALIDATION] [FAIL]` listing.
  - After the report, plots and ledger merge are written, `run_comparison()` raises `RuntimeError`, so `compare.py` exits 1.
  - **No ledger (for example `--chain-backend none`) counts as a failure,** because nothing can be certified.
  - A `zkp_sampled` run with the same coverage as `zkp` gets a `zkp_sampled_not_sampling` diagnostic.
- **`compare.py`** validates by default. `--no-validate-zkp` opts out and its help text says such runs are not publication evidence.
- **`scripts/validate_zkp_runs.py`** applies the same validator to stored results.
- **`tests/test_zkp_validation.py`** has 10 tests covering: a complete ledger; partial participation; a missing ledger failing closed; a missing round; an aggregated client without proofs; an empty anchor; inconsistent counts; a `num_proofs` mismatch; the unchecked-verification warning; and sampled-coverage detection.

**Not done (proposed for Step 4):** certifying verification *outcomes*. That needs each aggregation path to record which clients verified and whether a fallback happened, and the ledger/report to persist it. It touches `fl/privacy/zkp.py`, `fl/privacy/he_zkp.py`, `fl/server.py` and `fl/chain.py`, which is outside the harness, and it overlaps directly with the fail-closed rewrite. Until then, the harness certifies proof *emission* only, and says so.

## Other S2 findings from the raw data

| ID | Finding | Evidence |
|---|---|---|
| N-1 | Every `zkp_sampled` number is mislabeled full ZKP | Item 1 |
| N-2 | The first ZKP mode in a run absorbs one-off circuit compile plus Groth16 setup in its `proof_generation` time | Item 1: proof-gen maximum outliers |
| N-3 | No replication: every row is n = 1, seed 42. `scripts/run_repeated_experiments.sh` exists but no repeated-seed output is stored | Every table has `n = 1`; noise floor up to 12.5% (item 1) |
| N-4 | Quality metrics are means over all rounds, not final-model scores. Creditcard accuracy (99.9% for every mode) reflects class imbalance and says nothing; use AUPRC | `benchmark.summary()` stores mean/min/max only |
| N-5 | **HE upload on image datasets is smaller than the plaintext model,** which real full-model CKKS/TFHE encryption cannot produce | MNIST: baseline 177,704 B, `he_tenseal` 165,189 B, `he_concrete_tfhe` 71,038 B. CIFAR: 248,024 / 230,386 / 94,744 B. On tabular data CKKS is 57.8× plaintext. **Resolved in audit/binding.md B-1 and B-2:** TenSEAL encrypts only `model.0.weight`/`model.0.bias` by default, which matches no CNN layer, so nothing is encrypted on images. Real TFHE is auto-disabled on image datasets and sends plaintext quantized int32. No image-dataset HE row measured encryption at all. |
| N-6 | The component-timer "wall-clock" differs from measured elapsed by up to 11% | CIFAR baseline 462.5 vs 520.6 s |
| N-7 | Stored results cannot be regenerated from committed code | findings.md S1-03 update |

## Published figures the raw data does not support

The README "Performance Reference" table (`README.md`, mirrored in `docs/README.md:326-347` and `docs/DP.md:899-902`) says "healthcare dataset, 3 clients, 3 rounds". **No 3-round run exists in `results/`.** Every stored run is 20 rounds, so none of the table's timings can be traced to raw data. The comparisons below use the healthcare 20-round run.

| Published claim | Raw data | Verdict |
|---|---|---|
| Baseline total ~45 s | 7.7 s measured over 20 rounds | Unsupported |
| `he_tenseal` upload 244.7 MB/round | 673,619 B (0.64 MiB) | Unsupported (≈360× smaller) |
| CKKS expansion ~24,000× over plaintext (also `docs/FL.md:248`) | 57.8× | Unsupported |
| TFHE "14× less bandwidth than CKKS" | TFHE 18,679,335 B is **27.7× larger** than CKKS | Contradicted |
| `he_concrete_tfhe` upload 17.8 MB | 18,679,335 B = 17.8 MiB | Supported (figure only) |
| `zkp_sampled` ~22.4 s proof gen vs `zkp` ~52 s | Both 34.7 s per call; same configuration | Unsupported; the distinction is invalid |
| `zkp_sampled` ~520 s (11.6×) vs `zkp` ~1200 s | 680.8 s vs 660.6 s measured; same configuration | Unsupported |
| `he_concrete_tfhe` ~310 s (6.9×) | 176.8 s (22.9× baseline) | Unsupported |
| `he_concrete_tfhe_zkp` ~480 s, cheaper than `he_tenseal_zkp` ~670 s | 888.5 s vs 690.6 s: TFHE+ZKP is **slower** | Contradicted |
| Verification ~0.08 s per client | 0.049 s per client-round (`zkp`) | Unsupported |
| Baseline accuracy ~87.3%, `he_concrete_tfhe` ~84.8%, `dp` ~83.1% | Round-mean test accuracy 81.9%, 68.3%, 81.5%. Final-round values are not stored | Unsupported |
| "HE modes preserve accuracy (exact arithmetic)" | CKKS is approximate arithmetic; `he_concrete_tfhe` round-mean accuracy is 13.6 points below baseline | Contradicted |
| "Accuracy penalty is additive in triple modes" | n = 1; differences within per-round variation | Unsupported |
| DP noise <0.1 s | `dp_noise_addition` total 0.02 s | Supported |
| ZKP proof generation dominates ZKP-mode time | Per-client proof gen (total ÷ 3 clients) is ≥ 92% of measured elapsed: healthcare 682 s of 661 s, MNIST 10,415 s of 11,275 s | Supported |
| `docs/FL.md:569-575`: `zkp_sampled` proves 100 sampled coordinates, "seeded by round and client ID", 1 − 2⁻¹²⁸ soundness against Byzantine clients | No coordinate sampling exists in code; the mode never ran; soundness is undermined by S1-01/S1-04/S1-05/S1-07/S1-09 | Unsupported |
| `docs/FL.md:376` alpha-sweep accuracies for `zkp_sampled` | No alpha-sweep output stored in `results/` | Unsupported |
| `docs/FL.md:637` "all clients participate in every round" | Round 1 has 2/3 clients in baseline, dp, he_tenseal, zkp and zkp_sampled on every dataset | Contradicted |
| `docs/BC.md:931`, `docs/README.md:241` ledger event counts (6 events, 3 rounds) | Stored ledgers have 20 rounds; 40 events for ZKP modes | Stale |

The thesis text itself is not in this repository and was not checked. Any thesis table citing the figures above inherits these verdicts.

## Recommendations before any paper uses these numbers

1. **Rerun and store the raw data.** Rerun from committed code, with at least 3 seeds per mode, after the gnark `/prove` schema mismatch (S1-03) is fixed. Keep the gnark service and server logs next to the results.
2. **Drop `zkp_sampled` from all tables** until Step 5 implements real sampling and the harness actually instantiates it.
3. **Pay the Groth16 setup cost before timing starts,** or report it as its own column.
4. **Resolve N-5** before publishing any image-dataset HE upload or overhead figure.
5. **Store final-round quality metrics,** and report AUPRC for imbalanced datasets.
6. **Record verification outcomes** in the report and ledger (with Step 4) so ZKP runs can certify admission, not just emission.
