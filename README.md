# Privacy-Preserving Federated Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flower 1.36.0](https://img.shields.io/badge/flower-1.36.0-green.svg)](https://flower.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework comparing **ten** privacy-preserving federated learning configurations across Homomorphic Encryption (HE), Zero-Knowledge Proofs (ZKP), Differential Privacy (DP), and their combinations — with blockchain audit ledger support, sweep experiments, and non-IID dataset partitioning.

---

## Table of Contents

1. [Privacy Modes](#privacy-modes)
2. [Datasets](#datasets)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Sweep Experiments](#sweep-experiments)
6. [Statistical Significance](#statistical-significance)
7. [Blockchain Audit Ledger](#blockchain-audit-ledger)
8. [Results Directory Structure](#results-directory-structure)
9. [Architecture](#architecture)
10. [Performance Reference](#performance-reference)
11. [Environment Variables](#environment-variables)
12. [Attack Evaluation Scripts](#attack-evaluation-scripts)
13. [Troubleshooting](#troubleshooting)
14. [Documentation](#documentation)

---

## Privacy Modes

Each mode addresses a distinct threat in the federated learning pipeline.

| Property | Mechanism | Threat |
|----------|-----------|--------|
| **Confidentiality** | Homomorphic Encryption | Honest-but-curious aggregation server |
| **Integrity** | Zero-Knowledge Proofs | Byzantine / gradient-poisoning clients |
| **Membership Privacy** | Differential Privacy | Membership inference on the published model |

| # | Mode | Key | Privacy Mechanism | Threat Addressed |
|---|------|-----|------------------|-----------------|
| 1 | Baseline | `baseline` | None | — |
| 2 | HE TenSEAL | `he_tenseal` | CKKS (TenSEAL) | Gradient confidentiality |
| 3 | HE Concrete TFHE | `he_concrete_tfhe` | TFHE (Concrete ML) | Gradient confidentiality (bandwidth-efficient) |
| 4 | ZKP Sampled | `zkp_sampled` | Groth16 zk-SNARK over server-sampled coordinates (commit–challenge) | None beyond `zkp`: the server already sees plaintext. Benchmark of sampled proving cost only |
| 5 | ZKP Full | `zkp` | Groth16 zk-SNARK, all layers | Gradient integrity — full coverage |
| 6 | DP | `dp` | Gaussian DP-SGD (Opacus) | Membership inference |
| 7 | HE TenSEAL + ZKP | `he_tenseal_zkp` | CKKS + Groth16 (unbound) | Confidentiality only — proof not bound to ciphertext |
| 8 | HE Concrete + ZKP | `he_concrete_tfhe_zkp` | TFHE + Groth16 (unbound) | Confidentiality only — proof not bound to ciphertext |
| 9 | HE TenSEAL + ZKP + DP | `he_tenseal_zkp_dp` | CKKS + Groth16 (unbound) + DP-SGD | Confidentiality + membership privacy; no integrity |
| 10 | HE Concrete + ZKP + DP | `he_concrete_tfhe_zkp_dp` | TFHE + Groth16 (unbound) + DP-SGD | Confidentiality + membership privacy; no integrity |
| 11 | HE ElGamal + ZKP | `he_elgamal_zkp` | Exponential ElGamal (BabyJubJub) + ciphertext-bound Groth16 | Confidentiality + integrity: the upload is range-checked and its update against the encrypted global model is norm-bounded |
| 12 | HE ElGamal + sampled ZKP | `he_elgamal_zkp_sampled` | As mode 11, proving only server-sampled committed coordinates (commit–challenge, two Flower rounds per round) | Confidentiality + probabilistic integrity: m out-of-bound coordinates detected with probability 1 − C(n−m, s)/C(n, s) ([docs/ZKP.md](docs/ZKP.md#64-he_elgamal_zkp_sampled)) |

> **Integrity in modes 7–10.** Their ZKP proof covers a client-chosen plaintext vector and is not bound to the ciphertext the server aggregates, so a client can prove an honest vector and upload a poisoned one (`tests/test_zkp_binding_attack.py`). Only mode 11 binds proofs to the aggregated ciphertexts. Its remaining limitations — a bounded update can still be malicious (the bound caps per-round influence, not direction), a single-party trusted setup, a shared client key, and per-chunk update norms visible to the server — are listed in `ppflx/privacy/he_elgamal_zkp.py`.

### Triple Modes (9 & 10)

Modes 9 and 10 layer all three mechanisms at distinct pipeline stages:

1. **DP-SGD** (client training) — injects calibrated Gaussian noise into gradients; provides formal ε-DP guarantee against membership inference on the published model
2. **Groth16 ZKP** (pre-upload) — proves a norm bound over a client-chosen plaintext vector; the proof is not bound to the uploaded ciphertext, so it certifies nothing about what is aggregated
3. **CKKS / TFHE encryption** (upload) — encrypts the noisy, norm-bounded gradient in transit; protects against a curious aggregation server

Each mechanism is independent; their composition is safe and additive in overhead.

---

## Datasets

| Key | Description | Samples | Classes | Input |
|-----|-------------|---------|---------|-------|
| `healthcare` | Clinical binary classification | 918 | 2 | 13 tabular features |
| `creditcard` | Credit card fraud detection (imbalanced) | 284,807 | 2 | 30 tabular features |
| `stock` | Stock trend prediction | — | 3 | Tabular |
| `mnist` | MNIST handwritten digits | 70,000 | 10 | 1×28×28 image |
| `cifar` / `cifar10` | CIFAR-10 object recognition | 60,000 | 10 | 3×32×32 image |

All datasets support **non-IID Dirichlet partitioning** via `--dirichlet-alpha`:

| α | Distribution character |
|---|----------------------|
| `0.1` | Extreme non-IID — each client holds ~1–2 classes |
| `0.5` | Moderate heterogeneity |
| `1.0` | Mild heterogeneity |
| `10.0` | Near-IID — roughly uniform label distribution |

Omitting `--dirichlet-alpha` uses stratified IID partitioning (default).

---

## Setup

```bash
conda create -n flEnv python=3.12 -y
conda activate flEnv
pip install -r requirements.txt
```

### One-Time Key and Parameter Generation
python compare.py --dataset healthcare --simulation --dp --dp_params keys/dp/dp_params.json --benchmark
Use the unified key/params CLI implemented in `ppflx.keys` instead of the removed top-level helper scripts.

```bash
cd fl_ppml

# HE keys — TenSEAL CKKS context (example)
# creates keys/he_tenseal/{secret_context.bin,public_context.bin}

The helper script wraps the current compare runner; if you need to change container ports or datasets, inspect [scripts/run_docker_compare.sh](scripts/run_docker_compare.sh).
# DP parameters — default ε=1.0, δ=1e-5 (example)
# creates keys/dp/dp_params.json
python -m ppflx.keys generate dp --output keys/dp/dp_params.json --epsilon 1.0 --delta 1e-5

# ZKP params (example)
python -m ppflx.keys generate zkp --output keys/zkp/zkp_params.json
```

### gnark ZKP Service (required for ZKP modes)

ZKP modes use a Go Groth16 service, split into two roles: clients prove against a **prover**, and the server verifies against a separate **verifier** that holds only verifying keys. Neither role runs setup. Both load pinned keys and refuse to start if the keys are missing or don't match the manifest (see [docs/ZKP.md, section 7](docs/ZKP.md#7-keys-and-trusted-setup)).

```bash
cd zkp_gnark_service && go build -o gnark_service . && cd ..
```

The verifying keys and `manifest.json` are committed in `zkp_gnark_service/keys/`. Proving keys (hundreds of MB) are not committed; they live in a local cache. On a fresh checkout they must be regenerated. That run also re-pins the verifying keys, so commit the new `keys/` directory:

```bash
zkp_gnark_service/gnark_service setup --keys-dir zkp_gnark_service/keys --pk-dir ~/.cache/fl_ppml/gnark_pk --force
```

`compare.py` starts both roles itself. To run them by hand:

```bash
zkp_gnark_service/gnark_service serve --role prover --keys-dir zkp_gnark_service/keys --pk-dir ~/.cache/fl_ppml/gnark_pk --port 9000 &
zkp_gnark_service/gnark_service serve --role verifier --keys-dir zkp_gnark_service/keys --port 9001 &
```

Every proof carries the SHA-256 of the verifying key it was made under. The server rejects any proof whose key isn't the pinned one, and each `round_outcomes` entry records the manifest hash. The setup is **single-party**: whoever ran `setup` could forge proofs. See [docs/ZKP.md, section 7.3](docs/ZKP.md#73-what-a-single-party-setup-does-and-does-not-give) for what a multi-party ceremony would change.

---

## Quick Start

### Default modes

```bash
python compare.py --dataset healthcare
```

The default set is modes 1–10. The ElGamal modes are slower and opt-in:

```bash
python compare.py --dataset healthcare --modes he_elgamal_zkp,he_elgamal_zkp_sampled
```

### Select specific modes

```bash
python compare.py --dataset healthcare --modes baseline,dp
python compare.py --dataset healthcare --modes he_tenseal,he_concrete_tfhe
python compare.py --dataset healthcare --modes he_tenseal_zkp_dp,he_concrete_tfhe_zkp_dp
```

### Image datasets with non-IID partitioning

```bash
python compare.py --dataset mnist --dirichlet-alpha 0.1 --simulation
python compare.py --dataset cifar --dirichlet-alpha 0.5 --simulation
```

### Single runs (`python -m ppflx_bench.launch`)

Runs use Flower 1.36 (Python 3.12). `ppflx_bench.launch` starts a local SuperLink and one SuperNode per client, submits the Flower App declared in [pyproject.toml](pyproject.toml) with `flwr run`, waits for it and stops every process. The ServerApp is `ppflx.server:server_app`, the ClientApp `ppflx.client:client_app`; every run config key in `[tool.flwr.app.config]` is also a flag.

```bash
python -m ppflx_bench.launch --mode baseline --dataset healthcare --data-path ./dataset/ --num-clients 3 --num-rounds 3
python -m ppflx_bench.launch --mode he_tenseal --data-path ./dataset/ --num-rounds 2 --results-dir results/he_single/
python -m ppflx_bench.launch --mode dp --simulation --data-path ./dataset/ --num-rounds 2
```

`--simulation` uses Flower's Simulation Runtime instead of SuperNode processes; HE modes then transport plaintext and their results are marked `[SIM]`. Logs go to the results directory: `server.log` (SuperLink), `serverapp.log` (ServerApp), `client_<i>.log` (SuperNode and its ClientApp processes). ZKP modes need the proof service, which `compare.py` starts for you.

### Docker (original 4 modes)

```bash
bash scripts/run_docker_compare.sh
```

The helper script wraps the current compare runner; if you need to change container ports or datasets, inspect [scripts/run_docker_compare.sh](scripts/run_docker_compare.sh).

---

## Sweep Experiments

### Epsilon Sweep — Privacy-Utility Tradeoff

Runs the `dp` mode at ε ∈ {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}:

```bash
python compare.py --dataset healthcare --simulation --epsilon-sweep
```

Output: `results/healthcare/dp_eps_<ε>/<timestamp>/benchmark_dp.json` per value, plus `dp_epsilon_sweep_summary.json`.

The noise multiplier is derived as σ = √(2 · ln(1.25 / δ)) / ε. The sentinel value `dp_epsilon=10.0` means "load ε from `dp_params.json`"; use `--dp-epsilon` or `--epsilon-sweep` to override at runtime.

### Alpha Sweep — Non-IID Heterogeneity

Runs all 10 modes at α ∈ {0.1, 0.5, 1.0, 10.0} Dirichlet concentration values:

```bash
python compare.py --dataset healthcare --simulation --alpha-sweep
```

Output: `results/healthcare/alpha_<α>/<timestamp>/comparison_report.json` per value, plus `alpha_sweep_summary.json`.

### Override DP epsilon for a single run

```bash
python compare.py --dataset healthcare --modes dp --dp-epsilon 0.5 --simulation
```

---

## Statistical Significance

Each run uses a fixed random seed (default `42`). To estimate **mean ± std** across runs, pass `--seed` with different values and then aggregate with `scripts/aggregate_statistics.py`.

### Seed control

```bash
# --seed is forwarded to every simulation subprocess; default is 42
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 42 --simulation
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 123 --simulation
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 456 --simulation
```

Each run creates a new `results/healthcare/<timestamp>/` directory. The seed used is recorded in every `comparison_report.json` entry as `"seed": 42`.

### Aggregate statistics

```bash
# After N runs, compute mean ± std per mode:
python scripts/aggregate_statistics.py --root results/healthcare/

# Warn if any mode has fewer than 5 runs:
python scripts/aggregate_statistics.py --root results/healthcare/ --min-runs 5

# Filter to specific modes only:
python scripts/aggregate_statistics.py --root results/healthcare/ --modes baseline,dp

# Skip the bar-chart PNG:
python scripts/aggregate_statistics.py --root results/healthcare/ --no-plot
```

The script scans only direct `YYYYMMDD_HHMMSS` children of `--root` — it ignores the merged dataset-level `comparison_report.json` and sweep subdirs (`alpha_*/`, `dp_eps_*/`). Outputs:

| File | Description |
|------|-------------|
| `results/healthcare/statistical_summary.json` | Per-mode mean, std, min, max, 95% CI, N |
| `results/healthcare/statistical_summary.png` | Grouped bar chart with ± std error bars (requires matplotlib) |

**Console output example**:

```
====================================================================
  Statistical Summary — Healthcare Dataset  (5 run(s))
====================================================================
Mode                     N          Accuracy                F1         Time(s)
--------------------------------------------------------------------
baseline                 5       82.3 ± 1.2%     0.821 ± 0.013       8.2 ± 0.4
dp (ε=1.0)               5       77.1 ± 0.8%     0.769 ± 0.009       9.1 ± 0.3
he_tenseal               3       82.3 ± 1.1%     0.820 ± 0.011      38.1 ± 1.2
====================================================================
Saved → results/healthcare/statistical_summary.json
```

### Automated multi-run loop

`scripts/run_repeated_experiments.sh` runs the full experiment loop automatically:

```bash
# 5 runs of fast modes, 3 runs of medium/slow modes, then aggregate
bash scripts/run_repeated_experiments.sh healthcare
bash scripts/run_repeated_experiments.sh creditcard 20 5   # dataset rounds clients
```

---

## Blockchain Audit Ledger

Every run records a per-round audit trail. Two backends:

| Backend | Description |
|---------|-------------|
| `mock` (default) | Writes JSON ledger files locally — zero external dependencies |
| `web3` | Submits transactions to a live EVM node (set `chain_rpc_url` in `ppflx/config.py`) |
| `none` | Disables the ledger entirely |

```bash
python compare.py --dataset healthcare --chain-backend mock        # default
python compare.py --dataset healthcare --chain-backend none        # disable
python compare.py --dataset healthcare --chain-ledger-dir /tmp/ledgers
```

After every run a blockchain audit table is printed:

```
=== Blockchain Audit Summary ===
Mode                     Events  ModelCommit  ProofAnchor  Last Block
baseline                      3            3            0           3
dp                            3            3            0           3
he_tenseal                    3            3            0           3
he_concrete_tfhe              3            3            0           3
zkp                           6            3            3           6
zkp_sampled                   6            3            3           6
he_tenseal_zkp                6            3            3           6
he_concrete_tfhe_zkp          6            3            3           6
he_tenseal_zkp_dp             6            3            3           6
he_concrete_tfhe_zkp_dp       6            3            3           6
TOTAL                        48           30           18
```

- **ModelCommit** — recorded every aggregation round for every mode
- **ProofAnchor** — recorded every round only for ZKP-containing modes; stores a hash of each client's Groth16 proof

The combined ledger is saved to `results/<dataset>/<timestamp>/ledger_comparison.json`.

---

## Results Directory Structure

```
results/
└── healthcare/
    ├── comparison_report.json        ← merged results for all modes
    ├── <timestamp>/
    │   ├── comparison_report.json
    │   ├── ledger_comparison.json    ← merged blockchain audit trail
    │   ├── ledgers/
    │   │   ├── ledger_baseline.json
    │   │   ├── ledger_dp.json
    │   │   ├── ledger_he_tenseal.json
    │   │   ├── ledger_he_concrete_tfhe.json
    │   │   ├── ledger_zkp.json
    │   │   ├── ledger_zkp_sampled.json
    │   │   ├── ledger_he_tenseal_zkp.json
    │   │   ├── ledger_he_concrete_tfhe_zkp.json
    │   │   ├── ledger_he_tenseal_zkp_dp.json
    │   │   └── ledger_he_concrete_tfhe_zkp_dp.json
    │   ├── baseline/benchmark.json
    │   ├── he_tenseal/benchmark.json
    │   ├── he_concrete_tfhe/benchmark.json
    │   ├── zkp_sampled/benchmark.json
    │   ├── zkp/benchmark.json
    │   ├── dp/benchmark.json
    │   ├── he_tenseal_zkp/benchmark.json
    │   ├── he_concrete_tfhe_zkp/benchmark.json
    │   ├── he_tenseal_zkp_dp/benchmark.json
    │   └── he_concrete_tfhe_zkp_dp/benchmark.json
    ├── dp_eps_0.5/<timestamp>/benchmark_dp.json
    ├── dp_eps_1.0/<timestamp>/benchmark_dp.json
    ├── dp_eps_2.0/<timestamp>/benchmark_dp.json
    ├── dp_eps_3.0/<timestamp>/benchmark_dp.json
    ├── dp_eps_5.0/<timestamp>/benchmark_dp.json
    ├── dp_eps_8.0/<timestamp>/benchmark_dp.json
    ├── dp_epsilon_sweep_summary.json
    ├── alpha_0.1/<timestamp>/comparison_report.json
    ├── alpha_0.5/<timestamp>/comparison_report.json
    ├── alpha_1.0/<timestamp>/comparison_report.json
    ├── alpha_10.0/<timestamp>/comparison_report.json
    ├── alpha_sweep_summary.json
    └── statistical_summary.json   ← after running aggregate_statistics.py
```

**Result merging**: Re-running a single mode overwrites only that mode's entry; other modes are preserved.

**Seed tracking**: Each `comparison_report.json` entry includes `"seed": <int>` recording the `--seed` value used, so `aggregate_statistics.py` can group runs correctly across different seeds.

---

## Architecture

```
fl_ppml/
├── compare.py                  ← main CLI: all 10 modes, sweeps, blockchain
├── pyproject.toml              ← Flower App: ServerApp/ClientApp components and run config
├── (key generation moved)      ← use `python -m ppflx.keys generate ...` to create HE/DP/ZKP params
├── requirements.txt
├── scripts/
│   ├── aggregate_results.py    ← Docker-run result aggregation
│   ├── aggregate_statistics.py ← multi-run mean ± std across seeds
│   └── run_repeated_experiments.sh ← loop: N runs per seed → aggregate
├── ppflx/                         ← core FL engine
│   ├── launch.py               ← SuperLink/SuperNode launcher (python -m ppflx_bench.launch)
│   ├── server.py               ← ServerApp + FedPrivate strategy (Message API)
│   ├── client.py               ← ClientApp + FlowerClient
│   ├── runner.py               ← run_mode() entry point
│   ├── chain.py                ← blockchain audit (mock / web3)
│   ├── config.py               ← global configuration
│   ├── experiment.py           ← per-mode experiment driver
│   ├── datasets/               ← dataset registry, loaders and Dirichlet partitioning
│   ├── privacy/
│   │   ├── he_tenseal.py       ← CKKS encryption via TenSEAL
│   │   ├── he_concrete.py      ← TFHE via Concrete ML
│   │   ├── zkp_client.py       ← Groth16 proof generation (gnark HTTP)
│   │   ├── zkp_server.py       ← Groth16 proof verification
│   │   └── dp.py               ← DP-SGD noise via Opacus
│   └── compare/
│       ├── runner.py           ← sweep orchestration
│       └── report.py           ← result merging + audit summary
├── zkp_gnark_service/          ← Go service: Groth16 prove + verify on :9000
├── tests/                      ← pytest suite (sweeps, modes, datasets)
└── docs/                       ← reference guides (see docs/README.md)
  ├── README.md               ← guides index
  ├── FL.md                   ← federated learning theory + Dirichlet + alpha sweep
  ├── DP.md                   ← DP theory + epsilon sweep + runtime override
  ├── FHE.md                  ← HE theory + TenSEAL + Concrete ML
  ├── ZKP.md                  ← ZKP theory + gnark + environment variables
  └── BC.md                   ← blockchain integration
```

---

## Documentation

This checkout includes the guides index at [docs/README.md](docs/README.md) plus the module and script entry points below. The separate `ATTACKS.md` guide is not part of this tree.

| File | Purpose |
|------|---------|
| [docs/README.md](docs/README.md) | Guides index for FL, DP, FHE, ZKP, and blockchain topics |
| [compare.py](compare.py) | Main comparison CLI for the 10 privacy modes |
| [ppflx_bench/launch.py](ppflx_bench/launch.py) | Single runs on a local SuperLink and SuperNodes (`python -m ppflx_bench.launch`) |
| [ppflx_bench/compare/registry.py](ppflx_bench/compare/registry.py) | Dataset and mode registry, prerequisites, defaults |
| [ppflx/keys/cli.py](ppflx/keys/cli.py) | Key / parameter generation CLI (`python -m ppflx.keys ...`) |
| [scripts/aggregate_statistics.py](scripts/aggregate_statistics.py) | Mean ± std aggregation over repeated runs |
| [scripts/run_repeated_experiments.sh](scripts/run_repeated_experiments.sh) | Convenience loop for repeated runs |
| [zkp_gnark_service/main.go](zkp_gnark_service/main.go) | gnark prove/verify HTTP service |
| [tests/test_fl_keys.py](tests/test_fl_keys.py) | Key-generation tests |

---

## Performance Reference

The stored results under `results/` were produced before the current ZKP protocols, key handling and fail-closed checks, so their timings and bandwidth figures describe older code and are not reproduced here. They will be regenerated with the current code.

Current single-proof measurements (Apple M3 Pro, 18 GB):

| Circuit | Constraints | Prove | Verify |
|---|---|---|---|
| Norm (`zkp`, CKKS/TFHE composites), 256 values per proof | 103,365 | 0.65 s | 2.3–3.6 ms |
| ElGamal (`he_elgamal_zkp`), 128 coordinates per proof | 1,274,949 | 3.13 s | 7.4 ms |

See [docs/ZKP.md, section 11](docs/ZKP.md#11-performance) for proofs per round, key sizes and end-to-end timings.

**DP**: The only mode providing a formal (ε, δ)-DP guarantee against membership inference on the published model. HE and ZKP rest on computational hardness assumptions.

---

## Environment Variables

All tuning is via environment variables — no code changes required. Variables are forwarded to subprocesses by the compare runner.

### ZKP

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_ZKP_BACKEND` | `gnark` | `gnark` = Groth16 zk-SNARK; `pedersen` = legacy commitment stub (no verification), refused unless `FL_ZKP_ALLOW_PEDERSEN_STUB=1` |
| `FL_ZKP_ALLOW_PEDERSEN_STUB` | `0` | `1` = knowingly run the unverified pedersen stub; every round is recorded as `unverified_stub` |
| `FL_ZKP_SAMPLE_PCT` | `0.1` | Sampled modes: fraction of model coordinates proven per client per round, in (0, 1]. Set on the server; the per-round seed is drawn by the server after clients commit and recorded in `round_outcomes` |
| `FL_ZKP_PARALLELISM` | `1` | Proofs generated concurrently per client (each proof uses several cores inside the prover service) |
| `FL_ZKP_SCALE` | `1000000` | Float→int64 scale for the proof circuit |
| `FL_ZKP_MAX_NORM` | calibrated per dataset | Overrides the server's **update**-norm bound B on ‖w_local − w_global‖₂ (not a weight norm). Default: `PER_STEP_UPDATE_NORM[dataset] × local_epochs × max_client_batches` in `ppflx/core/update_bound.py` (the server computes the batch count from the same partition clients use), calibrated with `scripts/calibrate_update_norm.py`. Clients clip their update to B before proving. DP runs need their own calibration (DP noise enlarges honest updates); the DP clipping norm is a per-step gradient clip and is not a valid value |
| `FL_ZKP_TIMEOUT` | `600` | Fallback HTTP timeout (seconds) for the proof services; `FL_ZKP_PROVE_TIMEOUT` (default 1800) and `FL_ZKP_VERIFY_TIMEOUT` / `FL_ZKP_VERIFY_LIGHT_TIMEOUT` (default 900) take precedence |

**Failure handling.** Security-relevant paths fail closed:

- A client whose proof generation fails raises instead of uploading.
- The server checks every upload against its own model schema and proof policy: full coverage, shapes, scale and bound. Clients that fail are rejected; the round aborts if the proof service is unreachable.
- If fewer clients are admitted than `min_fit_clients`, the global model is unchanged and nothing is written to the ledger.
- Every round's outcome (`aggregated`, `no_quorum`, `infrastructure_abort`), including rejected clients and reasons, is recorded under `round_outcomes` in `comparison_report.json`. ZKP runs with any non-aggregated round fail validation.
- Sampled modes (`zkp_sampled`, `he_elgamal_zkp_sampled`) use two Flower rounds per round (commit, then challenge). A client that commits but doesn't answer the challenge, or answers without having committed, is rejected.
- DP without a params file refuses to run unless `--dp_epsilon` is passed explicitly.

### HE

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_ENCRYPT_LAYERS` | `ALL` | TenSEAL layers to encrypt; unlisted layers are sent in plaintext. Names not in the model are an error |
| `FL_CONCRETE_TFHE_FORCE_REAL` | `0` | `1` = run real TFHE on image datasets (high RAM) |
| `FL_CONCRETE_TFHE_ALLOW_SIMULATED` | `0` | `1` = knowingly send plaintext quantized weights on image datasets; otherwise TFHE on images refuses to run |
| `FL_ELGAMAL_SCALE` | `10000` | `he_elgamal_zkp` quantization: q = round(w·scale), \|q\| < 2¹⁷ (so \|w\| < 13.1). The proven bound is W²·⌈B·scale + √n/2⌉²; the √n/2 rounding slack is small only when B·scale ≫ √n, which is why the default rose from 1000 |
| `FL_CLIENT_WAIT_TIMEOUT` | `600` | Seconds the server waits for `min_avail_clients` before a round; if they don't arrive it stops the run with an error instead of Flower's 24-hour wait |
| `FL_SERVER_GRACE` | `600` | Harness: seconds a run may stay active after every SuperNode exited before it is stopped and the mode marked failed |
| `FL_ZKP_PROVER_URL` | `http://127.0.0.1:9000` | gnark prover role (clients) |
| `FL_ZKP_VERIFIER_URL` | `http://127.0.0.1:9001` | gnark verifier role (server) |
| `FL_ZKP_KEYS_DIR` | `zkp_gnark_service/keys` | Pinned manifest and verifying keys. The circuit sizes (norm chunk, ElGamal coordinates per proof) come from this manifest |
| `FL_ZKP_PK_DIR` | `~/.cache/fl_ppml/gnark_pk` | Proving-key cache (prover only) |
| `FL_CONCRETE_TFHE_BIT_WIDTH` | `14` | TFHE quantization bit width (2–16). Lower = more accuracy loss |
| `FL_CONCRETE_TFHE_ADAPTIVE_QUANT` | `0` | `1` = per-layer quantization scale fitting (~0.5–1% accuracy recovery) |

### Transport

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_CLIENT_TIMEOUT` | `7200` (6 h for HE/ZKP modes) | Harness: time budget per run in seconds, plus 30 min headroom; `FL_SERVER_TIMEOUT` sets the total directly |
| `FL_CLIENT_WAIT_TIMEOUT` | `600` | ServerApp: seconds to wait for enough SuperNodes before a round, then stop the run |

---

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ZKP modes: `Connection refused :9000`/`:9001` | gnark prover/verifier not running | see "gnark ZKP Service" above; `compare.py` starts both |
| `No pinned ZKP key manifest` / `Proving keys … not in` | keys never generated on this machine | run `gnark_service setup` (above) |
| `HTTP 503 … verifying key` | service started from different keys than the manifest | restart the services from `zkp_gnark_service/keys` |
| `proof_verification = 0.0` in results | Pedersen backend selected | `export FL_ZKP_BACKEND=gnark` |
| TenSEAL `scale out of bounds` | CKKS coefficient overflow | Already fixed; ensure `global_scale=2^40` |
| TFHE accuracy 2–3% lower | int8 quantization error | Expected trade-off |
| DP accuracy unchanged during `--epsilon-sweep` | Sentinel `dp_epsilon=10.0` used | Pass `--dp-epsilon` or use `--epsilon-sweep` |
| DP accuracy drops significantly | ε too small (strong noise) | Increase ε when generating DP params, e.g. `python -m ppflx.keys generate dp --epsilon 1.0` |
| `FileNotFoundError: keys/he_tenseal/secret_context.bin` | HE keys not generated | `python -m ppflx.keys generate he_tenseal` |
| TenSEAL `RuntimeError: incompatible version` | key files written by another TenSEAL version | `python -m ppflx.keys generate he_tenseal --overwrite` |
| TFHE rounds end `no_results`, `ClientApp stopped responding`; client log shows an LLVM `Assertion failed` | `ppflx/keys/prebuilt/` bundles compiled by another Concrete version abort the ClientApp natively | move the bundles out of `ppflx/keys/prebuilt/`; fresh ones are generated on the next run |
| `FileNotFoundError: keys/dp/dp_params.json` | DP params not generated | `python -m ppflx.keys generate dp --output keys/dp/dp_params.json` |
| `port 1909x is in use` from `ppflx_bench.launch` | a SuperLink or SuperNode from an interrupted run is still running | stop it (`lsof -ti tcp:19093`), or wait for the other run to finish; runs use fixed ports |
| Blockchain table shows all zeros | Stale ledger from pre-fix run | Re-run; parser unwraps `{"ledger": [...]}` format correctly |
| `ledger_comparison.json` missing | `--chain-backend none` was set | Re-run without `--chain-backend none` |
| `he_tenseal_zkp_dp` not found | Missing from mode list | Fixed: all 10 modes in `compare.py` default |
| CIFAR `key not found` | Registry used `cifar10` only | Fixed: `@register_dataset("cifar")` alias added |
| MNIST 0-byte file on parallel download | Race condition in parallel extract | Fixed via `fcntl.flock` exclusive lock |
| `node partition … does not match num-clients` | SuperNode `--node-config` disagrees with the run's `num-clients` | start SuperNodes with `partition-id=<i> num-partitions=<num-clients>` |
| CIFAR CNN input shape mismatch | `in_channels` hardcoded to 1 | Fixed: `Net` uses dynamic `in_channels` + computed `flat_dim` |

---

## References

- [Flower Federated Learning Framework](https://flower.ai)
- [TenSEAL — CKKS Homomorphic Encryption](https://github.com/OpenMined/TenSEAL)
- [Concrete ML — TFHE via Zama](https://github.com/zama-ai/concrete-ml)
- [gnark — Groth16 zk-SNARK in Go](https://github.com/consensys/gnark)
- [Opacus — DP-SGD for PyTorch](https://opacus.ai)
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
- Bonawitz et al., "Towards Federated Learning at Scale", SysML 2019
