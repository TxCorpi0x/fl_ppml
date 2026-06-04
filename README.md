# Privacy-Preserving Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flower 1.8.0](https://img.shields.io/badge/flower-1.8.0-green.svg)](https://flower.ai)
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
| 4 | ZKP Sampled | `zkp_sampled` | Groth16 zk-SNARK, sampled layers | Gradient integrity |
| 5 | ZKP Full | `zkp` | Groth16 zk-SNARK, all layers | Gradient integrity — full coverage |
| 6 | DP | `dp` | Gaussian DP-SGD (Opacus) | Membership inference |
| 7 | HE TenSEAL + ZKP | `he_tenseal_zkp` | CKKS + Groth16 | Confidentiality + Integrity |
| 8 | HE Concrete + ZKP | `he_concrete_tfhe_zkp` | TFHE + Groth16 | Confidentiality + Integrity (bandwidth-efficient) |
| 9 | HE TenSEAL + ZKP + DP | `he_tenseal_zkp_dp` | CKKS + Groth16 + DP-SGD | **Full triad** |
| 10 | HE Concrete + ZKP + DP | `he_concrete_tfhe_zkp_dp` | TFHE + Groth16 + DP-SGD | **Full triad** (bandwidth-efficient) |

### Triple Modes (9 & 10)

Modes 9 and 10 layer all three mechanisms at distinct pipeline stages:

1. **DP-SGD** (client training) — injects calibrated Gaussian noise into gradients; provides formal ε-DP guarantee against membership inference on the published model
2. **Groth16 ZKP** (pre-upload) — proves the noisy gradient's ℓ₂ norm is bounded; certifies client honesty to the server without revealing the gradient
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
conda create -n flEnv python=3.10 -y
conda activate flEnv
pip install -r requirements.txt
```

### One-Time Key and Parameter Generation
python compare.py --dataset healthcare --simulation --dp --dp_params keys/dp/dp_params.pkl --benchmark
Use the unified key/params CLI implemented in `fl.keys` instead of the removed top-level helper scripts.

```bash
cd fl_ppml

# HE keys — TenSEAL CKKS context (example)
# creates keys/he_tenseal/{secret_key.pkl,public_key.pkl}

The helper script wraps the current compare runner; if you need to change container ports or datasets, inspect [scripts/run_docker_compare.sh](scripts/run_docker_compare.sh).
# DP parameters — default ε=1.0, δ=1e-5 (example)
# creates keys/dp/dp_params.pkl
python -m fl.keys generate dp --output keys/dp/dp_params.pkl --epsilon 1.0 --delta 1e-5

# ZKP params (example)
python -m fl.keys generate zkp --output keys/zkp/zkp_params.pkl
```

### gnark ZKP Service (required for ZKP modes)

Build and run the gnark-based ZKP HTTP service used by ZKP modes. The repository expects the binary name `gnark_service`.

```bash
cd zkp_gnark_service
go build -o gnark_service main.go
./gnark_service &    # listens on :9000 by default (or set ZKP_SERVICE_PORT)
cd ..
```

---

## Quick Start

### All 10 modes (recommended)

```bash
python compare.py --dataset healthcare
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

### Simulation mode (in-process, no gRPC — fast for development)

```bash
python simulation.py simulation --rounds 2 --number_clients 2 --max_epochs 1 --benchmark
python simulation.py simulation --he --rounds 2 --benchmark
python simulation.py simulation --dp --dp_params dp_params.pkl --benchmark
```

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

The noise multiplier is derived as σ = √(2 · ln(1.25 / δ)) / ε. The sentinel value `dp_epsilon=10.0` means "load ε from `dp_params.pkl`"; use `--dp-epsilon` or `--epsilon-sweep` to override at runtime.

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
| `web3` | Submits transactions to a live EVM node (set `chain_rpc_url` in `fl/config.py`) |
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
├── simulation.py               ← single-mode wrapper (legacy --he/--zkp/--dp flags)
├── (key generation moved)      ← use `python -m fl.keys generate ...` to create HE/DP/ZKP params
├── requirements.txt
├── scripts/
│   ├── aggregate_results.py    ← Docker-run result aggregation
│   ├── aggregate_statistics.py ← multi-run mean ± std across seeds
│   └── run_repeated_experiments.sh ← loop: N runs per seed → aggregate
├── fl/                         ← core FL engine
│   ├── runner.py               ← run_mode() entry point
│   ├── chain.py                ← blockchain audit (mock / web3)
│   ├── config.py               ← global configuration
│   ├── experiment.py           ← per-mode experiment driver
│   ├── datasets.py             ← dataset registry + Dirichlet partitioning
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
| [simulation.py](simulation.py) | Legacy simulation wrapper for single-mode runs |
| [fl/compare/registry.py](fl/compare/registry.py) | Dataset and mode registry, prerequisites, defaults |
| [fl/keys/cli.py](fl/keys/cli.py) | Key / parameter generation CLI (`python -m fl.keys ...`) |
| [scripts/aggregate_statistics.py](scripts/aggregate_statistics.py) | Mean ± std aggregation over repeated runs |
| [scripts/run_repeated_experiments.sh](scripts/run_repeated_experiments.sh) | Convenience loop for repeated runs |
| [zkp_gnark_service/main.go](zkp_gnark_service/main.go) | gnark prove/verify HTTP service |
| [tests/test_fl_keys.py](tests/test_fl_keys.py) | Key-generation tests |

---

## Performance Reference

Measured on Apple Silicon (M-series), healthcare dataset, 3 clients, 3 rounds.

| Mode | Accuracy | Upload/round | Total Time | Enc+Dec | Proof Gen |
|------|----------|--------------|-----------|---------|----------|
| `baseline` | ~87.3% | 0.01 MB | ~45s (1×) | 0s | 0s |
| `he_tenseal` | ~87.1% | 244.7 MB | ~148s (3.3×) | ~1.5s | 0s |
| `he_concrete_tfhe` | ~84.8% | 17.8 MB | ~310s (6.9×) | ~5.3s | 0s |
| `zkp_sampled` | ~87.2% | 0.01 MB | ~520s (11.6×) | 0s | ~22.4s |
| `zkp` | ~87.2% | 0.01 MB | ~1200s | 0s | ~52s |
| `dp` (ε=1.0) | ~83.1% | 0.01 MB | ~48s (1.1×) | <0.1s | 0s |
| `he_tenseal_zkp` | ~87.0% | 244.7 MB | ~670s (14.9×) | ~1.8s | ~22.4s |
| `he_concrete_tfhe_zkp` | ~84.7% | 17.8 MB | ~480s (10.7×) | ~5.3s | ~22.4s |
| `he_tenseal_zkp_dp` | ~82.8% | 244.7 MB | ~675s | ~1.8s | ~22.4s |
| `he_concrete_tfhe_zkp_dp` | ~82.3% | 17.8 MB | ~485s | ~5.3s | ~22.4s |

**Bandwidth**: CKKS ciphertext expansion is ~24,000× over plaintext (0.01 MB → 244.7 MB per round). TFHE uses quantized int8 weights — 14× less bandwidth than CKKS.

**Latency**: ZKP Groth16 proof generation dominates timing in all ZKP-containing modes (~22s per client per round). Verification is cheap (~0.08s).

**Accuracy**: HE modes preserve accuracy (exact arithmetic). DP and TFHE modes trade accuracy for their guarantees — the loss is additive in triple modes.

**DP**: The only mode providing a formal information-theoretic (ε-DP) guarantee against membership inference on the published model. HE and ZKP rest on computational hardness assumptions.

---

## Environment Variables

All tuning is via environment variables — no code changes required. Variables are forwarded to subprocesses by the compare runner.

### ZKP

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_ZKP_BACKEND` | `gnark` | `gnark` = Groth16 zk-SNARK; `pedersen` = legacy commitment (no soundness) |
| `FL_ZKP_SELECT_BY` | `size` | `size` = largest layers; `random` = rotate across rounds |
| `FL_ZKP_NUM_LAYERS` | `1` | Layers proven per client per round |
| `FL_ZKP_SAMPLE_PCT` | — | Fraction of layers to prove (alternative to `FL_ZKP_NUM_LAYERS`) |
| `FL_ZKP_SAMPLE_SEED` | — | Seed for reproducible layer sampling |
| `FL_ZKP_PARALLELISM` | `4` | Concurrent proof workers |
| `FL_ZKP_SCALE` | `1000000` | Float→int64 scale for the proof circuit |
| `FL_ZKP_MAX_NORM` | `100.0` | Max ℓ₂ gradient norm in circuit; match to DP clipping norm when combining |
| `FL_ZKP_TIMEOUT` | `120` | Per-call timeout (seconds) for the gnark HTTP service |

### HE

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_ENCRYPT_LAYERS` | `model.0.weight,model.0.bias` | Layers to encrypt. `ALL` = full gradient privacy |
| `FL_CONCRETE_TFHE_BIT_WIDTH` | `14` | TFHE quantization bit width (2–16). Lower = more accuracy loss |
| `FL_CONCRETE_TFHE_ADAPTIVE_QUANT` | `0` | `1` = per-layer quantization scale fitting (~0.5–1% accuracy recovery) |

### Transport

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_GRPC_MAX_MESSAGE_LENGTH` | `2147483647` | Max gRPC payload (bytes). Default 2 GiB covers TenSEAL ciphertexts |
| `FL_CLIENT_TIMEOUT` | `7200` | Server wait per run (seconds). HE+ZKP modes can take 10+ min/round |

---

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ZKP modes: `Connection refused :9000` | gnark service not running | `cd zkp_gnark_service && ./gnark_service` |
| `proof_verification = 0.0` in results | Pedersen backend selected | `export FL_ZKP_BACKEND=gnark` |
| TenSEAL `scale out of bounds` | CKKS coefficient overflow | Already fixed; ensure `global_scale=2^40` |
| TFHE accuracy 2–3% lower | int8 quantization error | Expected trade-off |
| DP accuracy unchanged during `--epsilon-sweep` | Sentinel `dp_epsilon=10.0` used | Pass `--dp-epsilon` or use `--epsilon-sweep` |
| DP accuracy drops significantly | ε too small (strong noise) | Increase ε when generating DP params, e.g. `python -m fl.keys generate dp --epsilon 1.0` |
| `FileNotFoundError: keys/he_tenseal/secret_key.pkl` | HE keys not generated | `python -m fl.keys generate he_tenseal` |
| `FileNotFoundError: keys/dp/dp_params.pkl` | DP params not generated | `python -m fl.keys generate dp --output keys/dp/dp_params.pkl` |
| Port 8081–8084 busy | Docker port conflict | Change ports in `docker-compose.yml` |
| Blockchain table shows all zeros | Stale ledger from pre-fix run | Re-run; parser unwraps `{"ledger": [...]}` format correctly |
| `ledger_comparison.json` missing | `--chain-backend none` was set | Re-run without `--chain-backend none` |
| `he_tenseal_zkp_dp` not found | Missing from mode list | Fixed: all 10 modes in `compare.py` default |
| CIFAR `key not found` | Registry used `cifar10` only | Fixed: `@register_dataset("cifar")` alias added |
| MNIST 0-byte file on parallel download | Race condition in parallel extract | Fixed via `fcntl.flock` exclusive lock |
| Client 2 IndexError on startup | `--number_clients` missing from subprocess args | Fixed in `experiment.py` `common_args` |
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
