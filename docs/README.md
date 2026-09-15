# Privacy-Preserving Federated Learning — Guides

This directory contains comprehensive reference guides covering the privacy technologies, attack surfaces, and regulatory considerations for this framework. Each guide merges conceptual theory with practical setup instructions.

| Guide | Topics Covered |
|-------|---------------|
| [FL.md](FL.md)   | Federated learning theory · FedAvg · convergence · heterogeneity · Byzantine robustness · cross-silo vs cross-device · incentive mechanisms · **Dirichlet non-IID partitioning · alpha sweep experiment** |
| [FHE.md](FHE.md) | FHE theory · TenSEAL & CKKS · Concrete ML & TFHE · threshold/multi-key FHE · selective encryption · **HE environment variables reference** |
| [ZKP.md](ZKP.md) | ZKP theory · gnark Groth16 · Pedersen commitments · IVC/NOVA · KZG commitments · post-quantum ZKP · FL integration · **ZKP environment variables reference** |
| [DP.md](DP.md)   | DP theory · DP-SGD · Opacus · ghost clipping · privacy auditing · lower bounds · composition theorems · **epsilon sweep experiment · runtime epsilon override · privacy-utility tradeoff curve** |
| [BC.md](BC.md)   | Blockchain integration · ZKP + FHE + FL + DB on-chain · smart contracts · BCFL taxonomy · GDPR/immutability tension · ZK-rollups · incentive design · current systems |
> Note: the separate `ATTACKS.md` guide is not part of this checkout.

**Reference Papers:** Downloaded PDFs of all cited papers are in [../references/PAPERS.md](../references/PAPERS.md).

---

## Federated Learning Comparison Guide

**All 10 Privacy Modes | Automated Benchmarking | Healthcare, Financial, MNIST & CIFAR-10 Datasets**

> See [README.md](README.md) for full documentation navigation.

### Overview

This framework compares **ten** privacy-preserving federated learning modes:

| Mode | Key | Privacy Mechanism | Threat Addressed |
|------|-----|------------------|------------------|
| **Baseline** | `baseline` | None | — |
| **HE TenSEAL** | `he_tenseal` | CKKS (TenSEAL) | Gradient confidentiality (HBC server) |
| **HE Concrete TFHE** | `he_concrete_tfhe` | TFHE (Concrete ML + quantization) | Gradient confidentiality |
| **ZKP Sampled** | `zkp_sampled` | Groth16 zk-SNARK (gnark), sampled layers | Gradient integrity (Byzantine clients) |
| **ZKP (full)** | `zkp` | Groth16 zk-SNARK (gnark), all layers | Gradient integrity — full coverage |
| **DP** | `dp` | Gaussian noise — DP-SGD (Opacus) | Membership inference |
| **HE TenSEAL + ZKP** | `he_tenseal_zkp` | CKKS + Groth16 | Confidentiality **+** Integrity |
| **HE Concrete + ZKP** | `he_concrete_tfhe_zkp` | TFHE + Groth16 | Confidentiality **+** Integrity (bandwidth-efficient) |
| **HE TenSEAL + ZKP + DP** | `he_tenseal_zkp_dp` | CKKS + Groth16 + DP | Full triad: Confidentiality **+** Integrity **+** Membership Privacy |
| **HE Concrete + ZKP + DP** | `he_concrete_tfhe_zkp_dp` | TFHE + Groth16 + DP | Full triad — bandwidth-efficient variant |

> **HBC** = Honest-But-Curious server: follows protocol but tries to infer private information from the gradients it aggregates.

The two **triple modes** (`he_tenseal_zkp_dp` and `he_concrete_tfhe_zkp_dp`) represent the maximum practical privacy configuration:
1. DP noise is injected during local training, privatizing the gradient before serialization
2. Groth16 ZKP proves the noisy gradient's ℓ₂ norm is bounded, certifying client honesty
3. CKKS or TFHE encrypts the noisy, norm-bounded gradient for transit to the server

This layering is composable and safe: each mechanism operates at a distinct pipeline stage.

### Quick Start

## 1. One-time Setup

```bash
cd fl_ppml

## Generate HE keys (TenSEAL CKKS context)
python -m fl.keys generate he_tenseal --secret keys/he_tenseal/secret_context.bin --public keys/he_tenseal/public_context.bin

## Generate DP parameters (ε=1.0, δ=1e-5)
python -m fl.keys generate dp --output keys/dp/dp_params.json --epsilon 1.0 --delta 1e-5

## Build and start gnark proof service (required for ZKP modes)
cd zkp_gnark_service
go build -o gnark_service main.go
./gnark_service &          # runs in background on :9000
cd ..
```

#### 2. Run All 10 Modes

```bash
## Automated comparison — all 10 modes sequentially
python compare.py --dataset healthcare

## Or select specific modes
python compare.py --dataset healthcare --modes baseline,dp
python compare.py --dataset healthcare --modes he_tenseal,he_concrete_tfhe
python compare.py --dataset healthcare --modes zkp_sampled,he_tenseal_zkp,he_tenseal_zkp_dp

## MNIST / CIFAR-10 with non-IID Dirichlet partitioning
python compare.py --dataset mnist --dirichlet-alpha 0.1 --simulation
python compare.py --dataset cifar --dirichlet-alpha 0.5 --simulation

## Privacy-utility tradeoff: DP epsilon sweep
python compare.py --dataset healthcare --simulation --epsilon-sweep

## Non-IID heterogeneity: Dirichlet alpha sweep
python compare.py --dataset healthcare --simulation --alpha-sweep

## Choose blockchain backend (default: mock — writes JSON ledger locally)
python compare.py --dataset healthcare --chain-backend mock
python compare.py --dataset healthcare --chain-backend none   # disable ledger

## Statistical significance: repeat runs with different seeds
python compare.py --dataset healthcare --modes baseline,dp --seed 42 --simulation
python compare.py --dataset healthcare --modes baseline,dp --seed 123 --simulation
python compare.py --dataset healthcare --modes baseline,dp --seed 456 --simulation
python scripts/aggregate_statistics.py --root results/healthcare/
```

#### 3. View Results

Results are auto-merged into `./results/<dataset>/comparison_report.json` after each run:

```bash
## Pretty-print the merged report
cat results/healthcare/comparison_report.json | python -m json.tool

## Open the analysis notebook
jupyter notebook ../privacy_comparison_analysis.ipynb
```

Per-mode breakdowns:
```
results/healthcare/<timestamp>/
├── comparison_report.json           ← merged benchmark results (includes "seed" field per mode)
├── ledger_comparison.json           ← combined blockchain audit trail
├── ledgers/
│   ├── ledger_baseline.json
│   ├── ledger_dp.json
│   ├── ledger_he_tenseal.json
│   ├── ledger_he_concrete_tfhe.json
│   ├── ledger_zkp.json
│   ├── ledger_zkp_sampled.json
│   ├── ledger_he_tenseal_zkp.json
│   ├── ledger_he_concrete_tfhe_zkp.json
│   ├── ledger_he_tenseal_zkp_dp.json
│   └── ledger_he_concrete_tfhe_zkp_dp.json
├── baseline/benchmark.json
├── he_tenseal/benchmark.json
├── he_concrete_tfhe/benchmark.json
├── zkp_sampled/benchmark.json
├── zkp/benchmark.json
├── dp/benchmark.json
├── he_tenseal_zkp/benchmark.json
├── he_concrete_tfhe_zkp/benchmark.json
├── he_tenseal_zkp_dp/benchmark.json
└── he_concrete_tfhe_zkp_dp/benchmark.json

## Sweep experiment outputs
results/healthcare/
├── dp_eps_0.5/<timestamp>/benchmark_dp.json
├── dp_eps_1.0/<timestamp>/benchmark_dp.json
├── dp_eps_2.0/<timestamp>/benchmark_dp.json
├── dp_eps_3.0/<timestamp>/benchmark_dp.json
├── dp_eps_5.0/<timestamp>/benchmark_dp.json
├── dp_eps_8.0/<timestamp>/benchmark_dp.json
├── dp_epsilon_sweep_summary.json    ← consolidated epsilon sweep results
├── alpha_0.1/<timestamp>/comparison_report.json
├── alpha_0.5/<timestamp>/comparison_report.json
├── alpha_1.0/<timestamp>/comparison_report.json
├── alpha_10.0/<timestamp>/comparison_report.json
└── alpha_sweep_summary.json         ← consolidated alpha sweep results

## After multi-seed runs:
results/healthcare/
├── 20260312_100000/comparison_report.json   ← seed=42
├── 20260312_110000/comparison_report.json   ← seed=123
├── 20260312_120000/comparison_report.json   ← seed=456
└── statistical_summary.json                 ← mean ± std (from aggregate_statistics.py)
```

### Comparison Metrics

The comparison provides:

#### Performance Metrics
- **Total Training Time**: End-to-end FL execution time
- **Client Fit Time**: Average time per client training round
- **Server Aggregate Time**: Time for parameter aggregation
- **Communication Overhead**: Upload/download data volume

#### Privacy Metrics
- **Cryptographic Overhead**: Time spent on encryption/proofs/noise
- **Proof Generation**: Time to create ZKP proofs (ZKP mode)
- **Proof Verification**: Time to verify proofs (ZKP mode)
- **Encryption/Decryption**: Time for HE operations (HE mode)
- **DP Noise Addition**: Time for gradient noise (DP mode)

#### Model Quality Metrics  
- **Training Accuracy**: Local model accuracy per client
- **Validation Accuracy**: Model accuracy on validation set
- **Global Model Accuracy**: Aggregated model performance
- **Loss Convergence**: Initial vs final model loss

### Available Comparison Methods

#### 1. Compare Runner (Recommended — all 10 modes)

```bash
## All 10 modes on healthcare dataset
python compare.py --dataset healthcare

## Specific subset
python compare.py --dataset stock --modes baseline,he_tenseal,dp

## Triple modes only
python compare.py --dataset healthcare --modes he_tenseal_zkp_dp,he_concrete_tfhe_zkp_dp

## Image datasets
python compare.py --dataset mnist --simulation --modes baseline,dp
python compare.py --dataset cifar --simulation --modes baseline,he_tenseal

## Sweep experiments
python compare.py --dataset healthcare --simulation --epsilon-sweep   # ε ∈ {0.5,1,2,3,5,8}
python compare.py --dataset healthcare --simulation --alpha-sweep     # α ∈ {0.1,0.5,1,10}

## Non-IID fixed alpha
python compare.py --dataset mnist --dirichlet-alpha 0.1 --simulation

## Override DP epsilon for a single run
python compare.py --dataset healthcare --modes dp --dp-epsilon 0.5 --simulation

## With blockchain options
python compare.py --dataset healthcare --chain-backend mock   # default: writes ledger JSONs
python compare.py --dataset healthcare --chain-backend none   # disable blockchain audit
python compare.py --dataset healthcare --chain-ledger-dir /tmp/my_ledgers
```

**Blockchain Audit Ledger**

Every run records a per-round audit trail via `fl/chain.py`. Two backends are available:

| Backend | Description |
|---------|-------------|
| `mock` (default) | Writes JSON ledger files locally; zero external dependencies |
| `web3` | Submits transactions to a live EVM node (see `fl/config.py` for `chain_rpc_url`) |
| `none` | Disables the audit ledger entirely |

After every run the console prints a blockchain audit table:

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
TOTAL                        36           24           12
```

- **ModelCommit** — recorded every aggregation round for every mode
- **ProofAnchor** — recorded every round only for ZKP-containing modes; stores a hash of each client's proof

The combined ledger is saved to `results/<dataset>/<timestamp>/ledger_comparison.json`.

**What the runner does:**
1. Detects which modes need the gnark service; auto-starts it if not running
2. Runs each mode as a subprocess with real gRPC transport (Flower)
3. Collects `benchmark.json` from each mode
4. Merges results into `results/<dataset>/<timestamp>/comparison_report.json`
5. Merges per-mode chain ledger JSONs into `ledger_comparison.json`
6. Prints a formatted summary table and a blockchain audit table (ModelCommit + ProofAnchor counts per mode)

**Result merging**: Running the same mode twice overwrites its entry; other modes are preserved. Safe to re-run individual modes without losing others' data.

**Statistical Significance**

Each run uses a fixed random seed (default `42`). Pass `--seed` with different values across runs, then aggregate with `scripts/aggregate_statistics.py`:

```bash
## Repeated runs with different seeds
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 42  --simulation
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 123 --simulation
python compare.py --dataset healthcare --modes baseline,dp --rounds 5 --seed 456 --simulation

## Aggregate mean ± std (scans YYYYMMDD_HHMMSS dirs under --root)
python scripts/aggregate_statistics.py --root results/healthcare/
python scripts/aggregate_statistics.py --root results/healthcare/ --min-runs 5       # warn if < 5
python scripts/aggregate_statistics.py --root results/healthcare/ --modes baseline,dp # subset
python scripts/aggregate_statistics.py --root results/healthcare/ --no-plot          # skip PNG

## Or run the full multi-seed loop in one shot:
bash scripts/run_repeated_experiments.sh healthcare
bash scripts/run_repeated_experiments.sh creditcard 20 5   # dataset rounds clients
```

Outputs written to `results/<dataset>/`:

| File | Description |
|------|-------------|
| `statistical_summary.json` | Per-mode mean, std, min, max, 95% CI, N |
| `statistical_summary.png` | Grouped bar chart with ± std error bars (requires matplotlib) |

The seed used for each run is recorded in `comparison_report.json` under `"seed"` per mode entry.

#### 2. Docker Mode (Original 4 modes)

For containerized, reproducible runs of the original 4 modes (baseline/he/zkp/dp):

```bash
## Run all 4 original modes
bash scripts/run_docker_compare.sh

## Aggregate Docker results
docker compose --profile aggregate run --rm aggregate
```

See [README.md](README.md) for full Docker instructions.

#### 3. Simulation Mode (development/debug)

```bash
## Quick local test (no gRPC overhead)
python simulation.py simulation \
  --rounds 2 --number_clients 2 \
  --max_epochs 1 --benchmark

## With specific privacy mode
python simulation.py simulation --he --rounds 2 --benchmark
python simulation.py simulation --zkp --zkp_backend gnark --benchmark
python simulation.py simulation --dp --dp_params dp_params.json --benchmark
```

### Understanding Results

#### Typical Performance Profile (healthcare dataset, 3 rounds, 3 clients, Apple Silicon)

| Mode | Accuracy | Upload/round | Total Time | Enc+Dec | Proof Gen | Notes |
|------|----------|-------------|-----------|---------|----------|-------|
| **baseline** | ~87.3% | 0.01 MB | ~45s (1×) | 0s | 0s | Reference |
| **he_tenseal** | ~87.1% | **244.7 MB** | ~148s (3.3×) | ~1.5s | 0s | CKKS ciphertext expansion |
| **he_concrete_tfhe** | ~84.8% | 17.8 MB | ~310s (6.9×) | ~5.3s | 0s | Quantization loss |
| **zkp_sampled** | ~87.2% | 0.01 MB | ~520s (11.6×) | 0s | ~22.4s | Proving bottleneck |
| **zkp** | ~87.2% | 0.01 MB | ~1200s | 0s | ~52s | All layers proven |
| **dp** ε=1.0 | ~83.1% | 0.01 MB | ~48s (1.1×) | <0.1s | 0s | Noise loss |
| **he_tenseal_zkp** | ~87.0% | **244.7 MB** | ~670s (14.9×) | ~1.8s | ~22.4s | Dual overhead |
| **he_concrete_tfhe_zkp** | ~84.7% | 17.8 MB | ~480s (10.7×) | ~5.3s | ~22.4s | Best bandwidth+security |
| **he_tenseal_zkp_dp** | ~82.8% | **244.7 MB** | ~675s | ~1.8s | ~22.4s | Full triad |  
| **he_concrete_tfhe_zkp_dp** | ~82.3% | 17.8 MB | ~485s | ~5.3s | ~22.4s | Full triad, efficient |

#### Key Observations

**Bandwidth**: CKKS (TenSEAL) ciphertext expansion is ~24,000× over plaintext (0.01 MB → 244.7 MB). TFHE (Concrete) uses quantized int8 weights — 14× less bandwidth than CKKS.

**Latency**: ZKP proof generation dominates timing in ZKP-containing modes (~22s per client per round). Groth16 verification is cheap (~0.08s per client).

**Accuracy**: HE modes preserve accuracy (exact arithmetic, no noise). DP and TFHE modes trade accuracy for their respective benefits.

**DP note**: DP is the fastest non-baseline mode (<0.1s crypto overhead) and the only one providing formal information-theoretic (ε-DP) guarantees against membership inference on the published model. All HE and ZKP modes rely on computational security under mathematical hardness assumptions.

**Triple modes**: The `he_tenseal_zkp_dp` and `he_concrete_tfhe_zkp_dp` modes provide all three guarantees simultaneously. Their total overhead is the sum of HE, ZKP, and DP overheads — approximately the same as `he_*_zkp` since DP noise addition takes <0.1s. The accuracy penalty is additive: TFHE quantization loss plus DP noise loss.

### Implementation Status (March 2026)

| Mode | Status | Benchmarking | Chain Audit | Notes |
|------|--------|-------------|-------------|-------|
| baseline | ✅ Working | ✅ Full | ✅ ModelCommit | Reference mode |
| he_tenseal | ✅ Working | ✅ Full | ✅ ModelCommit | CKKS via TenSEAL + MS SEAL |
| he_concrete_tfhe | ✅ Working | ✅ Full | ✅ ModelCommit | TFHE via Concrete ML |
| zkp_sampled | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | Groth16 via gnark, sampled layers |
| zkp | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | Groth16 via gnark, all layers |
| dp | ✅ Working | ✅ Full | ✅ ModelCommit | Gaussian noise via Opacus; runtime ε override via sentinel |
| he_tenseal_zkp | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | CKKS + Groth16 combined |
| he_concrete_tfhe_zkp | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | TFHE + Groth16 combined |
| he_tenseal_zkp_dp | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | CKKS + Groth16 + DP-SGD (triple mode) |
| he_concrete_tfhe_zkp_dp | ✅ Working | ✅ Full | ✅ ModelCommit + ProofAnchor | TFHE + Groth16 + DP-SGD (triple mode) |

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ZKP modes fail: `Connection refused :9000` | gnark service not running | `cd zkp_gnark_service && ./gnark_service` |
| `proof_verification = 0.0` in results | Wrong ZKP backend (Pedersen used) | `export FL_ZKP_BACKEND=gnark` |
| TenSEAL: `scale out of bounds` | CKKS coefficient modulus overflow | Already fixed; ensure `global_scale=2^40` |
| TFHE accuracy 2–3% lower | Quantization error (int8 weights) | Expected trade-off |
| DP accuracy unchanged during `--epsilon-sweep` | `dp_epsilon` sentinel (10.0) used | Pass `--dp-epsilon` flag or use `--epsilon-sweep` |
| DP accuracy drops significantly | ε too small (strong noise) | Increase ε when generating DP params, e.g. `python -m fl.keys generate dp --epsilon 1.0` |
| `FileNotFoundError: keys/he_tenseal/secret_context.bin` | HE keys not generated | `python -m fl.keys generate he_tenseal` |
| `... is not a TenSEAL key file of this version` / `... is not a JSON parameter file` | key or parameter file from before pickle files were retired | regenerate it with `python -m fl.keys generate <mode>` |
| `FileNotFoundError: dp_params.json` | DP params not generated | `python -m fl.keys generate dp --output dp_params.json` |
| Port 8081–8084 busy | Docker port conflict | Change ports in `docker-compose.yml` |
| Blockchain table shows all zeros | Stale ledger from pre-fix run | Re-run; parser correctly unwraps `{"ledger": [...]}` format |
| `ledger_comparison.json` missing | `--chain-backend none` was set | Re-run without `--chain-backend none` (default is `mock`) |
| `he_tenseal_zkp_dp` not in default modes | Triple modes absent from mode list | Fixed: all 10 modes in `compare.py` default |
| CIFAR `key not found` | Registry used `cifar10` only | Fixed: `@register_dataset("cifar")` alias added |
| MNIST 0-byte file on parallel download | Race condition in parallel extract | Fixed via `fcntl.flock` exclusive lock in `mnist.py` |
| Client 2 IndexError on startup | `--number_clients` missing from subprocess | Fixed in `experiment.py` `common_args` |
| CIFAR CNN shape mismatch | `in_channels` hardcoded to 1 | Fixed: `Net` uses dynamic `in_channels` + computed `flat_dim` |

### References

- **Flower Framework**: https://flower.ai
- **TenSEAL (CKKS HE)**: https://github.com/OpenMined/TenSEAL
- **Concrete ML (TFHE)**: https://github.com/zama-ai/concrete-ml
- **gnark (Groth16 ZKP)**: https://github.com/consensys/gnark
- **Opacus (DP-SGD)**: https://opacus.ai

---

### Environment Variables Reference

All tuning is done via environment variables — no code changes required. Variables are read at process startup and forwarded to client/server subprocesses by the compare runner.

#### ZKP / Proof Variables

| Variable | Default | Values | Effect on results |
|----------|---------|--------|------------------|
| `FL_ZKP_BACKEND` | `gnark` | `gnark`, `pedersen` | `gnark` = Groth16 zk-SNARK with soundness guarantee; `pedersen` = legacy commitment without soundness |
| `FL_ZKP_SELECT_BY` | `size` | `size`, `random` | `size` picks the largest layers (strongest norm-bound coverage per round); `random` rotates coverage across rounds |
| `FL_ZKP_NUM_LAYERS` | `1` | Positive integer | Layers proven per client per round. Lower = faster proofs, weaker per-round coverage. `1` is suitable for 2-layer models |
| `FL_ZKP_SAMPLE_PCT` | — | Float 0–1 | Alternative to `FL_ZKP_NUM_LAYERS`: prove this fraction of layers (ceiling). Useful when model depth varies |
| `FL_ZKP_SAMPLE_SEED` | — | Integer | Fixes layer sampling for reproducible benchmarks; omit to vary across rounds |
| `FL_ZKP_PARALLELISM` | `4` | Positive integer | Concurrent proof workers. Increase for high-core servers; diminishing returns above gnark host CPU count |
| `FL_ZKP_SCALE` | `1000000` | Positive integer | Float→int64 scale. Too low = precision loss; too high = integer overflow |
| `FL_ZKP_MAX_NORM` | calibrated | Positive float | Server's update-norm bound B on ‖w_local − w_global‖₂ (overrides the per-dataset calibration in `fl/core/update_bound.py`). Not the DP clipping norm |
| `FL_ZKP_TIMEOUT` | `120` | Seconds | Per-call timeout for the gnark HTTP service. Increase for large models or first-run compilation |
| `FL_ZKP_LAYERS` | `ALL` | `ALL` or CSV names | Layers to prove in full (non-sampled) ZKP mode |

#### HE Encryption Variables

| Variable | Default | Values | Effect on results |
|----------|---------|--------|------------------|
| `FL_ENCRYPT_LAYERS` | `model.0.weight,model.0.bias` | CSV layer names or `ALL` | Which layers to encrypt. `ALL` = full gradient privacy (maximum bandwidth); partial list = faster upload but plaintext leakage of remaining layers |
| `FL_CONCRETE_TFHE_BIT_WIDTH` | `14` | Integer 2–16 | TFHE quantization bit width. `8` ≈ 2–3% accuracy drop; `14` ≈ negligible loss; `16` ≈ lossless. Higher = larger ciphertexts and slower encryption |
| `FL_CONCRETE_TFHE_ADAPTIVE_QUANT` | `0` | `0`, `1` | `1` enables per-layer quantization scale fitting; recovers ~0.5–1% accuracy on heterogeneous models at ~5–10% overhead |

#### Transport / Timing Variables

| Variable | Default | Values | Effect on results |
|----------|---------|--------|------------------|
| `FL_GRPC_MAX_MESSAGE_LENGTH` | `2147483647` | Bytes (integer) | Max gRPC payload size. Default (2 GiB) covers TenSEAL CKKS ciphertexts (~245 MB/round). Lower values cause `RESOURCE_EXHAUSTED` errors with HE modes |
| `FL_CLIENT_TIMEOUT` | `7200` | Seconds | How long the server waits for all clients to complete per run. HE + ZKP modes can take 10+ minutes per round; set ≥ `num_rounds × max_round_time` |
| `FL_NUMBER_CLIENTS` | Set by `--num-clients` | Integer | Expected client count. Set automatically by the compare runner; override only in manual deployments |

#### Quick Export Block

```bash
## ZKP tuning
export FL_ZKP_SELECT_BY=size
export FL_ZKP_NUM_LAYERS=1
export FL_ZKP_PARALLELISM=4
export FL_ZKP_BACKEND=gnark

## HE tuning
export FL_ENCRYPT_LAYERS=model.0.weight,model.0.bias
export FL_CONCRETE_TFHE_BIT_WIDTH=14
export FL_CONCRETE_TFHE_ADAPTIVE_QUANT=0

## Transport
export FL_GRPC_MAX_MESSAGE_LENGTH=2147483647
export FL_CLIENT_TIMEOUT=7200

## Then simply:
python compare.py --dataset stock
```

> Full per-variable tuning details: [ZKP.md § 13 Configuration Reference](ZKP.md) · [FHE.md § HE Environment Variables Reference](FHE.md)

---

---

### Supported Datasets (all modes)

| Key | Description | Classes | Input Shape | Notes |
|-----|-------------|---------|-------------|-------|
| `healthcare` | Clinical binary classification (918 samples, 13 features) | 2 | Tabular | Stratified split, IID default |
| `creditcard` | Credit card fraud detection (284,807 transactions, imbalanced) | 2 | Tabular | Severe class imbalance |
| `stock` | Stock trend prediction (multi-class tabular) | 3 | Tabular | Multi-round default |
| `mnist` | MNIST handwritten digits (70,000 samples) | 10 | 1×28×28 | CNN; `fcntl.flock` download |
| `cifar` / `cifar10` | CIFAR-10 object recognition (60,000 samples) | 10 | 3×32×32 | CNN; both aliases work |

All datasets support **non-IID Dirichlet partitioning** via `--dirichlet-alpha α`. The Dirichlet(α) distribution assigns label proportions across clients:
- α = 0.1 → extreme non-IID (each client has data from ~1–2 classes)
- α = 0.5 → moderate heterogeneity
- α = 1.0 → mild heterogeneity  
- α = 10.0 → near-IID (each client has roughly uniform label distribution)

Setting `--dirichlet-alpha` to `None` (default) uses stratified IID partitioning across all clients.

**Last Updated:** March 2026  
**Status:** All 10 modes verified working ✅


---

## Docker Guide: Containerized Federated Learning

This guide covers running the Flower server and clients in Docker for the four Docker-based modes (`baseline`, `he_tenseal`, `zkp_sampled`, `dp`). The remaining six modes (`he_concrete_tfhe`, `zkp`, `he_tenseal_zkp`, `he_concrete_tfhe_zkp`, `he_tenseal_zkp_dp`, and `he_concrete_tfhe_zkp_dp`) run natively via `compare.py` — see [README.md](README.md).

> **Quick summary**: Docker covers 4 of the 10 modes. For all 10 modes, use the Python runner (no Docker required).

### Prerequisites

- Docker Desktop or Docker Engine 20.10+
- docker compose v2
- Local CIFAR-10 data present under `./data/cifar/` (see `README.md` for dataset setup)

### Build the Image

```bash
docker compose build
```

### One-shot: Run All 4 Docker Modes Sequentially

This will initialize keys/params, then run baseline, he_tenseal, zkp_sampled, and dp sequentially. Results are saved under `./results/<mode>`.

```bash
bash scripts/run_docker_compare.sh
```

After containers exit, aggregate the results into a single report and plot (using Docker, no local matplotlib needed):

```bash
docker compose --profile aggregate run --rm aggregate
```

This generates:
- `./results/docker_compare/comparison_report.json` - Detailed metrics
- `./results/docker_compare/comparison.png` - Visual comparison plots

### Run a Specific Mode (manually)

1. Initialize keys/params once:

```bash
docker compose --profile init run --rm init
```

2. Baseline:

```bash
docker compose --profile baseline up --abort-on-container-exit
docker compose --profile baseline down -v
```

3. Homomorphic Encryption (TenSEAL):

```bash
docker compose --profile he up --abort-on-container-exit
docker compose --profile he down -v
```

4. Zero-Knowledge Proofs:

```bash
docker compose --profile zkp up --abort-on-container-exit
docker compose --profile zkp down -v
```

5. Differential Privacy:

```bash
docker compose --profile dp up --abort-on-container-exit
docker compose --profile dp down -v
```

### Notes

- Networking: Clients connect to the server using the service hostname (e.g., `server_he:8082`) via `FL_SERVER_ADDRESS`.
- Healthchecks: Clients wait for the server port to be open via simple TCP checks.
- Keys/Params: use the unified CLI `fl.keys` (e.g. `python -m fl.keys generate he_tenseal`, `python -m fl.keys generate zkp`, `python -m fl.keys generate dp`) to write artifacts into the project directory mounted into all services.
- Results: Per-client training curves and benchmarks are saved under `./results/<mode>`.
- Aggregation/Plots: You can reuse `compare_methods_simple.py` to generate comparison plots from fresh runs, or adapt a simple aggregator to read `client_*_benchmark.json` files.
	- Included: `scripts/aggregate_results.py` to aggregate Docker-run results and generate `comparison_report.json` and `comparison.png`.
	- `scripts/aggregate_statistics.py` — compute mean ± std across repeated runs (different `--seed` values). Scans `YYYYMMDD_HHMMSS` dirs under `--root` and writes `statistical_summary.json` and `statistical_summary.png`.
	- `scripts/run_repeated_experiments.sh` — convenience wrapper that runs the full experiment loop N times with different seeds and then calls `aggregate_statistics.py` automatically.

### Running All 10 Modes (Without Docker)

The Python runner supports all 10 modes natively without Docker:

```bash
cd fl_ppml

## One-time setup
python -m fl.keys generate he_tenseal --overwrite
python -m fl.keys generate dp --overwrite
python -m fl.keys generate zkp --overwrite
cd zkp_gnark_service && go build -o gnark_service main.go && ./gnark_service & cd ..

## Run all 10 modes
python compare.py --dataset healthcare

## Results at: results/healthcare/<timestamp>/comparison_report.json
```

For gnark setup details see [ZKP.md](ZKP.md).

### Troubleshooting

- If ports 8081–8084 are in use, change the published ports in `docker-compose.yml`.
- TenSEAL or Concrete-ML build errors: the provided image installs `build-essential` and `cmake`. Ensure sufficient memory for building wheels.
- Dataset not found: Ensure CIFAR-10 exists under `./data/cifar/` as documented.
- ZKP mode in Docker: the gnark binary must be present at runtime. Build it before `docker compose build` and copy into the image, or run zkp_sampled natively via the Python runner.


---

## Credit Card Fraud Detection - Federated Learning Comparison

### Overview

This script demonstrates privacy-preserving federated learning on the **Credit Card Fraud Detection** dataset (284,807 transactions), showing clear timing differences between privacy methods at scale.

#### Dataset Details
- **Size**: 284,807 transactions (492 frauds, 284,315 legitimate)
- **Features**: 30 (Time, V1-V28 PCA components, Amount)
- **Privacy Sensitivity**: HIGH - Financial transaction data (PCI-DSS compliance)
- **FL Use Case**: Banks collaboratively train fraud models without sharing customer data

### Quick Start

#### 1. Basic Comparison (Fast Test)
```bash
## Test with 5,000 samples
python run_creditcard_comparison.py --subsample 5000 --modes baseline,zkp
```

#### 2. Full Dataset Comparison (Shows Clear Timing Differences)
```bash
## All privacy modes on full dataset (~284k samples)
python run_creditcard_comparison.py --modes baseline,he_tenseal,he_concrete_tfhe,zkp_sampled,dp
```

#### 3. gnark ZKP Backend (Recommended)
```bash
## Start gnark service first
cd zkp_gnark_service && ./gnark_service

## Run comparison with gnark
export FL_ZKP_BACKEND=gnark
python run_creditcard_comparison.py --modes baseline,zkp,dp
```

### Expected Performance

#### With Full Dataset (284,807 samples, 3 rounds, 2 clients):
- **Baseline**: ~30-60 seconds ⚡
- **HE_TenSEAL**: ~10-15 minutes 🔒 (encryption overhead)
- **HE_Concrete_TFHE**: ~15-25 minutes 🔐 (FHE circuit evaluation)
- **ZKP Sampled (gnark)**: ~12-18 minutes ✅ (proof generation scales with data)
- **DP**: ~30-60 seconds 🎲 (noise addition is fast)

#### With Subsample (5,000 samples):
- **Baseline**: ~5-10 seconds
- **HE_TenSEAL**: ~1-2 minutes
- **ZKP Sampled (gnark)**: ~1-2 minutes
- **DP**: ~5-10 seconds

### Command-Line Options

```bash
python run_creditcard_comparison.py [OPTIONS]

Options:
  --modes MODES              Comma-separated privacy modes
                            (default: baseline,he_tenseal,he_concrete_tfhe,zkp_sampled,dp)
  
  --num_clients N           Number of federated clients (default: 2)
  --rounds N                Number of FL rounds (default: 3)
  --subsample N             Use only N samples for faster testing
  
  --zkp_backend {gnark,pedersen}  ZKP backend (default: pedersen)
```

### Examples

#### Timing Demonstration
```bash
## Show dramatic timing differences at scale
python run_creditcard_comparison.py \
  --modes baseline,he_tenseal,zkp_sampled \
  --num_clients 3 \
  --rounds 2
```

#### Quick Test During Development
```bash
## Fast iteration with 10k samples
python run_creditcard_comparison.py \
  --subsample 10000 \
  --modes baseline,dp \
  --rounds 2
```

#### Production-Like Scenario
```bash
## 5 banks, 5 rounds, full dataset
python run_creditcard_comparison.py \
  --num_clients 5 \
  --rounds 5 \
  --modes baseline,he_tenseal,zkp_sampled,dp
```

### Results

Results are saved to `./results/creditcard_comparison/`:
- `comparison.png` - Visual comparison charts
- `comparison_report.json` - Detailed metrics (timing, accuracy, crypto overhead)
- Subdirectories for each mode with detailed logs

### Privacy Justification

**Why Federated Learning for Fraud Detection?**

1. **Regulatory Compliance**: Banks cannot share customer transaction data (PCI-DSS, GDPR)
2. **Competitive Advantage**: Banks don't want to reveal their fraud detection strategies
3. **Improved Models**: Cross-institution collaboration improves fraud detection accuracy
4. **Real-World Impact**: Protects millions of customers from financial fraud

### Dataset Source

Credit Card Fraud Detection Dataset:
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Location**: `../dataset/kaggle/input/mlg-ulb/creditcard.csv`
- **Citation**: Machine Learning Group - ULB (Université Libre de Bruxelles)

### Comparison with Healthcare Dataset

| Metric | Healthcare | Credit Card |
|--------|------------|-------------|
| Samples | 303 | 284,807 |
| Features | 13 | 30 |
| Privacy Level | High (HIPAA) | High (PCI-DSS) |
| Timing Difference | Small | **Large** ⭐ |
| Best For | Privacy concepts | **Timing demonstration** |

**Recommendation**: Use credit card dataset to demonstrate that privacy-preserving FL can scale to real-world production datasets.

### Troubleshooting

#### Out of Memory
```bash
## Use subsample for testing
python run_creditcard_comparison.py --subsample 50000
```

#### Too Slow
```bash
## Test with baseline and DP only (both fast)
python run_creditcard_comparison.py --modes baseline,dp --rounds 2
```

#### Dataset Not Found
```bash
## Place creditcard.csv at:
## dataset/kaggle/input/mlg-ulb/creditcard.csv
```

### Integration with Existing Scripts

This script uses the same infrastructure as `run_healthcare_comparison.py`:
- All privacy modes supported
- Same FL framework (Flower)
- Compatible with all existing benchmarking tools
- Drop-in replacement - just change `--dataset creditcard`
