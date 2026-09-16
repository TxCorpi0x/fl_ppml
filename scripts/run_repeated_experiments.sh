#!/usr/bin/env bash
# scripts/run_repeated_experiments.sh
#
# Run repeated FL experiments for statistical significance.
# Usage:
#   bash scripts/run_repeated_experiments.sh healthcare
#   bash scripts/run_repeated_experiments.sh creditcard
#   bash scripts/run_repeated_experiments.sh mnist
#
# After all runs complete:
#   python scripts/aggregate_statistics.py --root results/<dataset>/

set -e

# ---------------------------------------------------------------------------
# Pre-run prerequisite checks
# ---------------------------------------------------------------------------
PREREQ_FAIL=0

# 1. TenSEAL keys  (he_tenseal, he_tenseal_zkp, triple modes)
if [[ ! -f keys/he_tenseal/secret_context.bin ]]; then
    echo "[PREREQ] TenSEAL keys not found. Run: python -m ppflx.keys generate he_tenseal --overwrite"
    PREREQ_FAIL=1
fi

# 2. DP params  (dp, he_*_zkp_dp triple modes)
if [[ ! -f keys/dp/dp_params.json ]]; then
    echo "[PREREQ] DP params not found. Run: python -m ppflx.keys generate dp --output keys/dp/dp_params.json --overwrite"
    PREREQ_FAIL=1
fi

# 3. gnark ZKP service  (zkp_sampled, he_*_zkp, triple modes)
if ! pgrep -x "gnark_service" > /dev/null 2>&1; then
    echo "[PREREQ] gnark ZKP service is not running."
    echo "         Build and start it with:"
    echo "           cd zkp_gnark_service && go build -o gnark_service main.go && ./gnark_service &"
    PREREQ_FAIL=1
fi

if [[ $PREREQ_FAIL -eq 1 ]]; then
    echo ""
    echo "ERROR: One or more prerequisites are missing (see above)."
    echo "       Modes that depend on them will be silently skipped, producing"
    echo "       N=0 entries in the statistical summary."
    echo "       Fix the issues above, then re-run this script."
    exit 1
fi

DATASET="${1:-healthcare}"
ROUNDS="${2:-20}"
NUM_CLIENTS="${3:-3}"
OUTPUT_DIR="${4:-results/}"

echo "=================================================="
echo "  Repeated Experiments for Statistical Significance"
echo "  Dataset: $DATASET  |  Rounds: $ROUNDS  |  Clients: $NUM_CLIENTS"
echo "=================================================="

# Fast modes: 5 runs (baseline, dp — cheap to run)
FAST_MODES="baseline,dp"
FAST_SEEDS="42 123 456 789 1337"

echo ""
echo "Fast modes ($FAST_MODES) — 5 runs..."
for SEED in $FAST_SEEDS; do
    echo "  Seed $SEED..."
    python compare.py \
        --dataset "$DATASET" \
        --modes "$FAST_MODES" \
        --rounds "$ROUNDS" \
        --num-clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --simulation
done

# Medium modes: 3 runs (he_tenseal, zkp_sampled)
MEDIUM_MODES="he_tenseal,zkp_sampled"
MEDIUM_SEEDS="42 123 456"

echo ""
echo "Medium modes ($MEDIUM_MODES) — 3 runs..."
for SEED in $MEDIUM_SEEDS; do
    echo "  Seed $SEED..."
    python compare.py \
        --dataset "$DATASET" \
        --modes "$MEDIUM_MODES" \
        --rounds "$ROUNDS" \
        --num-clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --simulation
done

# Slow/expensive modes: 3 runs (triple modes)
SLOW_MODES="he_tenseal_zkp_dp,he_concrete_tfhe_zkp_dp"
SLOW_SEEDS="42 123 456"

echo ""
echo "Triple modes ($SLOW_MODES) — 3 runs..."
for SEED in $SLOW_SEEDS; do
    echo "  Seed $SEED..."
    python compare.py \
        --dataset "$DATASET" \
        --modes "$SLOW_MODES" \
        --rounds "$ROUNDS" \
        --num-clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --simulation
done

echo ""
echo "=================================================="
echo "  All runs complete. Aggregating statistics..."
echo "=================================================="
python scripts/aggregate_statistics.py --root "${OUTPUT_DIR}${DATASET}/"
echo "Done."
