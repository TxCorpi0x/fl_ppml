#!/usr/bin/env bash
set -euo pipefail

# Run all non-simulation modes in Docker using compose profiles.
# Requires Docker Desktop/Engine and docker-compose v2.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "[1/5] Building image..."
docker compose build

echo "[2/5] Initializing keys and params..."
docker compose --profile init run --rm init

echo "[3/5] Running BASELINE..."
docker compose --profile baseline up --abort-on-container-exit --quiet-pull
docker compose --profile baseline down -v

echo "[4/5] Running HE_TENSEAL..."
docker compose --profile he up --abort-on-container-exit --quiet-pull
docker compose --profile he down -v

echo "[5/5] Running ZKP..."
docker compose --profile zkp up --abort-on-container-exit --quiet-pull
docker compose --profile zkp down -v

echo "[6/5] Running DP..."
docker compose --profile dp up --abort-on-container-exit --quiet-pull
docker compose --profile dp down -v

echo "[OK] All modes completed. Results in ./results/{baseline,he_tenseal,zkp,dp}"
echo "Aggregating results inside Docker..."
docker compose --profile aggregate run --rm aggregate || true
docker compose --profile aggregate down -v || true

echo "Optionally aggregate/plot using compare_methods_simple.py or your own tooling."
