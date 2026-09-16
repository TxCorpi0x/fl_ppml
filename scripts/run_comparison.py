#!/usr/bin/env python3
"""
Run a comparison of multiple FL privacy modes.

Examples::

    python scripts/run_comparison.py --modes baseline he_concrete_tfhe zkp dp
    python scripts/run_comparison.py --modes all --dataset creditcard --rounds 3
    python scripts/run_comparison.py --modes baseline --clients 5 --rounds 5 --epochs 2
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure fl_ppml/ is on sys.path when running as a plain script
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


ALL_MODES = ["baseline", "he_tenseal", "he_concrete_tfhe", "zkp", "dp"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FL privacy modes")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline"],
        help="Modes to run, or 'all' for all 6 modes",
    )
    parser.add_argument("--dataset", default="creditcard")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./results/")
    args = parser.parse_args()

    modes = ALL_MODES if "all" in args.modes else args.modes

    from ppflx import FLConfig
    from ppflx_bench.runner import run_comparison

    config = FLConfig(
        dataset=args.dataset,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        device=args.device,
        seed=args.seed,
    )

    run_comparison(modes=modes, config=config, output_dir=args.output)


if __name__ == "__main__":
    main()
