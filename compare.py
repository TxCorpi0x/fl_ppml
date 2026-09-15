#!/usr/bin/env python3
"""
compare.py — Federated learning privacy-mode comparison CLI.

Examples
--------
# Run all modes for the healthcare dataset (real gRPC):
python compare.py --dataset healthcare

# Run specific modes with fewer rounds:
python compare.py --dataset creditcard --modes baseline,zkp,dp --rounds 3

# Simulation mode (Flower in-process, useful for quick testing):
python compare.py --dataset stock --simulation --num-clients 3

# Override epochs / batch size:
python compare.py --dataset healthcare --max-epochs 2 --batch-size 8

# Blockchain audit ledger (default: mock — zero deps, always on):
python compare.py --dataset healthcare --simulation --chain-backend mock
# Use an Ethereum node instead:
python compare.py --dataset healthcare --simulation --chain-backend web3
# Disable the ledger entirely:
python compare.py --dataset healthcare --simulation --chain-backend none

# Show available datasets and modes then exit:
python compare.py --list
"""
from __future__ import annotations

import warnings

# Suppress DeprecationWarnings from third-party libraries (skorch, setuptools)
# that use the deprecated distutils.version.LooseVersion class.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"skorch.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"setuptools.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"distutils.*")

import argparse
import sys


def _list_registry() -> None:
    from fl.compare.registry import DATASETS, MODES

    print("\nDatasets:")
    for name, ds in DATASETS.items():
        print(f"  {name:<15} epochs={ds.max_epochs:<3} batch={ds.batch_size:<4}")

    print("\nModes:")
    for name, m in MODES.items():
        req = f"  (requires: {m.requires_key})" if m.requires_key else ""
        print(f"  {name:<22} {m.display_name:<35}{req}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare federated learning privacy modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--dataset",
        default="healthcare",
        help="Dataset key (cifar, healthcare, creditcard, stock). Default: healthcare",
    )
    p.add_argument(
        "--modes",
        default="baseline,dp,he_tenseal,he_concrete_tfhe,zkp,zkp_sampled,he_tenseal_zkp,he_concrete_tfhe_zkp,he_tenseal_zkp_dp,he_concrete_tfhe_zkp_dp",
        help="Comma-separated mode keys to run. Default: all 10 modes.",
    )
    p.add_argument(
        "--num-clients",
        type=int,
        default=3,
        metavar="N",
        help="Number of FL clients, at least 2 (default: 3)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of FL rounds, at least 1 (default: 20). "
        "Use 2+ to exercise decrypting an aggregate before the next proven upload.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Training epochs per round. Default: from dataset registry.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Default: from dataset registry.",
    )
    p.add_argument(
        "--device", default="cpu", help="Compute device: cpu or cuda (default: cpu)"
    )
    p.add_argument(
        "--data-path",
        default="../dataset/",
        help="Root data directory (default: ../dataset/ — the centralised dataset/ folder at project root)",
    )
    p.add_argument(
        "--output-dir",
        default="results/",
        help="Root output directory (default: results/)",
    )
    p.add_argument(
        "--simulation",
        action="store_true",
        help=(
            "Use Flower in-process simulation instead of real gRPC server + clients. "
            "HE modes then transport plaintext (their results are marked [SIM])."
        ),
    )
    p.add_argument(
        "--chain-backend",
        default="mock",
        choices=["mock", "web3", "none"],
        help="Blockchain audit ledger backend (default: mock). 'none' disables.",
    )
    p.add_argument(
        "--chain-ledger-dir",
        default=None,
        metavar="DIR",
        help="Directory for per-mode ledger JSON files (default: --output-dir).",
    )
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        metavar="ALPHA",
        help=(
            "Dirichlet concentration for non-IID data partitioning "
            "(e.g. 0.1=extreme non-IID, 10.0=near-IID). "
            "Default: None (uniform IID split)."
        ),
    )
    p.add_argument(
        "--alpha-sweep",
        action="store_true",
        help=(
            "Run a Dirichlet alpha sweep over α∈{0.1, 0.5, 1.0, 10.0} for each "
            "selected mode. Results are saved to separate subdirectories."
        ),
    )
    p.add_argument(
        "--epsilon-sweep",
        action="store_true",
        help=(
            "Run a DP epsilon sweep over ε∈{0.5, 1.0, 2.0, 3.0, 5.0, 8.0}. "
            "Only DP mode is run. Generates the privacy-utility tradeoff curve. "
            "Results are saved to separate subdirectories."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and modes then exit.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). "
        "Change seed across runs to estimate variance.",
    )
    p.add_argument(
        "--no-validate-zkp",
        dest="validate_zkp",
        action="store_false",
        help="Skip ledger-based validation that every client emitted proofs in "
        "every round of ZKP-family modes. Unvalidated runs are not publication evidence.",
    )
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _list_registry()
        return 0

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else None

    import signal
    from fl.compare.experiment import cleanup_all_procs
    from fl.compare.runner import run_comparison, run_alpha_sweep, run_dp_epsilon_sweep

    def _sighandler(signum, frame):
        print(
            "\n[WARN]  Interrupted — stopping all child processes...", file=sys.stderr
        )
        cleanup_all_procs()
        sys.exit(130)

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    try:
        if args.epsilon_sweep:
            run_dp_epsilon_sweep(
                dataset=args.dataset,
                modes=modes,
                num_clients=args.num_clients,
                num_rounds=args.rounds,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=args.device,
                data_path=args.data_path,
                output_dir=args.output_dir,
                use_simulation=args.simulation,
                chain_backend=args.chain_backend,
                chain_ledger_dir=args.chain_ledger_dir,
                seed=args.seed,
            )
        elif args.alpha_sweep:
            run_alpha_sweep(
                dataset=args.dataset,
                modes=modes,
                num_clients=args.num_clients,
                num_rounds=args.rounds,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=args.device,
                data_path=args.data_path,
                output_dir=args.output_dir,
                use_simulation=args.simulation,
                chain_backend=args.chain_backend,
                chain_ledger_dir=args.chain_ledger_dir,
                seed=args.seed,
            )
        else:
            run_comparison(
                dataset=args.dataset,
                modes=modes,
                num_clients=args.num_clients,
                num_rounds=args.rounds,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=args.device,
                data_path=args.data_path,
                output_dir=args.output_dir,
                use_simulation=args.simulation,
                chain_backend=args.chain_backend,
                chain_ledger_dir=args.chain_ledger_dir,
                dirichlet_alpha=args.dirichlet_alpha,
                validate_zkp=args.validate_zkp,
                seed=args.seed,
            )
    except KeyboardInterrupt:
        print(
            "\n[WARN]  Interrupted — stopping all child processes...", file=sys.stderr
        )
        cleanup_all_procs()
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        cleanup_all_procs()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
