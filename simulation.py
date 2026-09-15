#!/usr/bin/env python3
"""
Simulation entrypoint — thin wrapper around fl.runner.run_mode().

Called by compare.py as::

    python simulation.py simulation --number_clients N --rounds N \\
        --dataset creditcard --max_epochs 1 --batch_size 32 ...

All heavy lifting is delegated to fl/runner.py; this file only parses the
legacy CLI flags and maps them to the clean FLConfig + mode_name API.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ── Argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FL simulation runner")
    subparsers = parser.add_subparsers(dest="subcommand")

    sim = subparsers.add_parser("simulation", help="Run Flower in-process simulation")

    # Training / federation basics
    sim.add_argument("--number_clients", type=int, default=2)
    sim.add_argument("--rounds", type=int, default=3)
    sim.add_argument("--max_epochs", type=int, default=1)
    sim.add_argument("--batch_size", type=int, default=32)
    sim.add_argument("--lr", type=float, default=0.001)
    sim.add_argument("--dataset", type=str, default="creditcard")
    sim.add_argument("--data_path", type=str, default="./data/")
    sim.add_argument("--device", type=str, default="cpu")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--num_workers", type=int, default=0)

    # Federation sampling
    sim.add_argument("--frac_fit", type=float, default=1.0)
    sim.add_argument("--frac_eval", type=float, default=0.5)
    sim.add_argument("--min_fit_clients", type=int, default=2)
    sim.add_argument("--min_eval_clients", type=int, default=None)
    sim.add_argument("--min_avail_clients", type=int, default=2)

    # Privacy-mode flags (legacy API — see _resolve_mode())
    sim.add_argument("--he", action="store_true", default=False)
    sim.add_argument(
        "--he_backend",
        type=str,
        default="tenseal",
        choices=["tenseal", "concrete", "concrete_tfhe"],
    )
    sim.add_argument("--path_keys", type=str, default="keys/he_tenseal/secret_context.bin")
    sim.add_argument(
        "--path_public_key", type=str, default="keys/he_tenseal/public_context.bin"
    )
    sim.add_argument("--zkp", action="store_true", default=False)
    sim.add_argument("--zkp_backend", type=str, default="gnark")
    sim.add_argument(
        "--privacy_mode",
        type=str,
        default=None,
        help="Registered privacy mode name; overrides the --he/--zkp/--dp flag combination.",
    )
    sim.add_argument("--zkp_params", type=str, default="keys/zkp/zkp_params.json")
    sim.add_argument("--dp", action="store_true", default=False)
    sim.add_argument("--dp_params", type=str, default="keys/dp/dp_params.json")
    sim.add_argument(
        "--dp_epsilon",
        type=float,
        default=None,
        help="Override DP privacy budget ε (default: use pre-generated key params).",
    )

    # Non-IID Dirichlet partitioning
    sim.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet concentration for non-IID partitioning (None=IID).",
    )

    # Output / benchmarking
    sim.add_argument("--benchmark", action="store_true", default=False)
    sim.add_argument("--save_results", type=str, default="./results/")
    sim.add_argument("--model_save", type=str, default="./model.pth")

    # Blockchain audit ledger
    sim.add_argument(
        "--chain_backend",
        type=str,
        default="mock",
        choices=["mock", "web3", "none"],
        help="Blockchain ledger backend (default: mock)",
    )
    sim.add_argument(
        "--chain_ledger_path",
        type=str,
        default="",
        help="Path for the ledger JSON file (default: <save_results>/ledger.json)",
    )

    # Legacy paths accepted for backward-compat but unused by the new pipeline
    sim.add_argument("--yaml_path", type=str, default="")
    sim.add_argument("--roc_path", type=str, default="")
    sim.add_argument("--matrix_path", type=str, default="")
    sim.add_argument("--split", type=int, default=10)
    sim.add_argument("--length", type=int, default=32)
    sim.add_argument("--path_crypted", type=str, default="")
    sim.add_argument("--data_path_val", type=str, default="")

    return parser


# ── Mode resolution ───────────────────────────────────────────────────────────


def _resolve_mode(args) -> str:
    """Map legacy --he / --zkp / --dp flags to a mode_name string."""
    if args.he and args.zkp and args.dp:
        # Triple combination: FHE + ZKP + DP
        backend = (args.he_backend or "tenseal").lower()
        if backend == "elgamal":
            raise ValueError("he_backend 'elgamal' does not support --dp")
        if backend in ("concrete_tfhe", "concrete"):
            return "he_concrete_tfhe_zkp_dp"
        return "he_tenseal_zkp_dp"
    if args.he and args.zkp:
        # Hybrid FHE + ZKP mode
        backend = (args.he_backend or "tenseal").lower()
        if backend == "elgamal":
            return "he_elgamal_zkp"
        if backend in ("concrete_tfhe", "concrete"):
            return "he_concrete_tfhe_zkp"
        return "he_tenseal_zkp"
    if args.he:
        backend = (args.he_backend or "tenseal").lower()
        if backend == "concrete_tfhe":
            return "he_concrete_tfhe"
        if backend == "concrete":
            return "he_concrete_tfhe"
        return "he_tenseal"
    if args.zkp:
        return "zkp"
    if args.dp:
        return "dp"
    return "baseline"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand != "simulation":
        parser.print_help()
        sys.exit(1)

    os.environ["FL_SIMULATION"] = "1"
    os.environ["FL_NUMBER_CLIENTS"] = str(args.number_clients)

    from fl import FLConfig
    from fl.runner import run_mode

    mode_name = args.privacy_mode or _resolve_mode(args)

    config = FLConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        num_clients=args.number_clients,
        num_rounds=args.rounds,
        local_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        frac_fit=args.frac_fit,
        frac_eval=args.frac_eval,
        min_fit_clients=args.min_fit_clients,
        min_eval_clients=args.min_eval_clients,
        min_avail_clients=args.min_avail_clients,
        privacy_mode=mode_name,
        he_tenseal_secret_path=args.path_keys,
        he_tenseal_public_path=args.path_public_key,
        zkp_backend=args.zkp_backend,
        zkp_params_path=args.zkp_params,
        dp_params_path=args.dp_params,
        results_dir=args.save_results or "./results/",
        model_save=args.model_save or "./model.pth",
        benchmark=args.benchmark,
        sim_mode=True,
        dirichlet_alpha=getattr(args, "dirichlet_alpha", None),
        dp_epsilon=args.dp_epsilon if args.dp_epsilon is not None else 10.0,
        chain_backend=args.chain_backend if args.chain_backend != "none" else "none",
        chain_ledger_path=(
            args.chain_ledger_path
            or os.path.join(args.save_results or "./results/", "ledger.json")
        ),
    )

    output_dir = args.save_results or "./results/"
    run_mode(mode_name, config, output_dir=output_dir)


if __name__ == "__main__":
    main()
