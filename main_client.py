#!/usr/bin/env python3
"""
Distributed-FL client entrypoint — thin wrapper around fl.client.make_client().

Called by compare.py (--no-simulation mode) as::

    python main_client.py client --id_client N --dataset X \\
        [--he --he_backend tenseal ...] [--benchmark] --save_results <dir>

All crypto and training logic is in the fl/ package; this file only parses
the legacy CLI flags and wires them to the clean API.
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
    parser = argparse.ArgumentParser(description="FL client")
    subparsers = parser.add_subparsers(dest="subcommand")

    cli = subparsers.add_parser(
        "client", help="Start a Flower federated-learning client"
    )

    # Identity
    cli.add_argument("--id_client", type=int, required=True)

    # Common training args (shared with server via compare.py)
    cli.add_argument("--dataset", type=str, default="creditcard")
    cli.add_argument("--data_path", type=str, default="./data/")
    cli.add_argument("--max_epochs", type=int, default=1)
    cli.add_argument("--batch_size", type=int, default=32)
    cli.add_argument("--lr", type=float, default=0.001)
    cli.add_argument("--device", type=str, default="cpu")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--num_workers", type=int, default=0)
    cli.add_argument("--number_clients", type=int, default=3)
    cli.add_argument("--rounds", type=int, default=3)

    # Privacy-mode flags (legacy API)
    cli.add_argument("--he", action="store_true", default=False)
    cli.add_argument("--he_backend", type=str, default="tenseal")
    cli.add_argument("--path_keys", type=str, default="keys/he_tenseal/secret_key.pkl")
    cli.add_argument(
        "--path_public_key", type=str, default="keys/he_tenseal/public_key.pkl"
    )
    cli.add_argument("--zkp", action="store_true", default=False)
    cli.add_argument("--zkp_backend", type=str, default="gnark")
    cli.add_argument("--zkp_params", type=str, default="keys/zkp/zkp_params.pkl")
    cli.add_argument("--dp", action="store_true", default=False)
    cli.add_argument("--dp_params", type=str, default="keys/dp/dp_params.pkl")
    cli.add_argument(
        "--dp_epsilon",
        type=float,
        default=None,
        help="Override DP privacy budget ε (default: use pre-generated key params).",
    )
    cli.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet concentration for non-IID partitioning (None=IID).",
    )

    # Output / benchmarking
    cli.add_argument("--benchmark", action="store_true", default=False)
    cli.add_argument("--save_results", type=str, default="./results/")
    cli.add_argument("--model_save", type=str, default="./model.pth")

    # Legacy extras (accepted for backward-compat, unused by new pipeline)
    cli.add_argument("--yaml_path", type=str, default="")
    cli.add_argument("--roc_path", type=str, default="")
    cli.add_argument("--matrix_path", type=str, default="")
    cli.add_argument("--path_crypted", type=str, default="")

    return parser


# ── Mode resolution ───────────────────────────────────────────────────────────


def _resolve_mode(args) -> str:
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
            # legacy alias: map old 'concrete' backend to the TFHE mode
            return "he_concrete_tfhe"
        return "he_tenseal"
    if args.zkp:
        return "zkp"
    if args.dp:
        return "dp"
    return "baseline"


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand != "client":
        parser.print_help()
        sys.exit(1)

    os.environ["FL_SIMULATION"] = "0"
    os.environ["FL_NUMBER_CLIENTS"] = str(args.number_clients)

    server_address = os.environ.get("FL_SERVER_ADDRESS", "[::]:8080")
    grpc_max = int(os.environ.get("FL_GRPC_MAX_MESSAGE_LENGTH", str(2_147_483_647)))

    # Mirror the server-side gRPC tuning so the client can also receive large
    # CKKS payloads (creditcard layer 0 = ~451 MB) without EMSGSIZE failures.
    os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "120000")
    os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "60000")
    os.environ.setdefault("GRPC_HTTP2_MAX_FRAME_SIZE", "16777215")

    from fl import FLConfig
    from fl.datasets import get_dataset_loader
    from fl.privacy import get_privacy_mode
    from fl.client import make_client
    from fl.core.benchmark import init_benchmark
    import flwr as fl

    mode_name = _resolve_mode(args)

    save_dir = args.save_results or "./results/"

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
        privacy_mode=mode_name,
        he_tenseal_secret_path=args.path_keys,
        he_tenseal_public_path=args.path_public_key,
        zkp_backend=args.zkp_backend,
        zkp_params_path=args.zkp_params,
        dp_params_path=args.dp_params,
        dp_epsilon=args.dp_epsilon if args.dp_epsilon is not None else 10.0,
        dirichlet_alpha=getattr(args, "dirichlet_alpha", None),
        results_dir=save_dir,
        model_save=args.model_save or "./model.pth",
        benchmark=args.benchmark,
        sim_mode=False,
    )

    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, valloaders, _ = Loader().load(config)

    mode = get_privacy_mode(mode_name)
    benchmark = (
        init_benchmark(mode_name, config.num_clients, config.num_rounds)
        if args.benchmark
        else None
    )

    client = make_client(
        cid=str(args.id_client),
        trainloaders=trainloaders,
        valloaders=valloaders,
        mode=mode,
        config=config,
        benchmark=benchmark,
    )

    print(f"Client {args.id_client} [{mode_name}] connecting to {server_address}")
    fl.client.start_client(
        server_address=server_address,
        grpc_max_message_length=grpc_max,
        client=client.to_client(),
    )

    # Save benchmark after training completes
    if benchmark:
        os.makedirs(save_dir, exist_ok=True)
        bench_path = os.path.join(save_dir, f"client_{args.id_client}_benchmark.json")
        benchmark.save(bench_path)
        benchmark.print_summary()
        print(f"Client benchmark saved → {bench_path}")


if __name__ == "__main__":
    main()
