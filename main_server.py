#!/usr/bin/env python3
"""
Distributed-FL server entrypoint — thin wrapper around fl.server.make_strategy().

Called by compare.py (--no-simulation mode) as::

    python main_server.py server --dataset X --rounds N --number_clients N \\
        [--he --he_backend tenseal ...] [--benchmark] --model_save <path>

All crypto and strategy logic is in the fl/ package; this file only parses
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
    parser = argparse.ArgumentParser(description="FL server")
    subparsers = parser.add_subparsers(dest="subcommand")

    srv = subparsers.add_parser("server", help="Start Flower federated-learning server")

    # Common training args (shared with clients via compare.py)
    srv.add_argument("--dataset", type=str, default="creditcard")
    srv.add_argument("--data_path", type=str, default="./data/")
    srv.add_argument("--max_epochs", type=int, default=1)
    srv.add_argument("--batch_size", type=int, default=32)
    srv.add_argument("--lr", type=float, default=0.001)
    srv.add_argument("--device", type=str, default="cpu")
    srv.add_argument("--seed", type=int, default=42)
    srv.add_argument("--num_workers", type=int, default=0)

    # Server-only args
    srv.add_argument("--number_clients", type=int, default=2)
    srv.add_argument("--rounds", type=int, default=3)
    srv.add_argument("--frac_fit", type=float, default=1.0)
    srv.add_argument("--frac_eval", type=float, default=0.5)
    srv.add_argument("--min_fit_clients", type=int, default=2)
    srv.add_argument("--min_eval_clients", type=int, default=None)
    srv.add_argument("--min_avail_clients", type=int, default=2)

    # Privacy-mode flags (legacy API)
    srv.add_argument("--he", action="store_true", default=False)
    srv.add_argument("--he_backend", type=str, default="tenseal")
    srv.add_argument("--path_keys", type=str, default="keys/he_tenseal/secret_key.pkl")
    srv.add_argument(
        "--path_public_key", type=str, default="keys/he_tenseal/public_key.pkl"
    )
    srv.add_argument("--zkp", action="store_true", default=False)
    srv.add_argument("--zkp_backend", type=str, default="gnark")
    srv.add_argument("--zkp_params", type=str, default="keys/zkp/zkp_params.pkl")
    srv.add_argument("--dp", action="store_true", default=False)
    srv.add_argument("--dp_params", type=str, default="keys/dp/dp_params.pkl")
    srv.add_argument(
        "--dp_epsilon",
        type=float,
        default=None,
        help="Override DP privacy budget ε (default: use pre-generated key params).",
    )
    srv.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet concentration for non-IID partitioning (None=IID).",
    )

    # Output / benchmarking
    srv.add_argument("--benchmark", action="store_true", default=False)
    srv.add_argument("--save_results", type=str, default="./results/")
    srv.add_argument("--model_save", type=str, default="./model.pth")

    # Blockchain audit ledger
    srv.add_argument(
        "--chain_backend",
        type=str,
        default="mock",
        choices=["mock", "web3", "none"],
        help="Blockchain ledger backend (default: mock)",
    )
    srv.add_argument(
        "--chain_ledger_path",
        type=str,
        default="",
        help="Path for the ledger JSON file (default: <save_results>/ledger.json)",
    )

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
        # Treat legacy "concrete" backend as TFHE mode (consolidated)
        if backend == "concrete":
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

    if args.subcommand != "server":
        parser.print_help()
        sys.exit(1)

    os.environ["FL_SIMULATION"] = "0"
    os.environ["FL_NUMBER_CLIENTS"] = str(args.number_clients)

    # Server address: env var set by compare.py takes priority
    server_address = os.environ.get("FL_SERVER_ADDRESS") or "0.0.0.0:8080"
    grpc_max = int(os.environ.get("FL_GRPC_MAX_MESSAGE_LENGTH", str(2_147_483_647)))

    # Increase gRPC keepalive and socket buffer settings to handle large CKKS
    # payloads (creditcard layer 0 = ~451 MB encrypted) on macOS where
    # recvmsg(2) fails with EMSGSIZE when the kernel socket buffer is exhausted.
    # These env vars are read by the gRPC C core before sockets are created.
    os.environ.setdefault("GRPC_SOCKET_MUTATOR", "")
    os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "120000")  # 2 min
    os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "60000")  # 1 min
    os.environ.setdefault("GRPC_HTTP2_MAX_FRAME_SIZE", "16777215")  # 16 MB frames

    from fl import FLConfig
    from fl.datasets import get_dataset_loader
    from fl.privacy import get_privacy_mode
    from fl.server import make_strategy
    from fl.core.benchmark import init_benchmark
    import flwr as fl

    mode_name = _resolve_mode(args)

    # Determine results dir from model_save path, fallback to --save_results
    model_save = args.model_save or "./model.pth"
    save_dir = args.save_results or os.path.dirname(model_save) or "./results/"

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
        dp_epsilon=args.dp_epsilon if args.dp_epsilon is not None else 10.0,
        dirichlet_alpha=getattr(args, "dirichlet_alpha", None),
        results_dir=save_dir,
        model_save=model_save,
        benchmark=args.benchmark,
        sim_mode=False,
        chain_backend=args.chain_backend if args.chain_backend != "none" else "none",
        chain_ledger_path=(
            args.chain_ledger_path or os.path.join(save_dir, "ledger.json")
        ),
    )

    # Load test data for server-side evaluation
    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    _, _, testloader = Loader().load(config)

    mode = get_privacy_mode(mode_name)
    benchmark = (
        init_benchmark(mode_name, config.num_clients, config.num_rounds)
        if args.benchmark
        else None
    )
    strategy = make_strategy(config, mode, testloader, benchmark=benchmark)

    print(f"Starting server [{mode_name}] at {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        grpc_max_message_length=grpc_max,
    )

    # Save benchmark after server completes
    if benchmark:
        os.makedirs(save_dir, exist_ok=True)
        bench_path = os.path.join(save_dir, "benchmark.json")
        benchmark.save(bench_path)
        benchmark.print_summary()
        print(f"Server benchmark saved → {bench_path}")


if __name__ == "__main__":
    main()
