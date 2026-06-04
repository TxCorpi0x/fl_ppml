#!/usr/bin/env python3
"""
Start the Flower server for distributed (non-simulation) FL.

Examples::

    python scripts/run_server.py --mode baseline --clients 5 --rounds 3
    python scripts/run_server.py --mode he_tenseal --clients 3 --rounds 5
    python scripts/run_server.py --mode zkp --dataset creditcard --rounds 3
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower federated-learning server")
    parser.add_argument("--mode", default="baseline", help="Privacy mode")
    parser.add_argument("--dataset", default="creditcard", help="Dataset name")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--server-address", default="[::]:8080")
    parser.add_argument("--output", default="./results/")
    args = parser.parse_args()

    os.environ["FL_SIMULATION"] = "0"  # real server mode
    os.environ["FL_NUMBER_CLIENTS"] = str(args.clients)

    from fl import FLConfig
    from fl.datasets import get_dataset_loader
    from fl.privacy import get_privacy_mode
    from fl.server import make_strategy
    from fl.core.benchmark import init_benchmark
    import flwr as fl

    config = FLConfig(
        dataset=args.dataset,
        privacy_mode=args.mode,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        device=args.device,
        results_dir=args.output,
        sim_mode=False,
    )

    # Load test data for server-side evaluation
    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    _, _, testloader = Loader().load(config)

    mode = get_privacy_mode(args.mode)
    benchmark = init_benchmark(args.mode, args.clients, args.rounds)
    strategy = make_strategy(config, mode, testloader, benchmark=benchmark)

    print(f"Starting server [{args.mode}] at {args.server_address}")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
