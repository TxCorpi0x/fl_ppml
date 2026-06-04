#!/usr/bin/env python3
"""
Start a Flower client for distributed (non-simulation) FL.

Examples::

    python scripts/run_client.py --cid 0 --mode baseline
    python scripts/run_client.py --cid 1 --mode he_tenseal --server localhost:8080
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower federated-learning client")
    parser.add_argument("--cid", required=True, help="Client ID (0-indexed)")
    parser.add_argument("--mode", default="baseline", help="Privacy mode")
    parser.add_argument("--dataset", default="creditcard")
    parser.add_argument(
        "--clients",
        type=int,
        default=5,
        help="Total number of clients (for data splitting)",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--server", default="localhost:8080", dest="server_address")
    parser.add_argument("--output", default="./results/")
    args = parser.parse_args()

    os.environ["FL_SIMULATION"] = "0"
    os.environ["FL_NUMBER_CLIENTS"] = str(args.clients)

    from fl import FLConfig
    from fl.datasets import get_dataset_loader
    from fl.privacy import get_privacy_mode
    from fl.client import make_client
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

    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, valloaders, _ = Loader().load(config)

    mode = get_privacy_mode(args.mode)
    client = make_client(
        cid=args.cid,
        trainloaders=trainloaders,
        valloaders=valloaders,
        mode=mode,
        config=config,
    )

    print(f"Client {args.cid} [{args.mode}] connecting to {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
