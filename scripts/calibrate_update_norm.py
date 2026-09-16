"""Calibrate the ZKP update-norm bound per local step (ppflx/core/update_bound.py).

Runs plain FedAvg with the harness's training settings (SGD, momentum 0.9,
dataset batch size, config learning rate) and one local epoch per round, for
each client count and each initial-model seed, recording every client's update
norm ‖w_local − w_global‖₂ divided by its optimiser steps (its batch count).

    PYTHONPATH=. python scripts/calibrate_update_norm.py --dataset healthcare --clients 2,3,5 --init-seeds 0,1,2,3,42

Prints the observed norms and the per-step bound KAPPA × max(‖Δ‖ / steps),
which goes into PER_STEP_UPDATE_NORM.

Why several client counts and initial models:
  - the per-step norm is not constant: updates grow sublinearly with steps, so
    configurations with fewer steps per epoch have larger per-step norms;
  - the first update's size depends on the initial model. The data partition
    uses --seed (the harness default, 42); ppflx.server.make_strategy seeds the
    initial model with the run's seed, which a deployment may change.

Data is read from --data-path (the repository's dataset/ directory by
default). The MNIST and CIFAR-10 loaders download from the internet when the
files are not found there.
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from ppflx_bench.compare.registry import DATASETS
from ppflx.config import FLConfig
from ppflx.core.update_bound import KAPPA
from ppflx_bench.datasets import get_dataset_loader
from ppflx.models.registry import get_model_for_batch


def flat(net):
    return np.concatenate([v.detach().cpu().numpy().reshape(-1).astype(np.float64) for v in net.state_dict().values()])


def local_epoch(global_net, loader, lr):
    net = copy.deepcopy(global_net)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    net.train()
    for x, y in loader:
        opt.zero_grad()
        loss_fn(net(x), y).backward()
        opt.step()
    return net


def load(dataset, num_clients, seed, data_path):
    # num_workers=0: the default spawns one DataLoader process per core every
    # epoch, which dominates runtime here. Shuffling order comes from the main
    # process RNG, so the measured norms are the same.
    config = FLConfig(
        dataset=dataset,
        num_clients=num_clients,
        seed=seed,
        batch_size=DATASETS[dataset].batch_size,
        num_workers=0,
        data_path=data_path,
    )
    Loader = get_dataset_loader(dataset)
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, _, testloader = Loader().load(config)
    return config, trainloaders, testloader


def run(config, trainloaders, testloader, rounds, init_seed):
    torch.manual_seed(init_seed)  # as ppflx.server.make_strategy does with the run's seed
    global_net = get_model_for_batch(next(iter(testloader)), config.num_classes)

    per_step, norms = [], []
    for _ in range(rounds):
        g = flat(global_net)
        locals_, weights = [], []
        for loader in trainloaders:
            local = flat(local_epoch(global_net, loader, config.learning_rate))
            locals_.append(local)
            weights.append(len(loader))  # what the client reports as num_examples
            norm = float(np.linalg.norm(local - g))
            norms.append(norm)
            per_step.append(norm / len(loader))
        new = sum(w * v for w, v in zip(weights, locals_)) / sum(weights)
        sd, off = global_net.state_dict(), 0
        for k, v in sd.items():
            sd[k] = torch.tensor(new[off : off + v.numel()], dtype=v.dtype).reshape(v.shape)
            off += v.numel()
        global_net.load_state_dict(sd)
    return {
        "clients": config.num_clients,
        "init_seed": init_seed,
        "batch_size": config.batch_size,
        "max_client_batches": max(len(t) for t in trainloaders),
        "max_update_norm": round(max(norms), 6),
        "max_per_step_norm": max(per_step),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clients", default="2,3,5", help="comma-separated client counts")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="data partition seed")
    parser.add_argument("--init-seeds", default="0,1,2,3,42", help="comma-separated initial-model seeds")
    parser.add_argument("--data-path", default="./dataset/", help="local dataset directory")
    args = parser.parse_args()

    runs = []
    for n in [int(c) for c in args.clients.split(",")]:
        config, trainloaders, testloader = load(args.dataset, n, args.seed, args.data_path)
        for init_seed in [int(s) for s in args.init_seeds.split(",")]:
            t0 = time.time()
            res = run(config, trainloaders, testloader, args.rounds, init_seed)
            res["seconds"] = round(time.time() - t0, 1)
            runs.append(res)
            print(json.dumps(res), flush=True)

    per_step = KAPPA * max(r["max_per_step_norm"] for r in runs)
    # Headroom the per-step bound leaves in each client count, against its worst initial model.
    configs = {}
    for r in runs:
        c = configs.setdefault(
            r["clients"], {"clients": r["clients"], "max_client_batches": r["max_client_batches"], "max_update_norm": 0.0}
        )
        c["max_update_norm"] = max(c["max_update_norm"], r["max_update_norm"])
    for c in configs.values():
        c["bound_at_this_config"] = round(per_step * c["max_client_batches"], 6)
        c["headroom"] = round(c["bound_at_this_config"] / c["max_update_norm"], 3)
    print(json.dumps({
        "dataset": args.dataset, "rounds": args.rounds, "seed": args.seed, "init_seeds": args.init_seeds,
        "local_epochs": 1, "kappa": KAPPA, "per_step_bound": float(f"{per_step:.6g}"),
        "configs": list(configs.values()),
    }), flush=True)


if __name__ == "__main__":
    main()
