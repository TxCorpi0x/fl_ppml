"""Calibrate the ZKP update-norm bound for a dataset (fl/core/update_bound.py).

Runs plain FedAvg from the seeded initial model with the harness's training
settings (SGD, momentum 0.9, dataset batch size, config learning rate) and one
local epoch per round, recording every client's update norm ‖w_local − w_global‖₂.

    PYTHONPATH=. python scripts/calibrate_update_norm.py --dataset healthcare --clients 3 --rounds 5

Prints the observed norms and the per-epoch bound KAPPA × max, which goes into
PER_EPOCH_UPDATE_NORM. The largest norm usually comes from the first rounds,
while the model is far from a minimum, so a handful of rounds is enough.
"""

import argparse
import copy
import json

import numpy as np
import torch

from fl.compare.registry import DATASETS
from fl.config import FLConfig
from fl.core.update_bound import KAPPA
from fl.datasets import get_dataset_loader
from fl.models.registry import get_model_for_batch


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = FLConfig(dataset=args.dataset, num_clients=args.clients, seed=args.seed,
                      batch_size=DATASETS[args.dataset].batch_size)
    Loader = get_dataset_loader(args.dataset)
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, _, testloader = Loader().load(config)
    torch.manual_seed(args.seed)
    global_net = get_model_for_batch(next(iter(testloader)), config.num_classes)

    rounds = []
    for r in range(1, args.rounds + 1):
        g = flat(global_net)
        locals_, weights, norms = [], [], []
        for loader in trainloaders:
            net = local_epoch(global_net, loader, config.learning_rate)
            locals_.append(flat(net))
            weights.append(len(loader))  # what the client reports as num_examples
            norms.append(float(np.linalg.norm(locals_[-1] - g)))
        new = sum(w * v for w, v in zip(weights, locals_)) / sum(weights)
        off = 0
        sd = global_net.state_dict()
        for k, v in sd.items():
            sd[k] = torch.tensor(new[off : off + v.numel()], dtype=v.dtype).reshape(v.shape)
            off += v.numel()
        global_net.load_state_dict(sd)
        rounds.append(norms)
        print(f"round {r}: update norms {[round(n, 5) for n in norms]}")

    worst = max(max(n) for n in rounds)
    print(json.dumps({
        "dataset": args.dataset, "clients": args.clients, "rounds": args.rounds, "seed": args.seed,
        "batch_size": config.batch_size, "learning_rate": config.learning_rate, "local_epochs": 1,
        "max_update_norm": round(worst, 6), "kappa": KAPPA, "per_epoch_bound": round(KAPPA * worst, 6),
    }))


if __name__ == "__main__":
    main()
