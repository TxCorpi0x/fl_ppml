"""
CIFAR-10 image dataset loader.

Source: Krizhevsky, 2009  —  https://www.cs.toronto.edu/~kriz/cifar.html
  50,000 training / 10,000 test colour images, 32×32×3, 10 classes.

Why add CIFAR-10?
-----------------
MNIST is grayscale and low-resolution; CIFAR-10 is the canonical RGB benchmark
that better reflects real-world vision models.  Key properties relevant to
privacy research:

  • Gradient inversion attacks reconstruct recognisable images (DLG, iDLG).
  • Non-IID severity has a stronger effect on accuracy than tabular datasets.
  • The CNN model (Net in ppflx/core/model_builder.py) is already compatible.

Usage
-----
    python compare.py --dataset cifar10 --simulation --modes baseline,dp,he_tenseal
    python compare.py --dataset cifar10 --simulation --alpha-sweep

Download
--------
torchvision downloads CIFAR-10 automatically to ``data/cifar10/`` on first run.
Expected download size: ~170 MB.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from ppflx_bench.datasets.base import DatasetSpec, DatasetLoader
from ppflx_bench.datasets.creditcard import _dirichlet_partition
from ppflx_bench.datasets.registry import register_dataset


@register_dataset("cifar10")
@register_dataset("cifar")  # legacy alias
class CIFAR10Loader(DatasetLoader):
    """Federated CIFAR-10 loader.

    Returns (trainloaders, valloaders, testloader) partitioned among
    ``config.num_clients`` clients.  Supports Dirichlet non-IID partitioning
    via ``config.dirichlet_alpha``.

    Model dispatch: the batch shape is [B, 3, 32, 32] (4-D), so
    ``get_model_for_batch`` selects the CNN automatically.
    """

    spec = DatasetSpec(
        name="cifar10",
        input_dim=3 * 32 * 32,  # 3072 — used only for fallback/logging
        num_classes=10,
        is_tabular=False,
        class_names=[
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    )

    # Standard CIFAR-10 preprocessing (no augmentation for FL benchmarks —
    # keeps training deterministic and results reproducible).
    _transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    def load(self, config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        import fcntl
        import os

        data_root = config.data_path or "./data/"

        print("Loading CIFAR-10 dataset…")
        # Serialize downloads across parallel server/client processes.
        lock_path = os.path.join(data_root, ".cifar10_download.lock")
        os.makedirs(data_root, exist_ok=True)
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            trainset = CIFAR10(
                root=data_root, train=True, download=True, transform=self._transform
            )
            testset = CIFAR10(
                root=data_root, train=False, download=True, transform=self._transform
            )

        n = config.num_clients
        val_frac = getattr(config, "val_split", 10) / 100.0
        alpha = getattr(config, "dirichlet_alpha", None)
        seed = getattr(config, "seed", 42)

        if alpha is not None:
            shards = _dirichlet_partition(trainset, n, alpha, seed)
        else:
            shard_size = len(trainset) // n
            lengths = [shard_size] * (n - 1)
            lengths.append(len(trainset) - sum(lengths))
            shards = list(
                random_split(
                    trainset,
                    lengths,
                    generator=torch.Generator().manual_seed(seed),
                )
            )

        trainloaders: List[DataLoader] = []
        valloaders: List[DataLoader] = []

        for shard in shards:
            val_len = max(1, int(len(shard) * val_frac))
            train_len = len(shard) - val_len
            train_split, val_split = random_split(
                shard,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(seed),
            )
            trainloaders.append(
                DataLoader(
                    train_split,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=getattr(config, "num_workers", 0),
                )
            )
            valloaders.append(
                DataLoader(
                    val_split,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=getattr(config, "num_workers", 0),
                )
            )

        testloader = DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, "num_workers", 0),
        )

        return trainloaders, valloaders, testloader
