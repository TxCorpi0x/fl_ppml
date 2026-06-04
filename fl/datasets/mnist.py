"""
MNIST handwritten digit dataset loader.

Source: Yann LeCun, http://yann.lecun.com/exdb/mnist/
  60,000 training / 10,000 test grayscale images, 28×28, 10 classes.

Why add image datasets?
------------------------
Gradient inversion attacks (DLG, iDLG, InvertGrad) are *far* more dangerous
on image data than on tabular data: the reconstructed images are visually
recognisable, making the privacy threat concrete and intuitive.  Tabular-only
benchmarks understate the real-world risk — a common reviewer criticism.

MNIST is the canonical lightweight image baseline:
  • Fast to train (CPU-friendly, small model).
  • Well-understood accuracy ceiling (~99% centralised).
  • Gradient inversion reconstructions are clearly legible.

Usage
-----
    python compare.py --dataset mnist --simulation --modes baseline,dp,he_tenseal

Download
--------
torchvision downloads MNIST automatically to ``data/mnist/`` on first run.
No manual download is required.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from fl.datasets.base import DatasetSpec, DatasetLoader
from fl.datasets.creditcard import _dirichlet_partition
from fl.datasets.registry import register_dataset


@register_dataset("mnist")
class MNISTLoader(DatasetLoader):
    """Federated MNIST loader.

    Returns (trainloaders, valloaders, testloader) partitioned among
    ``config.num_clients`` clients.  Supports Dirichlet non-IID partitioning
    via ``config.dirichlet_alpha``.
    """

    spec = DatasetSpec(
        name="mnist",
        input_dim=784,  # 1 × 28 × 28 — used only for tabular fallback
        num_classes=10,
        is_tabular=False,
        class_names=[str(i) for i in range(10)],
    )

    # Normalise to zero mean / unit std (standard MNIST preprocessing)
    _transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]
    )

    def load(self, config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        import fcntl
        import os

        data_root = config.data_path or "./data/"

        print("Loading MNIST dataset…")
        # Serialize downloads across parallel server/client processes to avoid
        # concurrent gz-extraction races that leave 0-byte files.
        lock_path = os.path.join(data_root, ".mnist_download.lock")
        os.makedirs(data_root, exist_ok=True)
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            trainset = MNIST(
                root=data_root, train=True, download=True, transform=self._transform
            )
            testset = MNIST(
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
