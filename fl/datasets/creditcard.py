"""
Credit card fraud detection dataset loader.

Source: Kaggle "Credit Card Fraud Detection"
  284,807 transactions, 0.17% fraud rate.
  Features: V1-V28 (PCA), Time, Amount → 30 total.

Adding a new dataset is exactly this simple:
  1. Create datasets/<name>.py with @register_dataset
  2. Implement load() → (trainloaders, valloaders, testloader)
  Done — no other files need changing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from fl.datasets.base import DatasetSpec, DatasetLoader
from fl.datasets.registry import register_dataset


@register_dataset("creditcard")
class CreditCardLoader(DatasetLoader):
    spec = DatasetSpec(
        name="creditcard",
        input_dim=30,
        num_classes=2,
        is_tabular=True,
        class_names=["legitimate", "fraud"],
    )

    def load(self, config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        """
        Load, split, and partition the creditcard dataset.

        Args:
            config: FLConfig instance.

        Returns:
            (trainloaders, valloaders, testloader)
        """
        try:
            from datasets import load_creditcard_data, CreditCardDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import datasets.py from the fl_ppml/ directory. "
                "Ensure you ran `pip install -e .` from that directory."
            ) from exc

        print("Loading Credit Card Fraud Detection dataset…")
        X_train, X_test, y_train, y_test = load_creditcard_data(subsample=None)

        trainset = CreditCardDataset(X_train, y_train)
        testset = CreditCardDataset(X_test, y_test)

        return _partition_tabular(trainset, testset, config)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for tabular datasets
# ─────────────────────────────────────────────────────────────────────────────


def _partition_tabular(
    trainset,
    testset,
    config,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Split a trainset into per-client train/val loaders and
    wrap testset in a global testloader.

    When ``config.dirichlet_alpha`` is set, data is distributed according to a
    Dirichlet distribution over class labels, producing non-IID client shards.
    Otherwise, a uniform random split is used (IID).
    """
    n = config.num_clients
    val_frac = config.val_split / 100.0
    alpha = getattr(config, "dirichlet_alpha", None)

    if alpha is not None:
        shards = _dirichlet_partition(trainset, n, alpha, config.seed)
    else:
        shard_size = len(trainset) // n
        lengths = [shard_size] * (n - 1)
        lengths.append(len(trainset) - sum(lengths))
        shards = list(
            random_split(
                trainset,
                lengths,
                generator=torch.Generator().manual_seed(config.seed),
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
            generator=torch.Generator().manual_seed(config.seed),
        )
        trainloaders.append(
            DataLoader(
                train_split,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
        )
        valloaders.append(
            DataLoader(
                val_split,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
        )

    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return trainloaders, valloaders, testloader


def _get_labels(dataset) -> np.ndarray:
    """Extract class labels from a dataset regardless of its concrete type.

    Tries common attributes (targets, y, labels) before falling back to a
    full iteration — which works for any :class:`torch.utils.data.Dataset`.
    """
    for attr in ("targets", "y", "labels"):
        val = getattr(dataset, attr, None)
        if val is not None:
            arr = np.asarray(val)
            if arr.ndim == 1:
                return arr
    # Fallback: iterate the dataset (slow but universal)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def _dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[Subset]:
    """Partition *dataset* into *num_clients* shards using a Dirichlet prior.

    Each client receives samples whose class distribution is drawn from
    Dir(alpha).  Small alpha → extreme non-IID; large alpha → near-IID.

    Parameters
    ----------
    dataset:
        A :class:`torch.utils.data.Dataset`.
    num_clients:
        Number of FL clients / output shards.
    alpha:
        Dirichlet concentration parameter.  Try 0.1, 0.5, 1.0, 10.0.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    List of :class:`torch.utils.data.Subset` objects, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    classes = np.unique(labels)
    num_classes = len(classes)

    # Map each class to its sample indices
    class_indices: List[np.ndarray] = []
    for c in classes:
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        class_indices.append(idx)

    # Draw Dirichlet proportions for each class
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c_idx in range(num_classes):
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to integer counts (floor, then fix rounding)
        counts = (proportions * len(class_indices[c_idx])).astype(int)
        deficit = len(class_indices[c_idx]) - counts.sum()
        # Award the rounding deficit to the client with the highest fractional part
        remainders = (proportions * len(class_indices[c_idx])) - counts
        for _ in range(deficit):
            winner = int(np.argmax(remainders))
            counts[winner] += 1
            remainders[winner] = -1.0

        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            client_indices[client_id].extend(class_indices[c_idx][start:end].tolist())
            start = end

    # Build Subset objects (each shard keeps a reference to the base dataset)
    shards: List[Subset] = []
    for indices in client_indices:
        if not indices:
            # Guard against a client receiving zero samples (rare with tiny α)
            # Give it a single random sample so DataLoader doesn't crash.
            indices = [rng.integers(0, len(dataset))]
        shards.append(Subset(dataset, indices))

    return shards
