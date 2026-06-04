"""
Heart disease (healthcare) dataset loader.

Source: UCI Heart Disease Dataset
  303 samples, 13 features, binary classification (disease / no disease).
"""

from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

from fl.datasets.base import DatasetSpec, DatasetLoader
from fl.datasets.registry import register_dataset
from fl.datasets.creditcard import _partition_tabular


@register_dataset("healthcare")
class HealthcareLoader(DatasetLoader):
    spec = DatasetSpec(
        name="healthcare",
        input_dim=13,
        num_classes=2,
        is_tabular=True,
        class_names=["no disease", "disease"],
    )

    def load(self, config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        try:
            from datasets import load_heart_disease_data, HealthcareDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import datasets.py from the fl_ppml/ directory."
            ) from exc

        print("Loading Healthcare (Heart Disease) dataset…")
        X_train, X_test, y_train, y_test = load_heart_disease_data()

        trainset = HealthcareDataset(X_train, y_train)
        testset = HealthcareDataset(X_test, y_test)

        return _partition_tabular(trainset, testset, config)
