"""
Stock market (CERN) dataset loader.

Source: Kaggle "Price Volume Data for US Stocks & ETFs"
  Target: StockMove (1 = price up, 0 = price down).
  7 engineered features from OHLC data.
"""

from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

from fl.datasets.base import DatasetSpec, DatasetLoader
from fl.datasets.registry import register_dataset
from fl.datasets.creditcard import _partition_tabular


@register_dataset("stock")
class StockLoader(DatasetLoader):
    spec = DatasetSpec(
        name="stock",
        input_dim=7,  # Volume, SMA, Std_20, Band_1, Band_2, ON_returns_signal, dist_from_mean
        num_classes=2,
        is_tabular=True,
        class_names=["down", "up"],
    )

    def load(self, config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
        try:
            from datasets import load_stock_market_data, StockMarketDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import datasets.py from the fl_ppml/ directory."
            ) from exc

        print("Loading Stock Market dataset…")
        X_train, X_test, y_train, y_test = load_stock_market_data()

        trainset = StockMarketDataset(X_train, y_train)
        testset = StockMarketDataset(X_test, y_test)

        return _partition_tabular(trainset, testset, config)
