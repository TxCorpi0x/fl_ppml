"""
Abstract base class for all dataset loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from torch.utils.data import DataLoader


@dataclass
class DatasetSpec:
    """Metadata descriptor for a dataset."""

    name: str
    input_dim: int
    num_classes: int
    is_tabular: bool = True
    class_names: List[str] = None  # e.g. ["no fraud", "fraud"]


class DatasetLoader(ABC):
    """
    Abstract base for all dataset loaders.

    Subclass one per dataset and decorate with::

        @register_dataset("my_dataset")
        class MyLoader(DatasetLoader):
            spec = DatasetSpec(name="my_dataset", input_dim=10, num_classes=2)

            def load(self, config):
                ...

    ``load`` must return a 3-tuple:
        (trainloaders, valloaders, testloader)
    where ``trainloaders[i]`` and ``valloaders[i]`` are the DataLoaders
    for client *i* and ``testloader`` is the global server-side test loader.
    """

    # Subclasses MUST override this class attribute.
    spec: DatasetSpec

    @abstractmethod
    def load(
        self,
        config,  # FLConfig — typed as Any to avoid circular import
    ) -> Tuple[
        List[DataLoader],  # trainloaders  — one per client
        List[DataLoader],  # valloaders    — one per client
        DataLoader,  # testloader    — global server-side evaluation
    ]: ...

    @classmethod
    def get_spec(cls) -> DatasetSpec:
        return cls.spec
