"""The dataset registry and its built-in loaders."""


def test_dataset_registry():
    from ppflx_bench.datasets import list_datasets, get_dataset_loader

    datasets = list_datasets()
    assert "creditcard" in datasets
    assert "healthcare" in datasets
    assert "stock" in datasets
    Loader = get_dataset_loader("creditcard")
    assert Loader is not None


def test_register_custom_dataset():
    from ppflx_bench.datasets.registry import register_dataset, get_dataset_loader, _REGISTRY
    from ppflx_bench.datasets.base import DatasetLoader, DatasetSpec

    @register_dataset("_test_ds")
    class TestDS(DatasetLoader):
        spec = DatasetSpec(name="_test_ds", input_dim=10, num_classes=2)

        def load(self, config):
            return [], [], None

    Cls = get_dataset_loader("_test_ds")
    assert Cls.get_spec().name == "_test_ds"

    del _REGISTRY["_test_ds"]
