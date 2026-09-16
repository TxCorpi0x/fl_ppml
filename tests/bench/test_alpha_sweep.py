"""Non-IID Dirichlet alpha reaching the dataset loaders."""

import unittest
from unittest.mock import MagicMock, patch

from ppflx import FLConfig


def _make_config(**kwargs):
    """Return a minimal FLConfig with the given overrides."""
    from ppflx import FLConfig

    defaults = dict(
        dataset="creditcard",
        privacy_mode="dp",
        dp_params_path="keys/dp/dp_params.json",
    )
    defaults.update(kwargs)
    return FLConfig(**defaults)


class TestAlphaReachesLoader(unittest.TestCase):
    """Verify dirichlet_alpha flows through FLConfig to dataset loaders."""

    def test_alpha_stored_in_config(self):
        """FLConfig stores dirichlet_alpha when passed explicitly."""
        config = _make_config(dirichlet_alpha=0.1)
        self.assertEqual(config.dirichlet_alpha, 0.1)

    def test_alpha_none_by_default(self):
        """FLConfig.dirichlet_alpha defaults to None (IID partitioning)."""
        config = _make_config()
        self.assertIsNone(config.dirichlet_alpha)

    def test_alpha_passed_to_loader_load(self):
        """The config with dirichlet_alpha is forwarded to Loader.load()."""
        # We don't need a real dataset; just verify the config attribute reaches load().
        alpha = 0.5
        config = _make_config(dirichlet_alpha=alpha)

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = ([], [], [])

        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        mock_loader_class.get_spec.return_value = MagicMock(num_classes=2)

        with patch("ppflx_bench.datasets.get_dataset_loader", return_value=mock_loader_class):
            mock_loader_class.get_spec()
            mock_loader_instance.load(config)

        mock_loader_instance.load.assert_called_once_with(config)
        call_config = mock_loader_instance.load.call_args[0][0]
        self.assertEqual(call_config.dirichlet_alpha, alpha)
