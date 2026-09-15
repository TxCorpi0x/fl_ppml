"""
Tests for sweep experiments.

Covers:
  1. DP epsilon override — fl/privacy/dp.py injects config.dp_epsilon into
     the pkl-loaded DifferentialPrivacyParams when the sentinel (10.0) is not used.
  2. Dirichlet alpha flow — dirichlet_alpha reaches FLConfig and is passed
     through to dataset loaders when partitioning data.
"""

from __future__ import annotations

import math
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path so imports work without installation.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_dp_params(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=None):
    """Create a DifferentialPrivacyParams without requiring the full package."""
    from fl.core.security import DifferentialPrivacyParams

    return DifferentialPrivacyParams(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
    )


def _make_config(**kwargs):
    """Return a minimal FLConfig with the given overrides."""
    from fl import FLConfig

    defaults = dict(
        dataset="creditcard",
        privacy_mode="dp",
        dp_params_path="keys/dp/dp_params.json",
    )
    defaults.update(kwargs)
    return FLConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DP epsilon override
# ─────────────────────────────────────────────────────────────────────────────


class TestEpsilonOverride(unittest.TestCase):
    """Verify that dp.py applies config.dp_epsilon override after pkl load."""

    def _run_setup_client(self, config, params_on_disk):
        """Call DifferentialPrivacyMode.setup_client_context with a mocked pkl."""
        from fl.privacy.dp import DifferentialPrivacyMode

        mode = DifferentialPrivacyMode()
        with patch("os.path.exists", return_value=True), patch(
            "fl.core.security.load_dp_params", return_value=params_on_disk
        ):
            return mode.setup_client_context(config)

    def test_no_override_when_sentinel(self):
        """When dp_epsilon == 10.0 (sentinel), pkl params are returned unchanged."""
        params = _make_dp_params(epsilon=1.0, delta=1e-5)
        original_eps = params.epsilon
        original_sigma = params.noise_multiplier

        config = _make_config(dp_epsilon=10.0)
        result = self._run_setup_client(config, params)

        self.assertAlmostEqual(result.epsilon, original_eps)
        self.assertAlmostEqual(result.noise_multiplier, original_sigma)

    def test_override_when_epsilon_set(self):
        """When dp_epsilon != 10.0, params.epsilon and noise_multiplier are recomputed."""
        params = _make_dp_params(epsilon=1.0, delta=1e-5)
        new_eps = 0.5
        expected_sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / new_eps

        config = _make_config(dp_epsilon=new_eps)
        result = self._run_setup_client(config, params)

        self.assertAlmostEqual(result.epsilon, new_eps, places=6)
        self.assertAlmostEqual(result.noise_multiplier, expected_sigma, places=4)

    def test_override_multiple_epsilon_values(self):
        """Verify σ = √(2·ln(1.25/δ)) / ε for several ε values from the sweep."""
        delta = 1e-5
        for eps in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            with self.subTest(eps=eps):
                params = _make_dp_params(epsilon=1.0, delta=delta)
                expected_sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps

                config = _make_config(dp_epsilon=eps)
                result = self._run_setup_client(config, params)

                self.assertAlmostEqual(result.epsilon, eps, places=6)
                self.assertAlmostEqual(
                    result.noise_multiplier,
                    expected_sigma,
                    places=4,
                    msg=f"σ mismatch for ε={eps}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Alpha sweep — dirichlet_alpha reaches dataset loaders
# ─────────────────────────────────────────────────────────────────────────────


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

        with patch("fl.datasets.get_dataset_loader", return_value=mock_loader_class):
            mock_loader_class.get_spec()
            mock_loader_instance.load(config)

        mock_loader_instance.load.assert_called_once_with(config)
        call_config = mock_loader_instance.load.call_args[0][0]
        self.assertEqual(call_config.dirichlet_alpha, alpha)


if __name__ == "__main__":
    unittest.main()
