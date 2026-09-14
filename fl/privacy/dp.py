"""
Differential Privacy (DP-SGD) mode.

DP noise is applied during the local training gradient step in engine.train()
(clipping + Gaussian/Laplace noise on each gradient batch).
Parameters are transported as plain numpy arrays — no additional encryption.
Aggregation uses standard FedAvg.

The per-client DifferentialPrivacyParams object configures:
  epsilon, delta, max_grad_norm, noise_multiplier, mechanism.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fl.privacy.base import PrivacyMode
from fl.privacy.registry import register_mode


@register_mode("dp")
class DifferentialPrivacyMode(PrivacyMode):
    """Federated learning with differentially private gradient updates (DP-SGD)."""

    @property
    def name(self) -> str:
        return "dp"

    def setup_client_context(self, config) -> Any:
        """
        Load DP params from disk.

        Returns:
            DifferentialPrivacyParams instance, or None if params unavailable.
        """
        from fl.core.security import DifferentialPrivacyParams, load_dp_params

        dp_path = config.dp_params_path
        if os.path.exists(dp_path):
            params = load_dp_params(dp_path)
            # Runtime override: if the caller explicitly set dp_epsilon (the
            # sentinel is 10.0 = "not overridden"), recompute noise_multiplier
            # to match the requested privacy budget.
            if hasattr(config, "dp_epsilon") and config.dp_epsilon != 10.0:
                import numpy as _np

                params.epsilon = config.dp_epsilon
                params.noise_multiplier = (
                    _np.sqrt(2 * _np.log(1.25 / params.delta)) / config.dp_epsilon
                )
                print(
                    f"[DP] Config override: ε → {config.dp_epsilon:.4f}"
                    f"  σ → {params.noise_multiplier:.4f}"
                )
            print(f"[DP] Parameters loaded from: {dp_path}")
            print(f"     ε={params.epsilon}  δ={params.delta}")
            print(f"     max_grad_norm={params.max_grad_norm}")
            print(f"     noise_multiplier={params.noise_multiplier:.4f}")
            return params

        # No params file: only an explicitly requested ε is acceptable. The
        # default dp_epsilon=10.0 is the "load from file" sentinel, and using
        # it silently weakened the privacy budget (audit/failmodes.md E-4).
        if getattr(config, "dp_epsilon", 10.0) == 10.0:
            raise FileNotFoundError(
                f"DP params not found: {dp_path}\n"
                "Run: python -m fl.keys generate dp, or pass an explicit --dp_epsilon."
            )
        import numpy as _np

        noise_multiplier = _np.sqrt(2 * _np.log(1.25 / config.dp_delta)) / config.dp_epsilon
        print(
            f"[DP] No params file; using explicit ε={config.dp_epsilon} δ={config.dp_delta} "
            f"σ={noise_multiplier:.4f}"
        )
        return DifferentialPrivacyParams(
            epsilon=config.dp_epsilon,
            delta=config.dp_delta,
            max_grad_norm=config.dp_max_grad_norm,
            noise_multiplier=float(noise_multiplier),
        )

    def setup_server_context(self, config) -> None:
        return None  # server needs no DP context

    # Parameter transport: plain numpy (DP noise was applied to gradients)
    # get_parameters, send_parameters, receive_parameters → use base defaults

    def post_fit_metrics(self, context: Any, benchmark=None) -> Dict:
        """Expose DP stats collected during training."""
        # dp_stats are stashed on the client instance; returned separately.
        # This hook is called after fit() — client passes stats via _dp_stats.
        return {}  # stats are attached by FlowerClient after training
