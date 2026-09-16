"""
Baseline privacy mode — no cryptographic protection.

Parameters are transported as plain numpy arrays.
Aggregation uses standard FedAvg (weighted average).
This is the reference for all benchmarking comparisons.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ppflx.privacy.base import PrivacyMode, _plain_params
from ppflx.privacy.registry import register_mode


@register_mode("baseline")
class BaselineMode(PrivacyMode):
    """Plain federated learning — no HE, no ZKP, no DP."""

    @property
    def name(self) -> str:
        return "baseline"

    def setup_client_context(self, config) -> None:
        return None

    def setup_server_context(self, config) -> None:
        return None

    # All other hooks use PrivacyMode defaults:
    # get_parameters → plain numpy
    # send_parameters → plain numpy
    # receive_parameters → set_parameters(net, params)
    # aggregate_fit_override → None (standard FedAvg)
