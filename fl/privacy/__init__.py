"""
Privacy mode plugin registry.

Usage::

    from fl.privacy import get_privacy_mode, list_modes

    mode = get_privacy_mode("he_tenseal")
    ctx  = mode.setup_client_context(config)
"""

from fl.privacy.base import PrivacyMode
from fl.privacy.registry import register_mode, get_privacy_mode, list_modes

# Auto-import built-in modes so their @register_mode decorators run.
from fl.privacy import (
    baseline,
    he_tenseal,
    he_concrete_tfhe,
    zkp,
    dp,
    he_zkp,  # he_tenseal_zkp, he_concrete_tfhe_zkp
    he_zkp_dp,  # he_tenseal_zkp_dp, he_concrete_tfhe_zkp_dp
)  # noqa: F401

__all__ = [
    "PrivacyMode",
    "register_mode",
    "get_privacy_mode",
    "list_modes",
]
