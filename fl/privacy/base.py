"""
Privacy mode plugin ABC.

Every privacy mode (baseline, he_tenseal, …) subclasses PrivacyMode
and overrides only the hooks it needs.  The FlowerClient and FedCustom
strategy delegate all mode-specific work to the plugin — they contain
no if/elif mode chains.

Adding a new privacy mode:
  1. Create fl/privacy/<new_mode>.py
  2. Subclass PrivacyMode
  3. Decorate the class with @register_mode("<name>")
  Done — no other files need editing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PrivacyMode(ABC):
    """
    Abstract base for all privacy modes.

    Terminology
    -----------
    context_client : opaque per-client crypto context returned by
                     setup_client_context().  May be None (baseline/dp),
                     a TenSEAL/Concrete object (HE modes), or a dict (ZKP).
    context_server : opaque per-server crypto context returned by
                     setup_server_context(). May be None.
    sim_mode       : when True the FL run is in-process (Flower simulation);
                     crypto is measured but plain numpy is transported.
    """

    # ── Identification ───────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry identifier (e.g. "he_tenseal")."""
        ...

    # ── Setup ────────────────────────────────────────────────────────────────

    @abstractmethod
    def setup_client_context(self, config) -> Any:
        """
        Create the per-client crypto context.
        Called once per client before FL begins.

        Returns:
            Crypto context (or None for modes without client-side state).
        """
        ...

    @abstractmethod
    def setup_server_context(self, config) -> Any:
        """
        Create the server-side crypto context.
        Called once on the server before FL begins.

        Returns:
            Crypto context (or None).
        """
        ...

    # ── Parameter flow — client send ─────────────────────────────────────────

    def get_parameters(
        self,
        net,
        context: Any,
        *,
        sim_mode: bool,
        benchmark=None,
    ) -> List[np.ndarray]:
        """
        Return model parameters for upload to the server.

        Default: plain numpy arrays (no encryption).
        Override to encrypt, sign, or compress parameters.
        """
        return _plain_params(net)

    def send_parameters(
        self,
        net,
        context: Any,
        *,
        sim_mode: bool,
        benchmark=None,
        encrypt_layers=None,
    ) -> List[np.ndarray]:
        """
        Return updated parameters after local training (upload to server).

        Default: plain numpy arrays.
        """
        return _plain_params(net)

    # ── Parameter flow — client receive ─────────────────────────────────────

    def receive_parameters(
        self,
        net,
        params: List[np.ndarray],
        context: Any,
        *,
        sim_mode: bool,
        benchmark=None,
        encrypt_layers=None,
    ) -> None:
        """
        Apply received server parameters to the local model.

        Default: plain set_parameters (no decryption).
        """
        from fl.core.params import set_parameters

        set_parameters(net, params, None, None)

    # ── Post-fit metrics hook ────────────────────────────────────────────────

    def post_fit_metrics(
        self,
        context: Any,
        benchmark=None,
    ) -> Dict:
        """
        Extra key/value pairs to return in the fit() metrics dict.

        For ZKP: include proof payloads.
        For DP:  include noise/clipping stats.
        Default: empty dict.
        """
        return {}

    # ── Server aggregation hooks ─────────────────────────────────────────────

    def use_client_for_initial_params(self, config) -> bool:
        """
        If True, server.initialize_parameters() returns None so Flower polls
        a real client's get_parameters() for the starting weights.  This
        ensures round-1 download uses whatever encoding the mode applies
        (e.g. TFHE encrypted) rather than a raw plaintext blob.

        Default: False (server pushes its own plain initial weights).
        """
        return False

    def bind_server_model(self, server_context: Any, model) -> None:
        """
        Give the server context the server's own global model.

        Called once by make_strategy. Modes that must validate uploads against
        a server-owned schema read it here instead of trusting client metadata.
        Default: no-op.
        """
        return None

    def on_fit_config(self, context: Any, fit_config: Dict) -> None:
        """
        Receive the server's fit config (including ``server_round``) at the
        start of each client fit(), before any parameters are sent.
        Default: no-op.
        """
        return None

    def aggregate_fit_override(
        self,
        server_round: int,
        results,
        failures,
        server_context: Any,
        config,
        benchmark=None,
    ) -> Optional[Tuple]:
        """
        Custom server-side aggregation.

        Return a (parameters, metrics_dict) tuple to short-circuit the
        standard FedAvg aggregation, or return None to fall through to
        standard FedAvg (after pre_aggregate()).
        """
        return None  # use standard FedAvg

    def pre_aggregate(
        self,
        results: List,
        config,
    ) -> List:
        """
        Pre-process client results before FedAvg.

        For he_concrete_tfhe: decompress uint8 CTE2 envelopes → float32 arrays.
        Default: pass through unchanged.
        """
        return results

    # ── Utility ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<PrivacyMode: {self.name}>"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers used by multiple modes
# ─────────────────────────────────────────────────────────────────────────────


def _plain_params(net) -> List[np.ndarray]:
    """Extract model parameters as a list of numpy float32 arrays."""
    return [
        val.detach().cpu().numpy().astype(np.float32, copy=False)
        for val in net.state_dict().values()
    ]
