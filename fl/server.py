"""
Clean Flower server strategy.

Key differences from the legacy server.py:
  - Zero module-level execution (no code runs on import)
  - No wildcard imports
  - Strategy is constructed via make_strategy() factory function
  - All mode-specific aggregation delegated to the PrivacyMode plugin
  - No if/elif mode chains

Adding a new privacy mode with custom aggregation:
  Override aggregate_fit_override() in the PrivacyMode subclass.
  No changes to this file.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import flwr as fl
import flwr.server.strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from fl.chain import get_chain
from fl.config import FLConfig
from fl.core.engine import test
from fl.core.security import aggregate_custom
from fl.core.benchmark import BenchmarkTimer, get_memory_usage_mb, get_benchmark
from fl.privacy.base import PrivacyMode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────


class FedPrivate(fl.server.strategy.Strategy):
    """
    FedAvg-based strategy that delegates mode-specific aggregation to
    the PrivacyMode plugin.  Strategy internals (sampling, routing, eval)
    are identical to FedAvg.

    Args:
        config:         Experiment configuration.
        mode:           PrivacyMode plugin (instantiated).
        server_context: Server-side crypto context (from mode.setup_server_context).
        central_model:  Global model for server-side evaluation.
        testloader:     DataLoader for server-side evaluation.
        device:         Device for server-side evaluation.
        benchmark:      Optional BenchmarkMetrics.
    """

    def __init__(
        self,
        config: FLConfig,
        mode: PrivacyMode,
        server_context,
        central_model: torch.nn.Module,
        testloader,
        device: torch.device,
        benchmark=None,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.server_context = server_context
        self.central = central_model
        self.testloader = testloader
        self.device = device
        self.benchmark = benchmark
        self.chain = get_chain(config)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return f"FedPrivate(mode={self.mode.name})"

    # ── Flower protocol ───────────────────────────────────────────────────────

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        # For modes like TFHE (server has no private key), let Flower poll a
        # client's get_parameters() so round-1 download is already encrypted.
        if self.mode.use_client_for_initial_params(self.config):
            return None
        params = [val.cpu().numpy() for val in self.central.state_dict().values()]
        return ndarrays_to_parameters(params)

    def num_fit_clients(self, num_available: int) -> Tuple[int, int]:
        return (
            max(int(num_available * self.fraction_fit), self.min_fit_clients),
            self.min_available_clients,
        )

    def num_evaluation_clients(self, num_available: int) -> Tuple[int, int]:
        return (
            max(int(num_available * self.fraction_evaluate), self.min_evaluate_clients),
            self.min_available_clients,
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num
        )
        fit_config = {
            "server_round": server_round,
            "local_epochs": self.config.local_epochs,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
        }
        return [(client, FitIns(parameters, fit_config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        bm = self.benchmark or get_benchmark()

        # Collect any per-client benchmark metrics reported in fit responses
        if bm is not None:
            for _, fit_res in results:
                m = fit_res.metrics or {}
                if "client_fit_time" in m:
                    bm.add_client_fit(m["client_fit_time"])
                if "upload_size" in m:
                    bm.add_upload_size(m["upload_size"])
                if "download_size" in m:
                    bm.add_download_size(m["download_size"])
                if "client_memory" in m:
                    bm.add_client_memory(m["client_memory"])
                if "train_loss" in m:
                    bm.add_train_loss(m["train_loss"])
                if "train_accuracy" in m:
                    bm.add_train_accuracy(m["train_accuracy"])
                if "val_loss" in m:
                    bm.add_val_loss(m["val_loss"])
                if "val_accuracy" in m:
                    bm.add_val_accuracy(m["val_accuracy"])
                if "proof_generation_time" in m:
                    bm.add_proof_generation(m["proof_generation_time"])
                if "proof_verification_time" in m:
                    bm.add_proof_verification(m["proof_verification_time"])
                if "encryption_time" in m:
                    bm.add_encryption(m["encryption_time"])
                if "decryption_time" in m:
                    bm.add_decryption(m["decryption_time"])
                if "dp_noise_time" in m:
                    bm.add_dp_noise(m["dp_noise_time"])

        # ── Mode-specific aggregation (e.g. HE encrypted, ZKP verified) ──────
        override = self.mode.aggregate_fit_override(
            server_round, results, failures, self.server_context, self.config, bm
        )
        if override is not None:
            params_agg, metrics_agg = override
            if params_agg is not None:
                if bm:
                    bm.add_server_memory(get_memory_usage_mb())
                # HE modes return encrypted parameters — the server only holds
                # the public key and cannot decrypt, so skip _update_central.
                if not self.config.is_he:
                    self._update_central(params_agg)
            self._chain_commit(server_round, params_agg, results)
            return params_agg, metrics_agg

        # ── Standard FedAvg (after mode pre-processing) ───────────────────────
        results = self.mode.pre_aggregate(results, self.config)
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        with BenchmarkTimer(bm, "server_aggregate"):
            aggregated = aggregate_custom(weights_results)
            params_agg = ndarrays_to_parameters(aggregated)

        if bm:
            bm.add_server_memory(get_memory_usage_mb())

        self._update_central(params_agg)
        self._chain_commit(server_round, params_agg, results)
        return params_agg, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        sample_size, min_num = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num
        )
        return [(c, EvaluateIns(parameters, {})) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        loss_agg = weighted_loss_avg(
            [(res.num_examples, res.loss) for _, res in results]
        )
        examples = [res.num_examples for _, res in results]
        total_examples = sum(examples)

        def _wavg(key: str) -> float:
            return (
                (
                    sum(
                        res.num_examples * (res.metrics or {}).get(key, 0)
                        for _, res in results
                    )
                    / total_examples
                )
                if total_examples
                else 0.0
            )

        metrics = {"accuracy": _wavg("accuracy")}

        # Collect F1 / precision / recall / AUPRC into the shared benchmark so
        # they appear in benchmark.json (previously these were computed client-
        # side but never sent back through Flower's metrics channel).
        bm = self.benchmark
        if bm:
            for _, res in results:
                m = res.metrics or {}
                if "test_f1" in m:
                    bm.add_test_f1(m["test_f1"] * 100)
                if "test_precision" in m:
                    bm.add_test_precision(m["test_precision"] * 100)
                if "test_recall" in m:
                    bm.add_test_recall(m["test_recall"] * 100)
                if "test_auprc" in m:
                    bm.add_test_auprc(m["test_auprc"] * 100)
                if "test_threshold" in m:
                    bm.add_test_threshold(m["test_threshold"])

        return loss_agg, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side global model evaluation (HE modes skip this)."""
        if self.config.is_he:
            return None  # can't run server-side eval with encrypted model

        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.central.state_dict().keys(), ndarrays)
            }
        )
        self.central.load_state_dict(state_dict, strict=True)

        loss, accuracy, _, _, _ = test(
            self.central,
            self.testloader,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=self.device,
        )
        print(
            f"Server-side eval round {server_round}: loss={loss:.4f}, accuracy={accuracy:.2f}%"
        )

        return loss, {"accuracy": accuracy}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_central(self, parameters: Parameters) -> None:
        """Update the central model state dict from aggregated parameters."""
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.central.state_dict().keys(), ndarrays)
            }
        )
        self.central.load_state_dict(state_dict, strict=True)

        # Persist checkpoint
        results_dir = self.config.results_dir
        if results_dir and self.config.model_save:
            import os

            os.makedirs(results_dir, exist_ok=True)
            torch.save(
                {"model_state_dict": self.central.state_dict()}, self.config.model_save
            )

    def _chain_commit(
        self,
        server_round: int,
        params_agg,
        results,
    ) -> None:
        """
        Compute model and client-update hashes then write both chain events.

        Called after every round regardless of privacy mode:
          - ModelCommit  → SHA-256 of the aggregated model parameters +
                           per-client update hashes.
          - ProofAnchor  → SHA-256 of accepted gnark proof payloads
                           (only emitted when the ZKP mode populated
                           ``self.mode._last_anchor_data``).
        """
        if self.chain is None:
            return

        import hashlib

        # ── Hash the aggregated model parameters ──────────────────────────────
        h = hashlib.sha256()
        if params_agg is not None:
            try:
                arrays = parameters_to_ndarrays(params_agg)
                for arr in arrays:
                    try:
                        h.update(arr.tobytes())
                    except Exception:
                        h.update(bytes(arr))
            except Exception:
                h.update(repr(params_agg).encode())
        model_hash = "0x" + h.hexdigest()

        # ── Hash each accepted client's raw update tensors ────────────────────
        client_hashes = []
        for _, fit_res in results:
            ch = hashlib.sha256()
            try:
                for tensor_bytes in fit_res.parameters.tensors or []:
                    ch.update(tensor_bytes)
            except Exception:
                ch.update(repr(fit_res).encode())
            client_hashes.append("0x" + ch.hexdigest())

        self.chain.commit_model(
            round=server_round,
            model_hash=model_hash,
            client_hashes=client_hashes,
            metadata={"mode": self.mode.name},
        )

        # ── Anchor ZKP proof hashes if the mode produced them ─────────────────
        anchor = getattr(self.mode, "_last_anchor_data", None)
        if anchor and anchor.get("round") == server_round:
            self.chain.anchor_proofs(
                round=server_round,
                proof_hashes=anchor["proof_hashes"],
                client_ids=anchor["client_ids"],
            )
            try:
                del self.mode._last_anchor_data
            except AttributeError:
                pass

        # ── Auto-save ledger after every round ────────────────────────────────
        if self.config.chain_ledger_path:
            try:
                self.chain.save(self.config.chain_ledger_path)
            except Exception as e:
                logger.warning("[Chain] Could not save ledger: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def make_strategy(
    config: FLConfig,
    mode: PrivacyMode,
    testloader,
    benchmark=None,
) -> FedPrivate:
    """
    Create a FedPrivate strategy for the given config and privacy mode.

    This is the canonical entry point for server setup.  It:
      1. Infers the model architecture from the test batch shape.
      2. Sets up the server-side crypto context from the mode plugin.
      3. Constructs and returns the FedPrivate strategy.

    Args:
        config:     Experiment configuration.
        mode:       PrivacyMode plugin (already instantiated).
        testloader: Server-side test DataLoader.
        benchmark:  Optional BenchmarkMetrics.

    Returns:
        A fully configured FedPrivate strategy ready for fl.server.start_server().
    """
    from fl.models import get_model_for_batch

    device = torch.device(config.device)
    sample_batch = next(iter(testloader))
    central = get_model_for_batch(sample_batch, config.num_classes).to(device)
    server_context = mode.setup_server_context(config)
    mode.bind_server_model(server_context, central)

    return FedPrivate(
        config=config,
        mode=mode,
        server_context=server_context,
        central_model=central,
        testloader=testloader,
        device=device,
        benchmark=benchmark,
        fraction_fit=config.frac_fit,
        fraction_evaluate=config.frac_eval,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.effective_min_eval_clients,
        min_available_clients=config.min_avail_clients,
    )
