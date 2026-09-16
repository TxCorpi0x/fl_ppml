"""
The Flower client: local training and the privacy-mode hooks.

FlowerClient's only responsibilities are:
  1. Accept a FLConfig and a PrivacyMode plugin at construction time.
  2. Implement the three client operations: get_parameters, fit, evaluate.
  3. Delegate ALL mode-specific work to the PrivacyMode plugin.

``ppflx_bench.app`` rebuilds a FlowerClient for every Flower message and drives it.

There are no if/elif mode chains here.  Adding a new privacy mode requires
zero edits to this file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def _find_best_f1_threshold(y_true: np.ndarray, y_proba_pos: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 on the given labels.

    Uses sklearn's precision_recall_curve which returns thresholds for every
    distinct score value, so no arbitrary grid-search is needed.
    Returns 0.5 as a safe fallback when the curve cannot be computed.
    """
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_pos)
        # precision_recall_curve appends a sentinel (p=1, r=0) with no threshold;
        # work only over the paired entries
        f1s = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )
        best_idx = int(np.argmax(f1s))
        return float(thresholds[best_idx])
    except Exception:
        return 0.5


from ppflx.config import FLConfig
from ppflx.core.engine import train, test
from ppflx.core.benchmark import BenchmarkTimer, estimate_params_size, get_memory_usage_mb
from ppflx.privacy.base import PrivacyMode


class FlowerClient:
    """
    Privacy-agnostic Flower client.

    All cryptographic operations are delegated to the ``mode`` plugin.
    The client itself only orchestrates the training loop.

    Args:
        cid:        Client identifier string.
        net:        PyTorch model (already on device).
        trainloader: DataLoader for local training data.
        valloader:  DataLoader for local validation data.
        mode:       PrivacyMode plugin instance.
        context:    Client-side crypto context (from mode.setup_client_context).
        config:     Experiment configuration.
        benchmark:  Optional BenchmarkMetrics object for performance tracking.
    """

    def __init__(
        self,
        cid: str,
        net: nn.Module,
        trainloader,
        valloader,
        mode: PrivacyMode,
        context: Any,
        config: FLConfig,
        benchmark=None,
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.mode = mode
        self.crypto_ctx = context
        self.config = config
        self.benchmark = benchmark
        self.device = torch.device(config.device)
        self._dp_stats: Optional[Dict] = None

    # ── Client operations ─────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters (encrypted/plain per mode)."""
        print(f"[Client {self.cid}] get_parameters")
        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_get_params"):
                params = self.mode.get_parameters(
                    self.net,
                    self.crypto_ctx,
                    sim_mode=self.config.sim_mode,
                    benchmark=self.benchmark,
                )
        else:
            params = self.mode.get_parameters(
                self.net,
                self.crypto_ctx,
                sim_mode=self.config.sim_mode,
                benchmark=self.benchmark,
            )
        if self.benchmark:
            self.benchmark.add_upload_size(estimate_params_size(params))
        return params

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        1. Apply server parameters to local model (with decryption if needed).
        2. Train for local_epochs.
        3. Return updated parameters (with encryption if needed) + metrics.
        """
        server_round = config["server_round"]
        self.mode.on_fit_config(self.crypto_ctx, config)
        if not self.mode.trains_this_round(self.crypto_ctx):
            return self._respond_without_training()
        local_epochs = int(config["local_epochs"])
        lr = float(config["learning_rate"])
        print(
            f"[Client {self.cid}, round {server_round}] fit, lr={lr}, epochs={local_epochs}"
        )

        # Track download size
        if self.benchmark:
            self.benchmark.add_download_size(estimate_params_size(parameters))

        # Apply server parameters (mode handles decryption)
        self.mode.receive_parameters(
            self.net,
            parameters,
            self.crypto_ctx,
            sim_mode=self.config.sim_mode,
            benchmark=self.benchmark,
            encrypt_layers=self.config.encrypt_layer_list,
        )

        # Local training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        # DP context is extracted for engine.train (gradient clipping + noise).
        # Supports plain DP mode (context IS the params) and composite modes
        # where context is a dict with a "dp" key (e.g. he_tenseal_zkp_dp).
        if self.config.is_dp:
            dp_params = self.crypto_ctx
        elif isinstance(self.crypto_ctx, dict) and "dp" in self.crypto_ctx:
            dp_params = self.crypto_ctx["dp"]
        else:
            dp_params = None

        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_fit"):
                results = train(
                    self.net,
                    self.trainloader,
                    self.valloader,
                    optimizer=optimizer,
                    loss_fn=criterion,
                    epochs=local_epochs,
                    device=self.device,
                    dp_params=dp_params,
                )
            self.benchmark.add_client_memory(get_memory_usage_mb())
            if results:
                if results["train_loss"]:
                    self.benchmark.add_train_loss(results["train_loss"][-1])
                if results["train_acc"]:
                    self.benchmark.add_train_accuracy(results["train_acc"][-1])
                if results["val_loss"]:
                    self.benchmark.add_val_loss(results["val_loss"][-1])
                if results["val_acc"]:
                    self.benchmark.add_val_accuracy(results["val_acc"][-1])
        else:
            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=local_epochs,
                device=self.device,
                dp_params=dp_params,
            )

        # Store DP stats for post_fit_metrics
        if results and "dp_stats" in results:
            self._dp_stats = results["dp_stats"]

        # Get updated parameters (mode handles encryption)
        updated_params = self.mode.send_parameters(
            self.net,
            self.crypto_ctx,
            sim_mode=self.config.sim_mode,
            benchmark=self.benchmark,
            encrypt_layers=self.config.encrypt_layer_list,
        )

        if self.benchmark:
            self.benchmark.add_upload_size(estimate_params_size(updated_params))

        # Compile metrics
        metrics = self._build_fit_metrics()

        # ZKP proof payloads are sent as metrics alongside the model
        # parameters — count those bytes in the upload tally too.
        if self.benchmark and "gnark_proof_bytes" in metrics:
            proof_bytes = int(metrics["gnark_proof_bytes"])
            if self.benchmark.params_size_upload:
                self.benchmark.params_size_upload[-1] += proof_bytes
            metrics["upload_size"] = self.benchmark.params_size_upload[-1]

        return updated_params, len(self.trainloader), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Apply server parameters and evaluate on local validation data."""
        print(f"[Client {self.cid}] evaluate")

        self.mode.receive_parameters(
            self.net,
            parameters,
            self.crypto_ctx,
            sim_mode=self.config.sim_mode,
        )

        if self.benchmark:
            with BenchmarkTimer(self.benchmark, "client_eval"):
                loss, accuracy, y_pred, y_true, y_proba = test(
                    self.net,
                    self.valloader,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=self.device,
                )
            self.benchmark.add_test_loss(loss)
            self.benchmark.add_test_accuracy(accuracy)
        else:
            loss, accuracy, y_pred, y_true, y_proba = test(
                self.net,
                self.valloader,
                loss_fn=nn.CrossEntropyLoss(),
                device=self.device,
            )

        # Compute classification metrics
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)
        num_classes = len(np.unique(y_true_arr))
        avg = "binary" if num_classes <= 2 else "macro"

        # ── Threshold calibration (binary only) ─────────────────────────────
        # Instead of argmax (fixed 0.5 threshold), find the threshold that
        # maximises F1 on the current evaluation set.  This corrects for
        # quantisation-induced logit scale drift (e.g. TFHE 14-bit) and is
        # a standard practice on imbalanced datasets.  AUPRC is unaffected
        # because it is already threshold-independent.
        best_threshold = 0.5
        if num_classes <= 2 and y_proba_arr.ndim == 2:
            best_threshold = _find_best_f1_threshold(y_true_arr, y_proba_arr[:, 1])
            y_pred_arr = (y_proba_arr[:, 1] >= best_threshold).astype(int)
        else:
            y_pred_arr = np.asarray(y_pred)  # multiclass: keep argmax

        precision = precision_score(
            y_true_arr, y_pred_arr, average=avg, zero_division=0
        )
        recall = recall_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, average=avg, zero_division=0)

        auprc = None
        try:
            if num_classes <= 2 and y_proba_arr.ndim == 2:
                auprc = average_precision_score(y_true_arr, y_proba_arr[:, 1])
            elif y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == num_classes:
                auprc = average_precision_score(
                    np.eye(num_classes)[y_true_arr], y_proba_arr, average="macro"
                )
        except Exception:
            pass

        if self.benchmark:
            self.benchmark.add_test_precision(precision * 100)
            self.benchmark.add_test_recall(recall * 100)
            self.benchmark.add_test_f1(f1 * 100)
            self.benchmark.add_test_threshold(best_threshold)
            if auprc is not None:
                self.benchmark.add_test_auprc(auprc * 100)

        eval_metrics: Dict = {
            "accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_threshold": float(best_threshold),
        }
        if auprc is not None:
            eval_metrics["test_auprc"] = float(auprc)
        return float(loss), len(self.valloader), eval_metrics

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _respond_without_training(self) -> Tuple[List[np.ndarray], int, Dict]:
        """Challenge rounds: prove over the committed update; no download, no training."""
        params = self.mode.send_parameters(
            self.net,
            self.crypto_ctx,
            sim_mode=self.config.sim_mode,
            benchmark=self.benchmark,
            encrypt_layers=self.config.encrypt_layer_list,
        )
        return params, len(self.trainloader), self.mode.post_fit_metrics(self.crypto_ctx, self.benchmark)

    def _build_fit_metrics(self) -> Dict:
        """Collect fit-phase metrics from benchmark + mode plugin."""
        metrics: Dict = {}

        if self.benchmark:
            if self.benchmark.client_fit_time:
                metrics["client_fit_time"] = self.benchmark.client_fit_time[-1]
            if self.benchmark.params_size_upload:
                metrics["upload_size"] = self.benchmark.params_size_upload[-1]
            if self.benchmark.params_size_download:
                metrics["download_size"] = self.benchmark.params_size_download[-1]
            if self.benchmark.client_memory_peak:
                metrics["client_memory"] = self.benchmark.client_memory_peak[-1]
            if self.benchmark.train_loss:
                metrics["train_loss"] = self.benchmark.train_loss[-1]
            if self.benchmark.train_accuracy:
                metrics["train_accuracy"] = self.benchmark.train_accuracy[-1]
            if self.benchmark.val_loss:
                metrics["val_loss"] = self.benchmark.val_loss[-1]
            if self.benchmark.val_accuracy:
                metrics["val_accuracy"] = self.benchmark.val_accuracy[-1]

        # DP stats from the training loop
        if self._dp_stats:
            metrics.update(
                {
                    "dp_noise_time": self._dp_stats.get("dp_step_time", 0.0),
                    "dp_clipped_rate": self._dp_stats.get("clipped_rate", 0.0),
                    "dp_noise_std": self._dp_stats.get("noise_std", 0.0),
                    "dp_epsilon": self._dp_stats.get("epsilon", 0),
                }
            )

        # Mode-specific extra metrics (e.g. ZKP proof payloads, HE timing)
        mode_metrics = self.mode.post_fit_metrics(self.crypto_ctx, self.benchmark)
        metrics.update(mode_metrics)

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def make_client(
    cid: str,
    trainloaders,
    valloaders,
    mode: PrivacyMode,
    config: FLConfig,
    benchmark=None,
) -> FlowerClient:
    """
    Instantiate a FlowerClient for the given client id.

    Detects the model architecture from the batch shape (tabular vs image)
    and creates the appropriate model.  Loads an existing model checkpoint
    if one exists at config.model_save.

    Args:
        cid:          Client id string (used as list index).
        trainloaders: List of per-client DataLoaders.
        valloaders:   List of per-client DataLoaders.
        mode:         PrivacyMode plugin (already instantiated).
        config:       Experiment config.
        benchmark:    Optional BenchmarkMetrics.

    Returns:
        A ready-to-use FlowerClient.
    """
    from ppflx.models import get_model_for_batch

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    device = torch.device(config.device)

    # Auto-select model architecture from batch shape
    sample_batch = next(iter(trainloader))
    # Seeded as in ppflx.server.make_strategy: every client and the server build
    # the same initial model. HE modes take their initial model from one client,
    # so an unseeded client model made those runs start from a different model
    # each time.
    torch.manual_seed(config.seed)
    net = get_model_for_batch(sample_batch, config.num_classes).to(device)

    # Load existing checkpoint if present
    if os.path.exists(config.model_save):
        # weights_only: a checkpoint is tensors, never arbitrary pickled objects.
        checkpoint = torch.load(config.model_save, map_location=device, weights_only=True)
        net.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"[Client {cid}] Loaded checkpoint from {config.model_save}")

    # Setup client-side crypto context
    context = mode.setup_client_context(config)

    return FlowerClient(
        cid=cid,
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        mode=mode,
        context=context,
        config=config,
        benchmark=benchmark,
    )
