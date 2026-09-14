"""
Performance benchmarking utilities for comparing FL with HE, ZKP, and baseline.
"""

import time
import psutil
import os
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    # Timing metrics (seconds)
    client_get_params_time: List[float] = field(default_factory=list)
    client_fit_time: List[float] = field(default_factory=list)
    client_eval_time: List[float] = field(default_factory=list)
    server_aggregate_time: List[float] = field(default_factory=list)

    # Memory metrics (MB)
    client_memory_peak: List[float] = field(default_factory=list)
    server_memory_peak: List[float] = field(default_factory=list)

    # Communication metrics (bytes)
    params_size_upload: List[int] = field(default_factory=list)
    params_size_download: List[int] = field(default_factory=list)

    # Cryptographic operation metrics
    encryption_time: List[float] = field(default_factory=list)
    decryption_time: List[float] = field(default_factory=list)
    proof_generation_time: List[float] = field(default_factory=list)
    proof_verification_time: List[float] = field(default_factory=list)
    dp_noise_addition_time: List[float] = field(default_factory=list)

    # Model quality metrics
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    test_precision: List[float] = field(default_factory=list)
    test_recall: List[float] = field(default_factory=list)
    test_f1: List[float] = field(default_factory=list)
    test_auprc: List[float] = field(default_factory=list)
    test_threshold: List[float] = field(
        default_factory=list
    )  # F1-optimal decision threshold

    # Per-round outcome reports recorded by the server strategy
    round_outcomes: List[Dict] = field(default_factory=list)

    # Metadata
    mode: str = "baseline"  # baseline, he, zkp
    num_clients: int = 0
    rounds: int = 0

    def add_round_outcome(self, report: Dict):
        """Record how a round ended (aggregated, rejected clients, aborted)."""
        self.round_outcomes.append(dict(report))

    def add_client_get_params(self, duration: float):
        """Add client parameter retrieval time."""
        self.client_get_params_time.append(duration)

    def add_client_fit(self, duration: float):
        """Add client training time."""
        self.client_fit_time.append(duration)

    def add_client_eval(self, duration: float):
        """Add client evaluation time."""
        self.client_eval_time.append(duration)

    def add_server_aggregate(self, duration: float):
        """Add server aggregation time."""
        self.server_aggregate_time.append(duration)

    def add_client_memory(self, memory_mb: float):
        """Add client memory usage."""
        self.client_memory_peak.append(memory_mb)

    def add_server_memory(self, memory_mb: float):
        """Add server memory usage."""
        self.server_memory_peak.append(memory_mb)

    def add_upload_size(self, size_bytes: int):
        """Add parameter upload size."""
        self.params_size_upload.append(size_bytes)

    def add_download_size(self, size_bytes: int):
        """Add parameter download size."""
        self.params_size_download.append(size_bytes)

    def add_encryption(self, duration: float):
        """Add encryption time."""
        self.encryption_time.append(duration)

    def add_decryption(self, duration: float):
        """Add decryption time."""
        self.decryption_time.append(duration)

    def add_proof_generation(self, duration: float):
        """Add proof generation time."""
        self.proof_generation_time.append(duration)

    def add_proof_verification(self, duration: float):
        """Add proof verification time."""
        self.proof_verification_time.append(duration)

    def add_dp_noise(self, duration: float):
        """Add differential privacy noise addition time."""
        self.dp_noise_addition_time.append(duration)

    def add_train_loss(self, loss: float):
        """Add training loss."""
        self.train_loss.append(loss)

    def add_train_accuracy(self, accuracy: float):
        """Add training accuracy."""
        self.train_accuracy.append(accuracy)

    def add_val_loss(self, loss: float):
        """Add validation loss."""
        self.val_loss.append(loss)

    def add_val_accuracy(self, accuracy: float):
        """Add validation accuracy."""
        self.val_accuracy.append(accuracy)

    def add_test_loss(self, loss: float):
        """Add test loss."""
        self.test_loss.append(loss)

    def add_test_accuracy(self, accuracy: float):
        """Add test accuracy."""
        self.test_accuracy.append(accuracy)

    def add_test_precision(self, precision: float):
        """Add test precision."""
        self.test_precision.append(precision)

    def add_test_recall(self, recall: float):
        """Add test recall."""
        self.test_recall.append(recall)

    def add_test_f1(self, f1: float):
        """Add test F1 score."""
        self.test_f1.append(f1)

    def add_test_auprc(self, auprc: float):
        """Add test AUPRC."""
        self.test_auprc.append(auprc)

    def add_test_threshold(self, threshold: float):
        """Add F1-optimal decision threshold used for this round's evaluation."""
        self.test_threshold.append(threshold)

    def summary(self) -> Dict:
        """Get summary statistics."""

        def stats(values: List[float]) -> Dict:
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "total": 0}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "total": float(np.sum(values)),
            }

        return {
            "mode": self.mode,
            "num_clients": self.num_clients,
            "rounds": self.rounds,
            "timing": {
                "client_get_params": stats(self.client_get_params_time),
                "client_fit": stats(self.client_fit_time),
                "client_eval": stats(self.client_eval_time),
                "server_aggregate": stats(self.server_aggregate_time),
                "encryption": stats(self.encryption_time),
                "decryption": stats(self.decryption_time),
                "proof_generation": stats(self.proof_generation_time),
                "proof_verification": stats(self.proof_verification_time),
                "dp_noise_addition": stats(self.dp_noise_addition_time),
            },
            "memory_mb": {
                "client_peak": stats(self.client_memory_peak),
                "server_peak": stats(self.server_memory_peak),
            },
            "communication_bytes": {
                "upload": stats(self.params_size_upload),
                "download": stats(self.params_size_download),
            },
            "model_quality": {
                "train_loss": stats(self.train_loss),
                "train_accuracy": stats(self.train_accuracy),
                "val_loss": stats(self.val_loss),
                "val_accuracy": stats(self.val_accuracy),
                "test_loss": stats(self.test_loss),
                "test_accuracy": stats(self.test_accuracy),
                "test_precision": stats(self.test_precision),
                "test_recall": stats(self.test_recall),
                "test_f1": stats(self.test_f1),
                "test_auprc": stats(self.test_auprc),
                "test_threshold": stats(self.test_threshold),
            },
            "round_outcomes": list(self.round_outcomes),
        }

    def save(self, filepath: str):
        """Save benchmark results to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Benchmark results saved to: {filepath}")

    def print_summary(self):
        """Print benchmark summary to console."""
        summary = self.summary()

        print("\n" + "=" * 60)
        print(f"BENCHMARK SUMMARY - Mode: {summary['mode'].upper()}")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Clients: {summary['num_clients']}")
        print(f"  Rounds: {summary['rounds']}")

        print(f"\nTiming (seconds):")
        for key, stats in summary["timing"].items():
            if stats["total"] > 0:
                print(f"  {key}:")
                print(f"    Mean: {stats['mean']:.4f}s  Std: {stats['std']:.4f}s")
                print(
                    f"    Min: {stats['min']:.4f}s  Max: {stats['max']:.4f}s  Total: {stats['total']:.4f}s"
                )

        print(f"\nMemory (MB):")
        for key, stats in summary["memory_mb"].items():
            if stats["total"] > 0:
                print(f"  {key}: Mean: {stats['mean']:.2f}  Max: {stats['max']:.2f}")

        print(f"\nCommunication (MB):")
        for key, stats in summary["communication_bytes"].items():
            if stats["total"] > 0:
                mb_stats = {k: v / (1024 * 1024) for k, v in stats.items()}
                print(
                    f"  {key}: Mean: {mb_stats['mean']:.2f}  Total: {mb_stats['total']:.2f}"
                )

        print(f"\nModel Quality:")
        quality = summary["model_quality"]
        if quality["train_loss"]["total"] > 0:
            print(f"  Training:")
            print(
                f"    Loss: Mean: {quality['train_loss']['mean']:.4f}  Final: {quality['train_loss']['min']:.4f}"
            )
            print(
                f"    Accuracy: Mean: {quality['train_accuracy']['mean']:.2f}%  Best: {quality['train_accuracy']['max']:.2f}%"
            )

        if quality["val_loss"]["total"] > 0:
            print(f"  Validation:")
            print(
                f"    Loss: Mean: {quality['val_loss']['mean']:.4f}  Final: {quality['val_loss']['min']:.4f}"
            )
            print(
                f"    Accuracy: Mean: {quality['val_accuracy']['mean']:.2f}%  Best: {quality['val_accuracy']['max']:.2f}%"
            )

        if quality["test_loss"]["total"] > 0:
            print(f"  Test:")
            print(
                f"    Loss: Mean: {quality['test_loss']['mean']:.4f}  Final: {quality['test_loss']['min']:.4f}"
            )
            print(
                f"    Accuracy: Mean: {quality['test_accuracy']['mean']:.2f}%  Best: {quality['test_accuracy']['max']:.2f}%"
            )
            if quality["test_precision"]["total"] > 0:
                print(
                    f"    Precision: Mean: {quality['test_precision']['mean']:.2f}%  Best: {quality['test_precision']['max']:.2f}%"
                )
            if quality["test_recall"]["total"] > 0:
                print(
                    f"    Recall: Mean: {quality['test_recall']['mean']:.2f}%  Best: {quality['test_recall']['max']:.2f}%"
                )
            if quality["test_f1"]["total"] > 0:
                print(
                    f"    F1: Mean: {quality['test_f1']['mean']:.2f}%  Best: {quality['test_f1']['max']:.2f}%"
                )
            if quality["test_auprc"]["total"] > 0:
                print(
                    f"    AUPRC: Mean: {quality['test_auprc']['mean']:.2f}%  Best: {quality['test_auprc']['max']:.2f}%"
                )

        print("=" * 60 + "\n")


class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        metrics: Optional[BenchmarkMetrics] = None,
        metric_type: Optional[str] = None,
    ):
        """
        Initialize timer.

        Args:
            metrics: BenchmarkMetrics object to update
            metric_type: Type of metric to record (e.g., 'client_fit')
        """
        self.metrics = metrics
        self.metric_type = metric_type
        self.start_time = None
        self.duration = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record metric."""
        self.duration = time.time() - self.start_time

        if self.metrics and self.metric_type:
            if self.metric_type == "client_fit":
                self.metrics.add_client_fit(self.duration)
            elif self.metric_type == "client_eval":
                self.metrics.add_client_eval(self.duration)
            elif self.metric_type == "client_get_params":
                self.metrics.add_client_get_params(self.duration)
            elif self.metric_type == "server_aggregate":
                self.metrics.add_server_aggregate(self.duration)
            elif self.metric_type == "encryption":
                self.metrics.add_encryption(self.duration)
            elif self.metric_type == "decryption":
                self.metrics.add_decryption(self.duration)
            elif self.metric_type == "proof_generation":
                self.metrics.add_proof_generation(self.duration)
            elif self.metric_type == "proof_verification":
                self.metrics.add_proof_verification(self.duration)
            elif self.metric_type == "dp_noise_addition":
                self.metrics.add_dp_noise(self.duration)


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_object_size(obj) -> int:
    """
    Get size of an object in bytes.

    Args:
        obj: Object to measure

    Returns:
        Size in bytes
    """
    return sys.getsizeof(obj)


def estimate_params_size(parameters) -> int:
    """
    Estimate total size of parameters in bytes.

    Args:
        parameters: List of numpy arrays or similar

    Returns:
        Estimated size in bytes
    """
    total_size = 0

    if isinstance(parameters, list):
        for param in parameters:
            if hasattr(param, "nbytes"):
                total_size += param.nbytes
            else:
                total_size += get_object_size(param)
    else:
        total_size = get_object_size(parameters)

    return total_size


# Global benchmark instance
_global_benchmark: Optional[BenchmarkMetrics] = None


def init_benchmark(mode: str, num_clients: int, rounds: int) -> BenchmarkMetrics:
    """
    Initialize global benchmark metrics.

    Args:
        mode: "baseline", "he", or "zkp"
        num_clients: Number of clients
        rounds: Number of rounds

    Returns:
        BenchmarkMetrics instance
    """
    global _global_benchmark
    _global_benchmark = BenchmarkMetrics(
        mode=mode, num_clients=num_clients, rounds=rounds
    )
    return _global_benchmark


def get_benchmark() -> Optional[BenchmarkMetrics]:
    """Get the global benchmark instance."""
    return _global_benchmark


def save_benchmark(filepath: str):
    """Save global benchmark to file."""
    if _global_benchmark:
        _global_benchmark.save(filepath)


def print_benchmark():
    """Print global benchmark summary."""
    if _global_benchmark:
        _global_benchmark.print_summary()
