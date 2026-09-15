"""
FL experiment runner.

Replaces compare_methods_simple.py with a clean, library-based API.

Usage::

    from fl import FLConfig
    from fl.runner import run_mode, run_comparison

    cfg = FLConfig(dataset="creditcard", num_rounds=3)

    # Compare all modes:
    results = run_comparison(
        modes=["baseline", "he_concrete_tfhe", "zkp", "dp"],
        config=cfg,
        output_dir="./results/",
    )

    # Or run a single mode:
    result = run_mode("baseline", cfg, output_dir="./results/")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from fl.config import FLConfig
from fl.core.benchmark import BenchmarkMetrics, init_benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Single-mode runner (in-process Flower simulation)
# ─────────────────────────────────────────────────────────────────────────────


def run_mode(
    mode_name: str,
    config: FLConfig,
    output_dir: str = "./results/",
) -> Dict:
    """
    Run a single FL experiment in-process using Flower simulation.

    Args:
        mode_name:  Privacy mode name (e.g. "baseline", "he_tenseal").
        config:     Experiment configuration.
        output_dir: Directory to save results and benchmark JSON.

    Returns:
        Dict with keys from BenchmarkMetrics.summary() plus "mode" and "duration_s".
    """
    import flwr as fl

    from fl.datasets import get_dataset_loader
    from fl.privacy import get_privacy_mode
    from fl.client import make_client
    from fl.server import make_strategy

    os.makedirs(output_dir, exist_ok=True)
    os.environ["FL_SIMULATION"] = "1"
    os.environ["FL_NUMBER_CLIENTS"] = str(config.num_clients)

    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode_name.upper()}")
    print(
        f"  Dataset: {config.dataset}  Clients: {config.num_clients}  Rounds: {config.num_rounds}"
    )
    print(f"{'=' * 60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    Loader = get_dataset_loader(config.dataset)
    spec = Loader.get_spec()
    config.num_classes = spec.num_classes  # propagate spec → config

    loader = Loader()
    trainloaders, valloaders, testloader = loader.load(config)

    # ── Privacy mode ──────────────────────────────────────────────────────────
    mode_cfg = deepcopy(config)
    mode_cfg.privacy_mode = mode_name

    mode = get_privacy_mode(mode_name)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if "he" in mode_name:
        print(
            "[WARN] In-process simulation: HE modes measure encryption cost but transport "
            "plaintext. Use the distributed runner for results about encrypted transport."
        )
    benchmark = init_benchmark(
        mode=mode_name,
        num_clients=config.num_clients,
        rounds=config.num_rounds,
        transport="simulated",
        zkp_backend=config.zkp_backend if "zkp" in mode_name else None,
    )

    # ── Strategy (server) ─────────────────────────────────────────────────────
    strategy = make_strategy(
        mode_cfg, mode, testloader, benchmark=benchmark, client_batches=max(len(t) for t in trainloaders)
    )

    # ── Client factory ────────────────────────────────────────────────────────
    def client_fn(cid: str):
        return make_client(
            cid=cid,
            trainloaders=trainloaders,
            valloaders=valloaders,
            mode=mode,
            config=mode_cfg,
            benchmark=benchmark,
        )

    # ── Simulation ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    duration = time.perf_counter() - t0

    # ── Save results ──────────────────────────────────────────────────────────
    summary = benchmark.summary()
    summary["duration_s"] = round(duration, 2)

    out_path = os.path.join(output_dir, f"benchmark_{mode_name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved → {out_path}")

    benchmark.print_summary()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Multi-mode comparison runner
# ─────────────────────────────────────────────────────────────────────────────


def run_comparison(
    modes: List[str],
    config: FLConfig,
    output_dir: str = "./results/",
    report_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run multiple FL experiments and produce a comparison report.

    Args:
        modes:       List of privacy mode names to compare.
        config:      Base experiment configuration (shared across modes).
        output_dir:  Directory for per-mode benchmark JSON files.
        report_path: Optional path for the consolidated comparison report.

    Returns:
        Dict mapping mode name → benchmark summary dict.
    """
    if report_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"comparison_report_{ts}.json")

    results: Dict[str, Dict] = {}

    for mode_name in modes:
        try:
            summary = run_mode(mode_name, deepcopy(config), output_dir=output_dir)
            results[mode_name] = summary
        except Exception as exc:
            print(f"\n[ERROR] Mode '{mode_name}' failed: {exc}")
            results[mode_name] = {"error": str(exc)}

    # ── Consolidated report ───────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "dataset": config.dataset,
            "num_clients": config.num_clients,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "modes": modes,
        },
        "results": results,
        "leaderboard": _build_leaderboard(results),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Comparison report saved → {report_path}")

    _print_leaderboard(report["leaderboard"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_metric(summary: Dict, *path) -> float:
    """Navigate nested dict safely; return 0.0 on missing keys."""
    node = summary
    for key in path:
        if not isinstance(node, dict):
            return 0.0
        node = node.get(key, {})
    return float(node) if isinstance(node, (int, float)) else 0.0


def _build_leaderboard(results: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for mode, summary in results.items():
        if "error" in summary:
            continue
        rows.append(
            {
                "mode": mode,
                "accuracy": _get_metric(
                    summary, "model_quality", "test_accuracy", "mean"
                ),
                "precision": _get_metric(
                    summary, "model_quality", "test_precision", "mean"
                ),
                "recall": _get_metric(summary, "model_quality", "test_recall", "mean"),
                "f1": _get_metric(summary, "model_quality", "test_f1", "mean"),
                "auprc": _get_metric(summary, "model_quality", "test_auprc", "mean"),
                "fit_time_s": _get_metric(summary, "timing", "client_fit", "total"),
                "upload_mb": _get_metric(
                    summary, "communication_bytes", "upload", "total"
                )
                / 1e6,
                "duration_s": summary.get("duration_s", 0.0),
            }
        )
    rows.sort(key=lambda r: r["f1"], reverse=True)
    return rows


def _print_leaderboard(rows: List[Dict]) -> None:
    if not rows:
        print("No successful results to display.")
        return
    header = f"{'Mode':<22} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'AUPRC%':>8} {'FitTime':>9} {'Upload MB':>10}"
    print("\n" + "=" * len(header))
    print("COMPARISON LEADERBOARD (sorted by F1)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mode']:<22}"
            f"{r['accuracy']:>7.2f}"
            f"{r['precision']:>7.2f}"
            f"{r['recall']:>7.2f}"
            f"{r['f1']:>7.2f}"
            f"{r['auprc']:>8.2f}"
            f"{r['fit_time_s']:>9.2f}s"
            f"{r['upload_mb']:>9.2f} MB"
        )
    print("=" * len(header) + "\n")
