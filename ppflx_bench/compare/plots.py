"""
Plot generation for comparison results.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ppflx_bench.compare.registry import MODES


def _best_accuracy(metrics: Dict) -> Optional[float]:
    """Return the best accuracy metric available (0-100 scale)."""
    for key in ("test_accuracy", "val_accuracy"):
        v = metrics.get(key)
        if v is None:
            continue
        # model_quality values are dicts with "mean", "max", etc.
        if isinstance(v, dict):
            v = v.get("max") or v.get("mean")
        if v is not None:
            return float(v) * 100 if float(v) <= 1.0 else float(v)
    return None


def create_plots(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Create a 2×3 figure summarising comparison results.

    Parameters
    ----------
    results:
        List of result dicts as returned by ``run_experiment()``, with the
        ``diagnostics`` key already attached by ``add_diagnostics()``.
    output_dir:
        Directory where the PNG is saved.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN]  matplotlib not installed — skipping plot generation.")
        return

    ok_results = [r for r in results if r.get("success") and r.get("benchmark")]
    if not ok_results:
        print("[WARN]  No successful results; skipping plots.")
        return

    labels = [r["mode"] for r in ok_results]
    colors = [
        MODES[r["mode"]].color if r["mode"] in MODES else "#607D8B" for r in ok_results
    ]
    x = np.arange(len(labels))
    bar_width = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Federated Learning — Privacy Mode Comparison", fontsize=14)

    # ── 1. Total time ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    totals = []
    for r in ok_results:
        t = r["benchmark"].get("timing", {})
        totals.append(
            t.get("client_fit", {}).get("total", 0)
            + t.get("server_aggregate", {}).get("total", 0)
        )
    ax.bar(x, totals, bar_width, color=colors)
    ax.set_title("Total Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Seconds")

    # ── 2. Avg client fit ─────────────────────────────────────────────────
    ax = axes[0, 1]
    fit_means = [
        r["benchmark"].get("timing", {}).get("client_fit", {}).get("mean", 0)
        for r in ok_results
    ]
    ax.bar(x, fit_means, bar_width, color=colors)
    ax.set_title("Avg Client Fit Time (s/round)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Seconds")

    # ── 3. Communication (MB) ─────────────────────────────────────────────
    ax = axes[0, 2]
    upload_mb = [
        r["benchmark"].get("communication_bytes", {}).get("upload", {}).get("total", 0)
        / 1e6
        for r in ok_results
    ]
    download_mb = [
        r["benchmark"]
        .get("communication_bytes", {})
        .get("download", {})
        .get("total", 0)
        / 1e6
        for r in ok_results
    ]
    w = bar_width / 2
    ax.bar(x - w / 2, upload_mb, w, label="Upload", color=colors, alpha=0.9)
    ax.bar(
        x + w / 2,
        download_mb,
        w,
        label="Download",
        color=colors,
        alpha=0.55,
        edgecolor="white",
    )
    ax.set_title("Communication (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MB")
    ax.legend(fontsize=8)

    # ── 4. Crypto overhead ───────────────────────────────────────────────
    ax = axes[1, 0]
    crypto_times = []
    for r in ok_results:
        t = r["benchmark"].get("timing", {})
        m = r["mode"]
        if "he" in m:
            v = t.get("encryption", {}).get("total", 0) + t.get("decryption", {}).get(
                "total", 0
            )
        elif "zkp" in m:
            v = t.get("proof_generation", {}).get("total", 0) + t.get(
                "proof_verification", {}
            ).get("total", 0)
        elif m == "dp":
            v = t.get("dp_noise_addition", {}).get("total", 0)
        else:
            v = 0
        crypto_times.append(v)
    ax.bar(x, crypto_times, bar_width, color=colors)
    ax.set_title("Crypto Overhead (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Seconds")

    # ── 5. Best accuracy ─────────────────────────────────────────────────
    ax = axes[1, 1]
    accs = [_best_accuracy(r["benchmark"].get("model_quality", {})) for r in ok_results]
    good = [a if a is not None else 0 for a in accs]
    ax.bar(x, good, bar_width, color=colors)
    ax.set_title("Best Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy %")
    ax.set_ylim(0, 100)

    # ── 6. Convergence (train vs val loss) ──────────────────────────────
    ax = axes[1, 2]
    init_loss = []
    final_loss = []
    for r in ok_results:
        mq = r["benchmark"].get("model_quality", {})
        train_l = mq.get("train_loss", {})
        val_l = mq.get("val_loss", {})
        # model_quality values are dicts with "mean", "min", "max", etc.
        if isinstance(train_l, dict):
            train_l = train_l.get("mean", 0) or 0
        if isinstance(val_l, dict):
            val_l = val_l.get("mean", 0) or 0
        init_loss.append(float(train_l))
        final_loss.append(float(val_l))
    w = bar_width / 2
    ax.bar(
        x - w / 2,
        init_loss,
        w,
        label="Train loss",
        color=colors,
        alpha=0.55,
        edgecolor="white",
    )
    ax.bar(x + w / 2, final_loss, w, label="Val loss", color=colors, alpha=0.9)
    ax.set_title("Loss Convergence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot saved → {out_path}")
