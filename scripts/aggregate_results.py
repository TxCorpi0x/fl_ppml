#!/usr/bin/env python3
"""
Aggregate non-simulation results produced by Docker/compose runs.

Scans per-mode result folders (e.g., ./results/baseline, ./results/he_tenseal, ...),
loads per-client benchmark files (client_<N>_benchmark.json), and produces:
  - comparison_report.json
  - comparison.png (summary plot)

Usage:
  python scripts/aggregate_results.py \
    --root ./results \
    --modes baseline,he_tenseal,zkp,dp \
    --output_dir ./results/docker_compare
"""

import argparse
import json
import os
import math
import csv
from datetime import datetime

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    _PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    np = None
    _PLOTTING_AVAILABLE = False


def _load_mode_benchmark(mode_dir: str):
    """Load and aggregate per-client benchmark files into a single mode summary.

    Returns a dict compatible with the original benchmark summary structure.
    """
    # Prefer aggregated benchmark.json if present
    bench_path = os.path.join(mode_dir, "benchmark.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r") as f:
            return json.load(f)

    # Else gather all client benchmark files
    client_files = sorted(
        [
            os.path.join(mode_dir, f)
            for f in os.listdir(mode_dir)
            if f.startswith("client_") and f.endswith("_benchmark.json")
        ]
    )
    if not client_files:
        return None

    client_benches = []
    for p in client_files:
        try:
            with open(p, "r") as f:
                client_benches.append(json.load(f))
        except Exception:
            continue
    if not client_benches:
        return None

    def agg_stat(key: str):
        stats = [c.get("timing", {}).get(key, {}) for c in client_benches]
        means = [s.get("mean", 0) for s in stats]
        mins = [s.get("min", 0) for s in stats]
        maxs = [s.get("max", 0) for s in stats]
        totals = [s.get("total", 0) for s in stats]
        n = len(stats)
        mean = sum(means) / n if n else 0
        var = sum((m - mean) ** 2 for m in means) / n if n else 0
        std = math.sqrt(var)
        return {
            "mean": float(mean),
            "std": float(std),
            "min": float(min(mins) if mins else 0),
            "max": float(max(maxs) if maxs else 0),
            "total": float(sum(totals)),
        }

    def agg_nested(section: str):
        keys = set()
        for c in client_benches:
            keys.update(c.get(section, {}).keys())
        out = {}
        for k in keys:
            entries = [c.get(section, {}).get(k, {}) for c in client_benches]
            means = [e.get("mean", 0) for e in entries]
            mins = [e.get("min", 0) for e in entries]
            maxs = [e.get("max", 0) for e in entries]
            totals = [e.get("total", 0) for e in entries]
            n = len(entries)
            mean = sum(means) / n if n else 0
            var = sum((m - mean) ** 2 for m in means) / n if n else 0
            std = math.sqrt(var)
            out[k] = {
                "mean": float(mean),
                "std": float(std),
                "min": float(min(mins) if mins else 0),
                "max": float(max(maxs) if maxs else 0),
                "total": float(sum(totals)),
            }
        return out

    sample = client_benches[0]
    agg = {
        "mode": sample.get("mode", os.path.basename(mode_dir)),
        "num_clients": len(client_benches),
        "rounds": max((c.get("rounds", 0) for c in client_benches)),
        "timing": {
            "client_get_params": agg_stat("client_get_params"),
            "client_fit": agg_stat("client_fit"),
            "client_eval": agg_stat("client_eval"),
            "server_aggregate": agg_stat("server_aggregate"),
            "encryption": agg_stat("encryption"),
            "decryption": agg_stat("decryption"),
            "proof_generation": agg_stat("proof_generation"),
            "proof_verification": agg_stat("proof_verification"),
            "dp_noise_addition": agg_stat("dp_noise_addition"),
        },
        "memory_mb": agg_nested("memory_mb"),
        "communication_bytes": agg_nested("communication_bytes"),
        "model_quality": agg_nested("model_quality"),
    }

    return agg


def _crypto_overhead(mode_name: str, b: dict) -> float:
    t = 0.0
    timing = b.get("timing", {})
    if mode_name.startswith("he"):
        # client-side encrypt + decrypt + server-side homomorphic arithmetic
        t += (
            timing.get("encryption", {}).get("total", 0)
            + timing.get("decryption", {}).get("total", 0)
            + timing.get("server_aggregate", {}).get("total", 0)
        )
    elif mode_name.startswith("zkp"):
        t += timing.get("proof_generation", {}).get("total", 0) + timing.get(
            "proof_verification", {}
        ).get("total", 0)
    elif mode_name == "dp":
        t += timing.get("dp_noise_addition", {}).get("total", 0)
    return float(t)


def _best_accuracy(b: dict) -> float:
    acc = 0.0
    mq = b.get("model_quality", {})
    if mq.get("test_accuracy", {}).get("total", 0) > 0:
        acc = mq["test_accuracy"].get("max", 0)
    elif mq.get("val_accuracy", {}).get("total", 0) > 0:
        acc = mq["val_accuracy"].get("max", 0)
    return float(acc)


def _create_plots(results: list, output_dir: str):
    if not _PLOTTING_AVAILABLE:
        print("Plotting dependencies not available; skipping plot generation.")
        return
    valid = [r for r in results if r["success"]]
    if len(valid) < 1:
        print("No successful modes to plot.")
        return

    modes = [r["mode"] for r in valid]
    benches = [r["benchmark"] for r in valid]

    colors = {
        "baseline": "#2ecc71",
        "he_tenseal": "#e74c3c",
        "he_concrete_tfhe": "#9b59b6",
        "zkp": "#3498db",
        "dp": "#f39c12",
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        "Federated Learning (Docker): Baseline vs HE vs ZKP vs DP",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Total time (client_fit + server_aggregate if present)
    ax = axes[0, 0]
    times = [
        b.get("timing", {}).get("client_fit", {}).get("total", 0)
        + b.get("timing", {}).get("server_aggregate", {}).get("total", 0)
        for b in benches
    ]
    bars = ax.bar(modes, times, color=[colors.get(m, "gray") for m in modes], alpha=0.7)
    ax.set_ylabel("Time (s)")
    ax.set_title("Total Training Time")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
        )

    # 2. Avg client fit time
    ax = axes[0, 1]
    fit_mean = [
        b.get("timing", {}).get("client_fit", {}).get("mean", 0) for b in benches
    ]
    bars = ax.bar(
        modes, fit_mean, color=[colors.get(m, "gray") for m in modes], alpha=0.7
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Avg Client Training Time")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fit_mean):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}s",
            ha="center",
            va="bottom",
        )

    # 3. Communication
    ax = axes[1, 0]
    upload = [
        b.get("communication_bytes", {}).get("upload", {}).get("total", 0)
        / (1024 * 1024)
        for b in benches
    ]
    download = [
        b.get("communication_bytes", {}).get("download", {}).get("total", 0)
        / (1024 * 1024)
        for b in benches
    ]
    x = np.arange(len(modes))
    width = 0.35
    ax.bar(x - width / 2, upload, width, label="Upload", alpha=0.7)
    ax.bar(x + width / 2, download, width, label="Download", alpha=0.7)
    ax.set_ylabel("Data (MB)")
    ax.set_title("Communication Overhead")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Crypto overhead
    ax = axes[1, 1]
    crypto = []
    labels = []
    for m, b in zip(modes, benches):
        crypto.append(_crypto_overhead(m, b))
        labels.append(m.upper().replace("_", "\n"))
    bars = ax.bar(
        labels, crypto, color=[colors.get(m, "gray") for m in modes], alpha=0.7
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Cryptographic Overhead")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, crypto):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}s",
                ha="center",
                va="bottom",
            )

    # 5. Accuracy
    ax = axes[0, 2]
    accs = [_best_accuracy(b) for b in benches]
    bars = ax.bar(modes, accs, color=[colors.get(m, "gray") for m in modes], alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best Test/Val Accuracy")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, accs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

    # 6. Loss (initial vs final if available)
    ax = axes[1, 2]
    loss_data = []
    for b in benches:
        mq = b.get("model_quality", {})
        if mq.get("test_loss", {}).get("total", 0) > 0:
            loss_data.append(
                {
                    "initial": mq["test_loss"].get("max", 0),
                    "final": mq["test_loss"].get("min", 0),
                }
            )
        elif mq.get("val_loss", {}).get("total", 0) > 0:
            loss_data.append(
                {
                    "initial": mq["val_loss"].get("max", 0),
                    "final": mq["val_loss"].get("min", 0),
                }
            )
        else:
            loss_data.append(None)

    if any(ld is not None for ld in loss_data):
        initial = [ld["initial"] if ld else 0 for ld in loss_data]
        final = [ld["final"] if ld else 0 for ld in loss_data]

        x = np.arange(len(modes))
        width = 0.35
        ax.bar(
            x - width / 2, initial, width, label="Initial", alpha=0.7, color="#e74c3c"
        )
        ax.bar(x + width / 2, final, width, label="Final", alpha=0.7, color="#2ecc71")
        ax.set_ylabel("Loss")
        ax.set_title("Model Convergence")
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    # 7. Peak client memory
    ax = axes[2, 0]
    mem = [
        b.get("memory_mb", {}).get("client_peak", {}).get("mean", 0) for b in benches
    ]
    bars = ax.bar(modes, mem, color=[colors.get(m, "gray") for m in modes], alpha=0.7)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak Client Memory")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mem):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.0f}MB",
                ha="center",
                va="bottom",
            )

    # 8. Proof generation (total)
    ax = axes[2, 1]
    proofs = [
        b.get("timing", {}).get("proof_generation", {}).get("total", 0) for b in benches
    ]
    bars = ax.bar(
        modes, proofs, color=[colors.get(m, "gray") for m in modes], alpha=0.7
    )
    ax.set_ylabel("Time (s)")
    ax.set_title("Proof Generation Total")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, proofs):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}s",
                ha="center",
                va="bottom",
            )

    # 9. Precision / Recall / F1
    ax = axes[2, 2]
    precs = [
        b.get("model_quality", {}).get("test_precision", {}).get("mean", 0)
        for b in benches
    ]
    recs = [
        b.get("model_quality", {}).get("test_recall", {}).get("mean", 0)
        for b in benches
    ]
    f1s = [
        b.get("model_quality", {}).get("test_f1", {}).get("mean", 0) for b in benches
    ]
    x = np.arange(len(modes))
    width = 0.25
    ax.bar(x - width, precs, width, label="Precision", alpha=0.8, color="#3498db")
    ax.bar(x, recs, width, label="Recall", alpha=0.8, color="#2ecc71")
    ax.bar(x + width, f1s, width, label="F1", alpha=0.8, color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Percent")
    ax.set_title("Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n[PLOT] Plot saved: {plot_path}")
    plt.close()


def _write_csv(results: list, output_dir: str):
    csv_path = os.path.join(output_dir, "comparison.csv")
    headers = [
        "mode",
        "success",
        "total_time_s",
        "avg_fit_s",
        "crypto_s",
        "proof_total_s",
        "mem_mb",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    rows = []
    for r in results:
        b = r.get("benchmark", {})
        timing = b.get("timing", {})
        total = timing.get("client_fit", {}).get("total", 0) + timing.get(
            "server_aggregate", {}
        ).get("total", 0)
        fit = timing.get("client_fit", {}).get("mean", 0)
        crypto = _crypto_overhead(r.get("mode", ""), b)
        proofs = timing.get("proof_generation", {}).get("total", 0)
        mem = b.get("memory_mb", {}).get("client_peak", {}).get("mean", 0)
        acc = _best_accuracy(b)
        prec = b.get("model_quality", {}).get("test_precision", {}).get("mean", 0)
        rec = b.get("model_quality", {}).get("test_recall", {}).get("mean", 0)
        f1 = b.get("model_quality", {}).get("test_f1", {}).get("mean", 0)
        rows.append(
            [
                r.get("mode", ""),
                r.get("success", False),
                f"{total:.3f}",
                f"{fit:.3f}",
                f"{crypto:.3f}",
                f"{proofs:.3f}",
                f"{mem:.1f}",
                f"{acc:.3f}",
                f"{prec:.3f}",
                f"{rec:.3f}",
                f"{f1:.3f}",
            ]
        )

    try:
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Saved CSV summary: {csv_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")


def _print_summary(results: list):
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (Docker Aggregation)")
    print("=" * 100 + "\n")

    print(
        f"{ 'Mode':<16} {'Status':<8} {'Time(s)':<10} {'Fit(s)':<10} {'Crypto(s)':<12} {'Proof(s)':<10} {'Mem(MB)':<10} {'Acc(%)':<8} {'Prec(%)':<8} {'Rec(%)':<8} {'F1(%)':<8}"
    )
    print("-" * 100)

    for r in results:
        status = "[OK]" if r["success"] else "[FAIL]"
        mode = r["mode"].upper()
        if r["success"]:
            b = r["benchmark"]
            timing = b.get("timing", {})
            total = timing.get("client_fit", {}).get("total", 0) + timing.get(
                "server_aggregate", {}
            ).get("total", 0)
            fit = timing.get("client_fit", {}).get("mean", 0)
            crypto = _crypto_overhead(r["mode"], b)
            proofs = timing.get("proof_generation", {}).get("total", 0)
            mem = b.get("memory_mb", {}).get("client_peak", {}).get("mean", 0)
            acc = _best_accuracy(b)
            prec = b.get("model_quality", {}).get("test_precision", {}).get("mean", 0)
            rec = b.get("model_quality", {}).get("test_recall", {}).get("mean", 0)
            f1 = b.get("model_quality", {}).get("test_f1", {}).get("mean", 0)
            print(
                f"{mode:<16} {status:<8} {total:<10.1f} {fit:<10.3f} {crypto:<12.1f} {proofs:<10.1f} {mem:<10.0f} {acc:<8.1f} {prec:<8.1f} {rec:<8.1f} {f1:<8.1f}"
            )
        else:
            print(
                f"{mode:<16} {status:<8} {'-':<10} {'-':<10} {'-':<12} {'-':<10} {'-':<10} {'-':<8}"
            )
    print()


def main():
    ap = argparse.ArgumentParser(description="Aggregate Docker non-simulation results")
    ap.add_argument(
        "--root",
        type=str,
        default="./results",
        help="Results root containing per-mode subfolders",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated modes to include (auto-detect by default)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./results/docker_compare",
        help="Output folder for report and plot",
    )
    args = ap.parse_args()

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        # Auto-detect subfolders containing client_*_benchmark.json
        modes = []
        if not os.path.isdir(args.root):
            print(f"No results root found: {args.root}")
            return
        for name in os.listdir(args.root):
            mode_dir = os.path.join(args.root, name)
            if not os.path.isdir(mode_dir):
                continue
            if any(
                fn.startswith("client_") and fn.endswith("_benchmark.json")
                for fn in os.listdir(mode_dir)
            ):
                modes.append(name)

    # If no mode folders found, attempt to aggregate across timestamped run
    # directories (dataset-level aggregation). Each run should contain a
    # docker_compare/comparison_report.json or comparison_report.json.
    if not modes:

        def _find_run_reports_quick(root_dir: str):
            run_reports = []
            if not os.path.isdir(root_dir):
                return run_reports
            for entry in os.listdir(root_dir):
                sub = os.path.join(root_dir, entry)
                if not os.path.isdir(sub):
                    continue
                c1 = os.path.join(sub, "docker_compare", "comparison_report.json")
                c2 = os.path.join(sub, "comparison_report.json")
                if os.path.exists(c1):
                    run_reports.append(c1)
                elif os.path.exists(c2):
                    run_reports.append(c2)
            return run_reports

        run_reports_quick = _find_run_reports_quick(args.root)
        if run_reports_quick:
            print(
                f"Found {len(run_reports_quick)} run reports under root; aggregating across runs into dataset-level report."
            )
            # reuse the aggregation helpers defined later by calling main again
            # We'll inline aggregation here to avoid code duplication.
            per_mode = {}
            for p in run_reports_quick:
                try:
                    with open(p, "r") as f:
                        rep = json.load(f)
                except Exception:
                    continue
                items = rep if isinstance(rep, list) else rep.get("results", [])
                for item in items:
                    mode = item.get("mode")
                    bench = item.get("benchmark", {}) or {}
                    if mode not in per_mode:
                        per_mode[mode] = []
                    per_mode[mode].append(bench)

            def _agg_stat_list(dicts, key):
                vals = [d.get("timing", {}).get(key, {}) for d in dicts]
                means = [v.get("mean", 0) for v in vals]
                mins = [v.get("min", 0) for v in vals]
                maxs = [v.get("max", 0) for v in vals]
                totals = [v.get("total", 0) for v in vals]
                n = len(vals)
                mean = sum(means) / n if n else 0
                std = (sum((m - mean) ** 2 for m in means) / n) ** 0.5 if n else 0
                return {
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(min(mins) if mins else 0),
                    "max": float(max(maxs) if maxs else 0),
                    "total": float(sum(totals)),
                }

            def _agg_section(dicts, section):
                keys = set()
                for d in dicts:
                    keys.update(d.get(section, {}).keys())
                out = {}
                for k in keys:
                    entries = [d.get(section, {}).get(k, {}) for d in dicts]
                    means = [e.get("mean", 0) for e in entries]
                    mins = [e.get("min", 0) for e in entries]
                    maxs = [e.get("max", 0) for e in entries]
                    totals = [e.get("total", 0) for e in entries]
                    n = len(entries)
                    mean = sum(means) / n if n else 0
                    std = (sum((m - mean) ** 2 for m in means) / n) ** 0.5 if n else 0
                    out[k] = {
                        "mean": float(mean),
                        "std": float(std),
                        "min": float(min(mins) if mins else 0),
                        "max": float(max(maxs) if maxs else 0),
                        "total": float(sum(totals)),
                    }
                return out

            aggregated_results = []
            for mode, benches in per_mode.items():
                sample = benches[0] if benches else {}
                agg = {
                    "mode": mode,
                    "success": True,
                    "benchmark": {
                        "mode": sample.get("mode", mode),
                        "num_clients": (
                            int(
                                sum((b.get("num_clients", 0) for b in benches))
                                / len(benches)
                            )
                            if benches
                            else 0
                        ),
                        "rounds": (
                            int(
                                sum((b.get("rounds", 0) for b in benches))
                                / len(benches)
                            )
                            if benches
                            else 0
                        ),
                        "timing": {
                            "client_get_params": _agg_stat_list(
                                benches, "client_get_params"
                            ),
                            "client_fit": _agg_stat_list(benches, "client_fit"),
                            "client_eval": _agg_stat_list(benches, "client_eval"),
                            "server_aggregate": _agg_stat_list(
                                benches, "server_aggregate"
                            ),
                            "encryption": _agg_stat_list(benches, "encryption"),
                            "decryption": _agg_stat_list(benches, "decryption"),
                            "proof_generation": _agg_stat_list(
                                benches, "proof_generation"
                            ),
                            "proof_verification": _agg_stat_list(
                                benches, "proof_verification"
                            ),
                            "dp_noise_addition": _agg_stat_list(
                                benches, "dp_noise_addition"
                            ),
                        },
                        "memory_mb": _agg_section(benches, "memory_mb"),
                        "communication_bytes": _agg_section(
                            benches, "communication_bytes"
                        ),
                        "model_quality": _agg_section(benches, "model_quality"),
                    },
                }
                aggregated_results.append(agg)

            dataset_report = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "root": args.root,
                    "modes": [r.get("mode") for r in aggregated_results],
                },
                "results": aggregated_results,
                "summary": {
                    "total": len(aggregated_results),
                    "successful": len(aggregated_results),
                    "failed": 0,
                },
            }
            out_path = os.path.join(args.root, "comparison_report.json")
            with open(out_path, "w") as f:
                json.dump(dataset_report, f, indent=2)
            print(f"Wrote dataset-level comparison report: {out_path}")
            # Also write CSV and plots into a docker_compare folder under the dataset
            out_dir = os.path.join(args.root, "docker_compare")
            os.makedirs(out_dir, exist_ok=True)
            # Build results list for helper functions
            results_list = [
                {
                    "mode": r["mode"],
                    "success": True,
                    "benchmark": r["benchmark"],
                    "result_dir": args.root,
                }
                for r in aggregated_results
            ]
            with open(os.path.join(out_dir, "comparison_report.json"), "w") as f:
                json.dump(dataset_report, f, indent=2)
            _create_plots(results_list, out_dir)
            _write_csv(results_list, out_dir)
            return
        else:
            print("No modes found to aggregate.")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # If the provided root contains timestamped run subfolders (dataset run folders),
    # attempt to aggregate across runs and produce a dataset-level comparison_report.json
    # that the notebook expects at results/<dataset>/comparison_report.json.
    # Detect case: root contains many timestamped directories which themselves contain
    # docker_compare/comparison_report.json (or comparison_report.json).
    def _find_run_reports(root_dir: str):
        run_reports = []
        for entry in os.listdir(root_dir):
            sub = os.path.join(root_dir, entry)
            if not os.path.isdir(sub):
                continue
            # check docker_compare first
            c1 = os.path.join(sub, "docker_compare", "comparison_report.json")
            c2 = os.path.join(sub, "comparison_report.json")
            if os.path.exists(c1):
                run_reports.append(c1)
            elif os.path.exists(c2):
                run_reports.append(c2)
        return run_reports

    def _aggregate_across_runs(report_paths: list):
        # Load all run reports and aggregate per-mode across runs (mean of means, sum totals)
        per_mode = {}
        for p in report_paths:
            try:
                with open(p, "r") as f:
                    rep = json.load(f)
            except Exception:
                continue
            items = rep if isinstance(rep, list) else rep.get("results", [])
            for item in items:
                mode = item.get("mode")
                bench = item.get("benchmark", {}) or {}
                if mode not in per_mode:
                    per_mode[mode] = []
                per_mode[mode].append(bench)

        # helper to aggregate stats lists of stat-dicts
        def agg_stat_list(dicts, key):
            # dicts: list of benchmark dicts; key: path under benchmark (e.g., 'timing'->'client_fit')
            vals = [d.get("timing", {}).get(key, {}) for d in dicts]
            means = [v.get("mean", 0) for v in vals]
            mins = [v.get("min", 0) for v in vals]
            maxs = [v.get("max", 0) for v in vals]
            totals = [v.get("total", 0) for v in vals]
            n = len(vals)
            mean = sum(means) / n if n else 0
            std = (sum((m - mean) ** 2 for m in means) / n) ** 0.5 if n else 0
            return {
                "mean": float(mean),
                "std": float(std),
                "min": float(min(mins) if mins else 0),
                "max": float(max(maxs) if maxs else 0),
                "total": float(sum(totals)),
            }

        def agg_section(dicts, section):
            keys = set()
            for d in dicts:
                keys.update(d.get(section, {}).keys())
            out = {}
            for k in keys:
                entries = [d.get(section, {}).get(k, {}) for d in dicts]
                means = [e.get("mean", 0) for e in entries]
                mins = [e.get("min", 0) for e in entries]
                maxs = [e.get("max", 0) for e in entries]
                totals = [e.get("total", 0) for e in entries]
                n = len(entries)
                mean = sum(means) / n if n else 0
                std = (sum((m - mean) ** 2 for m in means) / n) ** 0.5 if n else 0
                out[k] = {
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(min(mins) if mins else 0),
                    "max": float(max(maxs) if maxs else 0),
                    "total": float(sum(totals)),
                }
            return out

        aggregated_results = []
        for mode, benches in per_mode.items():
            sample = benches[0] if benches else {}
            agg = {
                "mode": mode,
                "success": True,
                "benchmark": {
                    "mode": sample.get("mode", mode),
                    "num_clients": (
                        int(
                            sum((b.get("num_clients", 0) for b in benches))
                            / len(benches)
                        )
                        if benches
                        else 0
                    ),
                    "rounds": (
                        int(sum((b.get("rounds", 0) for b in benches)) / len(benches))
                        if benches
                        else 0
                    ),
                    "timing": {
                        "client_get_params": agg_stat_list(
                            benches, "client_get_params"
                        ),
                        "client_fit": agg_stat_list(benches, "client_fit"),
                        "client_eval": agg_stat_list(benches, "client_eval"),
                        "server_aggregate": agg_stat_list(benches, "server_aggregate"),
                        "encryption": agg_stat_list(benches, "encryption"),
                        "decryption": agg_stat_list(benches, "decryption"),
                        "proof_generation": agg_stat_list(benches, "proof_generation"),
                        "proof_verification": agg_stat_list(
                            benches, "proof_verification"
                        ),
                        "dp_noise_addition": agg_stat_list(
                            benches, "dp_noise_addition"
                        ),
                    },
                    "memory_mb": agg_section(benches, "memory_mb"),
                    "communication_bytes": agg_section(benches, "communication_bytes"),
                    "model_quality": agg_section(benches, "model_quality"),
                },
            }
            aggregated_results.append(agg)
        return aggregated_results

    # If root contains run subfolders, aggregate across them into a dataset-level report
    run_reports = _find_run_reports(args.root)
    if run_reports:
        print(
            f"Found {len(run_reports)} run reports under root; aggregating across runs into dataset-level report."
        )
        agg_results = _aggregate_across_runs(run_reports)
        dataset_report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "root": args.root,
                "modes": [r.get("mode") for r in agg_results],
            },
            "results": agg_results,
            "summary": {
                "total": len(agg_results),
                "successful": len(agg_results),
                "failed": 0,
            },
        }
        out_path = os.path.join(args.root, "comparison_report.json")
        with open(out_path, "w") as f:
            json.dump(dataset_report, f, indent=2)
        print(f"Wrote dataset-level comparison report: {out_path}")

    results = []
    for mode in modes:
        mode_dir = os.path.join(args.root, mode)
        if not os.path.isdir(mode_dir):
            print(f"Skipping '{mode}' - directory not found: {mode_dir}")
            continue
        bench = _load_mode_benchmark(mode_dir)
        success = bench is not None and "timing" in bench
        results.append(
            {
                "mode": mode,
                "success": success,
                "benchmark": bench if bench else {},
                "result_dir": mode_dir,
            }
        )

    if not results:
        print("No modes found to aggregate.")
        return

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"root": args.root, "modes": modes},
        "results": results,
        "summary": {
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
        },
    }
    with open(os.path.join(args.output_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {os.path.join(args.output_dir, 'comparison_report.json')}")

    _create_plots(results, args.output_dir)
    _print_summary(results)


if __name__ == "__main__":
    main()
