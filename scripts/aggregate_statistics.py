#!/usr/bin/env python3
"""
Aggregate statistics from repeated FL comparison runs.

Scans all timestamped subdirectories under --root for comparison_report.json
files, groups results by mode, and computes mean ± std statistics across runs.

Usage:
  python scripts/aggregate_statistics.py --root results/healthcare/
  python scripts/aggregate_statistics.py --root results/creditcard/ --min-runs 3
  python scripts/aggregate_statistics.py --root results/healthcare/ --no-plot
  python scripts/aggregate_statistics.py --root results/healthcare/ \\
    --modes baseline,dp,he_tenseal

The --root directory is scanned for direct children matching the timestamp
pattern YYYYMMDD_HHMMSS (14-character strings). Run subdirs nested under
alpha_<value>/ or similar are NOT scanned to avoid mixing sweep results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Timestamp pattern for run directories: YYYYMMDD_HHMMSS (exactly 15 chars)
_TS_PATTERN = re.compile(r"^\d{8}_\d{6}$")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_stats(values: List[float]) -> Dict[str, Any]:
    """Compute mean, std, min, max, 95% CI for a list of floats.

    Uses Python's built-in statistics module only — no numpy/scipy required.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci_95": None,
            "n": 0,
        }
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if n > 1 else 0.0
    ci95 = 1.96 * std / (n**0.5)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "ci_95": round(ci95, 4),
        "n": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────


def _mq_mean(benchmark: Dict, key: str) -> Optional[float]:
    """Extract benchmark['model_quality'][key]['mean'], or None if missing/zero."""
    val = (benchmark.get("model_quality") or {}).get(key)
    if isinstance(val, dict):
        m = val.get("mean")
        return float(m) if m else None
    return None


def _extract_accuracy(result: Dict, benchmark: Dict) -> Optional[float]:
    # Prefer flat top-level fields (fraction 0-1) that other benchmark shapes use
    flat = (
        benchmark.get("test_accuracy")
        or benchmark.get("global_accuracy")
        or (benchmark.get("val_accuracy") or {}).get("mean")
    )
    if flat:
        return float(flat)
    # Real simulation output: model_quality.val_accuracy.mean is a *percentage* (0-100)
    for key in ("val_accuracy", "train_accuracy"):
        pct = _mq_mean(benchmark, key)
        if pct:
            return pct / 100.0  # normalise to fraction
    return None


def _extract_f1(result: Dict, benchmark: Dict) -> Optional[float]:
    flat = benchmark.get("test_f1") or benchmark.get("f1")
    if flat:
        return float(flat)
    # model_quality.test_f1.mean (may be zero if not computed)
    f1 = _mq_mean(benchmark, "test_f1")
    return f1 if f1 else None


def _extract_training_time(result: Dict, benchmark: Dict) -> Optional[float]:
    # benchmark["duration_s"] is the wall-clock FL run duration (most stable)
    if benchmark.get("duration_s"):
        return float(benchmark["duration_s"])
    timing_fit = (benchmark.get("timing") or {}).get("client_fit") or {}
    return (
        timing_fit.get("total")
        or benchmark.get("duration")
        or result.get("duration")
        or None
    )


def _extract_upload_bytes(result: Dict, benchmark: Dict) -> Optional[float]:
    upload = (benchmark.get("communication_bytes") or {}).get("upload") or {}
    return upload.get("mean") or result.get("upload_size_bytes") or None


def _extract_dp_epsilon(result: Dict, benchmark: Dict) -> Optional[float]:
    return result.get("dp_epsilon") or benchmark.get("dp_epsilon") or None


# ─────────────────────────────────────────────────────────────────────────────
# File scanning
# ─────────────────────────────────────────────────────────────────────────────


def _scan_run_dirs(root: str, verbose: bool) -> List[str]:
    """Return paths to comparison_report.json files in direct timestamp children of root.

    Only scans immediate children of root that match YYYYMMDD_HHMMSS — skips
    the dataset-level comparison_report.json and sweep subdirs like alpha_0.1/.
    """
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    found = []
    try:
        entries = sorted(os.listdir(root))
    except OSError as exc:
        print(f"[ERROR] Cannot read root directory: {exc}", file=sys.stderr)
        sys.exit(1)

    for entry in entries:
        if not _TS_PATTERN.match(entry):
            continue
        candidate = os.path.join(root, entry, "comparison_report.json")
        if os.path.isfile(candidate):
            if verbose:
                print(f"  Found: {candidate}")
            found.append(candidate)

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregation logic
# ─────────────────────────────────────────────────────────────────────────────


def aggregate(
    root: str,
    mode_filter: Optional[List[str]],
    min_runs: int,
    output_path: str,
    no_plot: bool,
    verbose: bool,
) -> Dict:
    """Load all runs, aggregate per-mode stats, save JSON and print table."""
    report_files = _scan_run_dirs(root, verbose)
    if not report_files:
        print(f"[WARN]  No timestamped run directories found under: {root}")
        print("    Make sure runs use YYYYMMDD_HHMMSS subdirectory names.")
        sys.exit(0)

    print(f"\nFound {len(report_files)} run(s) under {root}\n")

    # mode_key → list of per-run metric values
    mode_accuracies: Dict[str, List] = {}
    mode_f1s: Dict[str, List] = {}
    mode_times: Dict[str, List] = {}
    mode_upload: Dict[str, List] = {}
    mode_seeds: Dict[str, List] = {}
    mode_dp_eps: Dict[str, List] = {}  # for DP modes — scalar per run (if consistent)

    total_loaded = 0
    total_skipped = 0

    for report_path in report_files:
        try:
            with open(report_path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN]  Skipping {report_path}: cannot parse JSON ({exc})")
            total_skipped += 1
            continue

        if not isinstance(data, list):
            print(f"[WARN]  Skipping {report_path}: expected a list of result dicts")
            total_skipped += 1
            continue

        total_loaded += 1

        for result in data:
            if not isinstance(result, dict):
                continue

            mode = result.get("mode")
            if not mode:
                continue
            if mode_filter and mode not in mode_filter:
                continue

            benchmark = result.get("benchmark") or {}
            if not benchmark and not result.get("success", True):
                if verbose:
                    print(
                        f"  [WARN]  Mode '{mode}' in {report_path}: no benchmark (failed run), skipping"
                    )
                continue

            seed = result.get("seed")

            # initialise lists on first encounter
            for d in (
                mode_accuracies,
                mode_f1s,
                mode_times,
                mode_upload,
                mode_seeds,
                mode_dp_eps,
            ):
                d.setdefault(mode, [])

            mode_accuracies[mode].append(_extract_accuracy(result, benchmark))
            mode_f1s[mode].append(_extract_f1(result, benchmark))
            mode_times[mode].append(_extract_training_time(result, benchmark))
            mode_upload[mode].append(_extract_upload_bytes(result, benchmark))
            mode_seeds[mode].append(seed)
            mode_dp_eps[mode].append(_extract_dp_epsilon(result, benchmark))

    if not mode_accuracies:
        print("[WARN]  No mode results found after filtering. Nothing to aggregate.")
        sys.exit(0)

    # Determine dataset name from root path for display
    dataset_label = os.path.basename(os.path.normpath(root))

    # Build per-mode summary
    results_out: Dict[str, Any] = {}
    all_modes = sorted(mode_accuracies.keys())

    for mode in all_modes:
        seeds = [s for s in mode_seeds[mode] if s is not None]
        dp_eps_list = [e for e in mode_dp_eps[mode] if e is not None]
        dp_epsilon = dp_eps_list[0] if dp_eps_list else None

        entry: Dict[str, Any] = {
            "n_runs": len(mode_accuracies[mode]),
            "seeds": seeds,
            "accuracy": _compute_stats(mode_accuracies[mode]),
            "f1": _compute_stats(mode_f1s[mode]),
            "training_time_s": _compute_stats(mode_times[mode]),
            "upload_bytes": _compute_stats(mode_upload[mode]),
        }
        if dp_epsilon is not None:
            entry["dp_epsilon"] = dp_epsilon

        results_out[mode] = entry

    # Warn for modes with fewer than min_runs
    for mode, entry in results_out.items():
        n = entry["n_runs"]
        if n < min_runs:
            print(
                f"[WARN]  Mode '{mode}' has only {n} run{'s' if n != 1 else ''}"
                f" — {'std not available' if n == 1 else f'fewer than --min-runs={min_runs}'}",
                file=sys.stderr,
            )

    all_seeds: List[int] = sorted(
        set(s for seeds in mode_seeds.values() for s in seeds if s is not None)
    )

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "root": root,
        "total_runs_found": total_loaded,
        "total_runs_skipped": total_skipped,
        "modes_found": all_modes,
        "seeds_found": all_seeds,
        "results": results_out,
    }

    # Save JSON
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    _print_table(dataset_label, results_out, total_loaded, min_runs)
    print(f"Saved → {output_path}")

    # Optional plot
    if not no_plot:
        _plot(dataset_label, results_out, output_path)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_stat(stats: Dict, pct: bool = False) -> str:
    """Format mean ± std, or 'N/A' if no data."""
    if stats["n"] == 0 or stats["mean"] is None:
        return "N/A"
    mean = stats["mean"]
    std = stats["std"]
    n = stats["n"]
    if pct:
        mean_s = f"{mean * 100:.1f}"
        std_s = "—" if n == 1 else f"{std * 100:.1f}"
        return f"{mean_s} ± {std_s}%"
    else:
        mean_s = f"{mean:.3f}"
        std_s = "—" if n == 1 else f"{std:.3f}"
        return f"{mean_s} ± {std_s}"


def _fmt_time(stats: Dict) -> str:
    if stats["n"] == 0 or stats["mean"] is None:
        return "N/A"
    mean = stats["mean"]
    std = stats["std"]
    n = stats["n"]
    mean_s = f"{mean:.1f}"
    std_s = "—" if n == 1 else f"{std:.1f}"
    return f"{mean_s} ± {std_s}"


def _print_table(dataset: str, results: Dict, n_runs: int, min_runs: int) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Statistical Summary — {dataset.capitalize()} Dataset  ({n_runs} run(s))")
    print(sep)
    header = f"{'Mode':<22} {'N':>3}  {'Accuracy':>16}  {'F1':>16}  {'Time(s)':>14}"
    print(header)
    print("-" * 68)

    for mode, entry in sorted(results.items()):
        n = entry["n_runs"]
        dp_eps = entry.get("dp_epsilon")
        mode_label = f"{mode} (ε={dp_eps})" if dp_eps is not None else mode
        if len(mode_label) > 21:
            mode_label = mode_label[:21]

        acc_s = _fmt_stat(entry["accuracy"], pct=True)
        f1_s = _fmt_stat(entry["f1"])
        time_s = _fmt_time(entry["training_time_s"])

        print(f"{mode_label:<22} {n:>3}  {acc_s:>16}  {f1_s:>16}  {time_s:>14}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Optional matplotlib plot
# ─────────────────────────────────────────────────────────────────────────────


def _plot(dataset: str, results: Dict, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN]  matplotlib not available — skipping plot generation.")
        return

    # Try to get mode colors from registry
    mode_colors: Dict[str, str] = {}
    try:
        from fl.compare.registry import MODES

        for mk, mc in MODES.items():
            color = getattr(mc, "color", None)
            if color:
                mode_colors[mk] = color
    except Exception:
        pass

    default_palette = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]

    modes = sorted(results.keys())
    means = []
    stds = []
    ns = []
    colors = []

    for i, mode in enumerate(modes):
        acc = results[mode]["accuracy"]
        means.append(acc["mean"] if acc["mean"] is not None else 0.0)
        stds.append(acc["std"] if acc["std"] is not None and acc["n"] > 1 else 0.0)
        ns.append(acc["n"])
        colors.append(mode_colors.get(mode, default_palette[i % len(default_palette)]))

    fig, ax = plt.subplots(figsize=(max(8, len(modes) * 1.4), 5))
    x = list(range(len(modes)))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 + 0.005,
            f"N={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, min(1.05, max(means) * 1.25 + 0.05) if means else 1.05)
    ax.set_title(
        f"Accuracy by Privacy Mode — {dataset.capitalize()}  (mean ± std)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(os.path.dirname(output_path), "statistical_summary.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate statistics from repeated FL comparison runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        required=True,
        metavar="DIR",
        help="Root directory to scan (e.g. results/healthcare/)",
    )
    p.add_argument(
        "--modes",
        default=None,
        metavar="MODES",
        help="Comma-separated mode filter (default: all found modes)",
    )
    p.add_argument(
        "--min-runs",
        type=int,
        default=3,
        metavar="N",
        dest="min_runs",
        help="Warn if any mode has fewer than this many runs (default: 3)",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output JSON path (default: <root>/statistical_summary.json)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run details as they are loaded",
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    root = args.root
    mode_filter = [m.strip() for m in args.modes.split(",")] if args.modes else None
    output = args.output or os.path.join(root, "statistical_summary.json")

    aggregate(
        root=root,
        mode_filter=mode_filter,
        min_runs=args.min_runs,
        output_path=output,
        no_plot=args.no_plot,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
