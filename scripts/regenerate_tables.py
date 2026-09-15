#!/usr/bin/env python3
"""Regenerate publication tables from raw comparison reports.

The tool scans every comparison_report.json under results/ and writes one CSV
and one Markdown table per dataset. Byte-identical report copies are
deduplicated before aggregation, preventing preserved copies from appearing as
independent experimental seeds.

The wall-clock field is an estimate derived from component timers, not a
process-level elapsed-time measurement: client-side timing totals are divided
by client count (clients run in parallel), then server-side totals (proof
verification and aggregation, which run sequentially on the server) are added.

Quality metrics are the benchmark's mean over all evaluated rounds, not the
final-round model. Proofs per client per round come from the run's chain
ledger when one is stored next to the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

QUALITY_KEYS = ("test_f1", "test_accuracy", "test_auprc")


def _items(report: Any) -> list[dict[str, Any]]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]
    if isinstance(report, dict):
        return [item for item in report.get("results", []) if isinstance(item, dict)]
    return []


def _stat_value(section: dict[str, Any], key: str, field: str = "mean") -> float | None:
    value = section.get(key)
    if isinstance(value, dict):
        value = value.get(field)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _wall_clock_estimate(benchmark: dict[str, Any]) -> float | None:
    timing = benchmark.get("timing")
    if not isinstance(timing, dict):
        return None
    try:
        clients = max(int(benchmark.get("num_clients", 1)), 1)
    except (TypeError, ValueError):
        clients = 1
    client_total = sum(
        _stat_value(timing, key, "total") or 0.0
        for key in (
            "client_fit", "encryption", "decryption", "proof_generation",
            "dp_noise_addition",
        )
    )
    server_total = sum(
        _stat_value(timing, key, "total") or 0.0
        for key in ("proof_verification", "server_aggregate")
    )
    return client_total / clients + server_total


def _ledger_entries(report: Path, mode: str) -> list[dict[str, Any]] | None:
    ledger = report.parent / "ledgers" / f"ledger_{mode}.json"
    if not ledger.exists():
        return None
    raw = json.loads(ledger.read_text())
    return raw.get("ledger", []) if isinstance(raw, dict) else raw


def _elapsed_from_ledger(report: Path, mode: str) -> float | None:
    """Measured seconds from the first to the last ModelCommit of a run."""
    commits = [
        e["timestamp"] for e in _ledger_entries(report, mode) or []
        if e.get("type") == "ModelCommit" and isinstance(e.get("timestamp"), (int, float))
    ]
    return max(commits) - min(commits) if len(commits) > 1 else None


def _proofs_per_client(report: Path, mode: str) -> str | None:
    entries = _ledger_entries(report, mode)
    if entries is None:
        return None
    counts = sorted({
        len(e.get("proof_hashes") or []) // len(e["client_ids"])
        for e in entries
        if e.get("type") == "ProofAnchor" and e.get("client_ids")
    })
    return "/".join(str(c) for c in counts) if counts else None


def _record(item: dict[str, Any], source: Path, report_hash: str) -> dict[str, Any] | None:
    mode, benchmark = item.get("mode"), item.get("benchmark")
    if not isinstance(mode, str) or not isinstance(benchmark, dict):
        return None
    timing = benchmark.get("timing") or {}
    comms = benchmark.get("communication_bytes") or {}
    quality = benchmark.get("model_quality") or {}
    upload = _stat_value(comms, "upload")
    if upload is None:
        try:
            upload = float(item["upload_size_bytes"])
        except (KeyError, TypeError, ValueError):
            pass
    row: dict[str, Any] = {
        "mode": mode,
        "rounds": benchmark.get("rounds"),
        "clients": benchmark.get("num_clients"),
        "seed": item.get("seed"),
        "source": str(source),
        "report_sha256": report_hash,
        "wall_clock_s": _wall_clock_estimate(benchmark),
        "elapsed_s": _elapsed_from_ledger(source, mode),
        "proof_generation_total_s": _stat_value(timing, "proof_generation", "total"),
        "proof_verification_total_s": _stat_value(timing, "proof_verification", "total"),
        "encryption_total_s": _stat_value(timing, "encryption", "total"),
        "upload_bytes": upload,
        "proofs_per_client_round": _proofs_per_client(source, mode),
    }
    for key in QUALITY_KEYS:
        row[key] = _stat_value(quality, key)
    return row


def _mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, 0
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0, len(clean)


def _format(mean: float | None, std: float | None, places: int = 3) -> str:
    if mean is None:
        return "N/A"
    return f"{mean:.{places}f}" if not std else f"{mean:.{places}f} ± {std:.{places}f}"


def _dataset_for(results_root: Path, report: Path) -> str | None:
    try:
        relative = report.relative_to(results_root)
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def _load_reports(results_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicates: dict[str, int] = defaultdict(int)
    seen: dict[str, set[str]] = defaultdict(set)
    # Visit run directories that keep their ledgers before merged copies, so
    # deduplication retains the copy that can be cross-checked.
    paths = sorted(
        results_root.rglob("comparison_report.json"),
        key=lambda p: (not (p.parent / "ledgers").is_dir(), str(p)),
    )
    for path in paths:
        dataset = _dataset_for(results_root, path)
        if dataset is None:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen[dataset]:
            duplicates[dataset] += 1
            continue
        seen[dataset].add(digest)
        try:
            report = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        for item in _items(report):
            row = _record(item, path, digest)
            if row is not None:
                records[dataset].append(row)
    return records, duplicates


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for key in ("wall_clock", "elapsed"):
        baselines = {
            row["report_sha256"]: row[f"{key}_s"]
            for row in records if row["mode"] == "baseline"
        }
        for row in records:
            baseline = baselines.get(row["report_sha256"])
            value = row[f"{key}_s"]
            overhead = value - baseline if value is not None and baseline is not None else None
            row[f"{key}_overhead_s"] = overhead
            row[f"{key}_overhead_pct"] = (
                100 * overhead / baseline if baseline not in (None, 0) and overhead is not None else None
            )
    # Runs with different round or client counts are different experiments;
    # never average them into one row.
    for row in records:
        groups[(row["mode"], row["rounds"], row["clients"])].append(row)

    fields = (
        "elapsed_s", "elapsed_overhead_s", "elapsed_overhead_pct",
        "wall_clock_s", "wall_clock_overhead_s", "wall_clock_overhead_pct",
        "proof_generation_total_s", "proof_verification_total_s",
        "encryption_total_s", "upload_bytes", *QUALITY_KEYS,
    )
    output = []
    for (mode, rounds, clients), rows in sorted(groups.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        result: dict[str, Any] = {
            "mode": mode,
            "rounds": rounds,
            "clients": clients,
            "n_runs": len(rows),
            "seeds": ",".join(str(seed) for seed in sorted({row["seed"] for row in rows}, key=str)),
            "sources": ";".join(sorted({row["source"] for row in rows})),
            "proofs_per_client_round": ";".join(
                sorted({row["proofs_per_client_round"] for row in rows if row["proofs_per_client_round"]})
            ),
        }
        for field in fields:
            mean, std, n = _mean_std(row.get(field) for row in rows)
            result[f"{field}_mean"] = mean
            result[f"{field}_std"] = std
            result[f"{field}_n"] = n
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode", "rounds", "clients", "n_runs", "seeds",
        "elapsed_s_mean", "elapsed_s_std",
        "elapsed_overhead_s_mean", "elapsed_overhead_s_std",
        "elapsed_overhead_pct_mean", "elapsed_overhead_pct_std",
        "wall_clock_s_mean", "wall_clock_s_std",
        "wall_clock_overhead_s_mean", "wall_clock_overhead_s_std",
        "wall_clock_overhead_pct_mean", "wall_clock_overhead_pct_std",
        "proof_generation_total_s_mean", "proof_generation_total_s_std",
        "proof_verification_total_s_mean", "proof_verification_total_s_std",
        "encryption_total_s_mean", "encryption_total_s_std",
        "upload_bytes_mean", "upload_bytes_std",
        "test_f1_mean", "test_f1_std",
        "test_accuracy_mean", "test_accuracy_std",
        "test_auprc_mean", "test_auprc_std", "proofs_per_client_round", "sources",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, dataset: str, rows: list[dict[str, Any]], duplicates: int) -> None:
    lines = [
        f"# Regenerated results — {dataset}",
        "",
        "Source: unique raw comparison_report.json payloads under results/.",
        f"Excluded {duplicates} byte-identical preserved report copy/copies.",
        "",
        "Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.",
        "",
        "Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.",
        "",
        "Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.",
        "",
        "| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {rounds} | {clients} | {n_runs} | {seeds} | {elapsed} | {overhead_s} s ({overhead_pct}%) | {wall} | {proof_gen} | {proof_verify} | {encryption} | {upload} | {proofs} | {f1} | {accuracy} | {auprc} |".format(
                mode=row["mode"], rounds=row["rounds"], clients=row["clients"],
                n_runs=row["n_runs"], seeds=row["seeds"] or "N/A",
                elapsed=_format(row["elapsed_s_mean"], row["elapsed_s_std"], 1),
                overhead_s=_format(row["elapsed_overhead_s_mean"], row["elapsed_overhead_s_std"], 1),
                overhead_pct=_format(row["elapsed_overhead_pct_mean"], row["elapsed_overhead_pct_std"], 1),
                wall=_format(row["wall_clock_s_mean"], row["wall_clock_s_std"], 1),
                proof_gen=_format(row["proof_generation_total_s_mean"], row["proof_generation_total_s_std"]),
                proof_verify=_format(row["proof_verification_total_s_mean"], row["proof_verification_total_s_std"]),
                encryption=_format(row["encryption_total_s_mean"], row["encryption_total_s_std"]),
                upload=_format(row["upload_bytes_mean"], row["upload_bytes_std"], 1),
                proofs=row["proofs_per_client_round"] or "—",
                f1=_format(row["test_f1_mean"], row["test_f1_std"]),
                accuracy=_format(row["test_accuracy_mean"], row["test_accuracy_std"]),
                auprc=_format(row["test_auprc_mean"], row["test_auprc_std"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    records, duplicates = _load_reports(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_records in sorted(records.items()):
        summary = _summarize(dataset_records)
        _write_csv(args.output_dir / f"{dataset}.csv", summary)
        _write_markdown(args.output_dir / f"{dataset}.md", dataset, summary, duplicates.get(dataset, 0))
        print(f"{dataset}: {len(summary)} modes -> {args.output_dir / f'{dataset}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
