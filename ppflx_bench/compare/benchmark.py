"""Benchmark aggregation helpers (ported from compare.py)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


def is_stats_dict(value: Any) -> bool:
    return isinstance(value, dict) and {"mean", "std", "min", "max", "total"}.issubset(
        set(value.keys())
    )


def aggregate_stats_dicts(stats_list: List[Dict]) -> Dict:
    if not stats_list:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "total": 0}

    # Filter out clients that never recorded this metric (total == 0).
    active = [s for s in stats_list if float(s.get("total", 0)) != 0]
    if not active:
        active = stats_list

    import numpy as np  # local import keeps module lightweight at collection time

    means = [float(s.get("mean", 0)) for s in active]
    stds = [float(s.get("std", 0)) for s in active]
    mins = [float(s.get("min", 0)) for s in active]
    maxs = [float(s.get("max", 0)) for s in active]
    totals = [float(s.get("total", 0)) for s in stats_list]  # always sum all

    return {
        "mean": float(np.mean(means)) if means else 0.0,
        "std": float(np.mean(stds)) if stds else 0.0,
        "min": float(np.min(mins)) if mins else 0.0,
        "max": float(np.max(maxs)) if maxs else 0.0,
        "total": float(np.sum(totals)) if totals else 0.0,
    }


def aggregate_client_benchmarks(
    client_benchmarks: List[Dict],
) -> Optional[Dict]:
    if not client_benchmarks:
        return None

    aggregated = copy.deepcopy(client_benchmarks[0])
    sections = ["timing", "memory_mb", "communication_bytes", "model_quality"]

    for section in sections:
        merged_section: Dict = {}
        metric_keys: set = set()
        for bench in client_benchmarks:
            metric_keys.update(bench.get(section, {}).keys())

        for metric_key in metric_keys:
            stats_list = []
            for bench in client_benchmarks:
                val = bench.get(section, {}).get(metric_key)
                if is_stats_dict(val):
                    stats_list.append(val)
            merged_section[metric_key] = aggregate_stats_dicts(stats_list)

        aggregated[section] = merged_section

    aggregated["num_clients"] = len(client_benchmarks)
    return aggregated


def merge_server_and_clients(
    server_benchmark: Optional[Dict],
    client_aggregate: Optional[Dict],
) -> Optional[Dict]:
    if server_benchmark is None:
        return client_aggregate
    if client_aggregate is None:
        return server_benchmark

    merged = copy.deepcopy(server_benchmark)
    sections = ["timing", "memory_mb", "communication_bytes", "model_quality"]

    for section in sections:
        merged_section = merged.setdefault(section, {})
        client_section = client_aggregate.get(section, {})
        for metric_key, client_stats in client_section.items():
            server_stats = merged_section.get(metric_key)
            if not is_stats_dict(server_stats):
                merged_section[metric_key] = client_stats
                continue
            server_total = float(server_stats.get("total", 0))
            client_total = float(client_stats.get("total", 0))
            if server_total <= 0 and client_total > 0:
                merged_section[metric_key] = client_stats

    return merged
