#!/usr/bin/env python3
"""Quick verification of diagnostic functions."""

import sys

sys.path.insert(0, ".")

from fl.compare.benchmark import (
    is_stats_dict,
    aggregate_stats_dicts,
    aggregate_client_benchmarks,
    merge_server_and_clients,
)
from fl.compare.diagnostics import add_diagnostics

# Alias to match original test names
_is_stats_dict = is_stats_dict
_aggregate_stats_dicts = aggregate_stats_dicts
_aggregate_client_benchmarks = aggregate_client_benchmarks
_merge_server_and_clients = merge_server_and_clients
_add_diagnostics = add_diagnostics

print("[OK] All diagnostic functions imported successfully\n")

# Test 1: _is_stats_dict
test_valid = {"mean": 1, "std": 0.5, "min": 0, "max": 2, "total": 10}
test_invalid = {"mean": 1, "std": 0.5}
assert _is_stats_dict(test_valid), "should recognize valid stats dict"
assert not _is_stats_dict(test_invalid), "should reject incomplete stats dict"
print("[OK] _is_stats_dict works\n")

# Test 2: _aggregate_stats_dicts
stats_list = [
    {"mean": 10, "std": 1, "min": 9, "max": 11, "total": 100},
    {"mean": 20, "std": 2, "min": 18, "max": 22, "total": 200},
]
agg = _aggregate_stats_dicts(stats_list)
assert "mean" in agg and "total" in agg, "should aggregate to stats dict"
print(f'[OK] _aggregate_stats_dicts: mean={agg["mean"]:.1f}, total={agg["total"]:.0f}\n')

# Test 3: _add_diagnostics
results = [
    {
        "mode": "zkp",
        "benchmark": {
            "model_quality": {
                "test_accuracy": {"mean": 0.2, "total": 1},
                "test_precision": {"mean": 0.1, "total": 1},
                "test_recall": {"mean": 100, "total": 1},
                "test_f1": {"mean": 0.3, "total": 1},
            },
            "timing": {},
            "communication_bytes": {},
        },
    }
]
_add_diagnostics(results)
assert "diagnostics" in results[0], "should add diagnostics field"
assert "all_positive" in str(results[0]["diagnostics"]), "should flag collapse"
print(f'[OK] _add_diagnostics: {results[0]["diagnostics"]}\n')

print("=" * 60)
print("[OK] All verifications passed!")
print("   - Benchmark aggregation functions ready")
print("   - Diagnostic detection logic working")
print('   - Report will include "diagnostics" field')
print("=" * 60)
