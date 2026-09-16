#!/usr/bin/env python3
"""Validate stored ZKP-family runs from their chain ledgers.

Walks every ``ledgers/ledger_<mode>.json`` under the results root, validates
ZKP-family modes with ppflx_bench.compare.validation, and exits non-zero if any run
cannot be certified. See that module for what is and is not checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppflx_bench.compare.validation import (  # noqa: E402
    VERIFICATION_NOT_RECORDED,
    load_ledger_entries,
    sampled_coverage_warning,
    validate_zkp_ledger,
)

# Kept in sync with ppflx_bench.compare.registry.MODES without importing its runtime
# dependencies.
ZKP_MODES = (
    "zkp",
    "zkp_sampled",
    "he_tenseal_zkp",
    "he_concrete_tfhe_zkp",
    "he_tenseal_zkp_dp",
    "he_concrete_tfhe_zkp_dp",
)


def _expected_rounds(run_dir: Path, mode: str):
    report = run_dir / "comparison_report.json"
    if not report.exists():
        return None
    for item in json.loads(report.read_text()):
        if item.get("mode") == mode and isinstance(item.get("benchmark"), dict):
            return item["benchmark"].get("rounds")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    failed = 0
    checked = 0
    for ledger_dir in sorted(args.results_root.rglob("ledgers")):
        run_dir = ledger_dir.parent
        reports = {}
        for mode in ZKP_MODES:
            path = ledger_dir / f"ledger_{mode}.json"
            if not path.exists():
                continue
            report = validate_zkp_ledger(
                load_ledger_entries(str(path)), _expected_rounds(run_dir, mode)
            )
            reports[mode] = report
            checked += 1
            status = "PASS" if report["ok"] else "FAIL"
            failed += not report["ok"]
            print(
                f"{status}  {run_dir}  {mode:24s} rounds={report['rounds_checked']:<3} "
                f"proofs/client={report['proofs_per_client']}"
            )
            for err in report["errors"]:
                print(f"      error: {err}")
        warning = sampled_coverage_warning(reports.get("zkp"), reports.get("zkp_sampled"))
        if warning:
            print(f"WARN  {run_dir}  {warning}")

    if not checked:
        print("No ZKP-family ledgers found: nothing can be certified.")
        return 1
    print(f"\nNot checked: {VERIFICATION_NOT_RECORDED}")
    print(f"{checked} run(s) checked, {failed} failed validation")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
