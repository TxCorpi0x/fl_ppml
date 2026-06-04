"""Diagnostic annotations for comparison results."""

from __future__ import annotations

from typing import Dict, List, Optional


def add_diagnostics(results: List[Dict]) -> None:
    """Annotate each result dict with a ``diagnostics`` list of strings."""
    baseline_upload: Optional[float] = None
    for result in results:
        if result.get("mode") == "baseline" and result.get("benchmark"):
            baseline_upload = (
                result["benchmark"]
                .get("communication_bytes", {})
                .get("upload", {})
                .get("mean", 0)
            )
            break

    for result in results:
        bench = result.get("benchmark") or {}
        mode = result.get("mode", "")
        diagnostics: List[str] = []

        mq = bench.get("model_quality", {})
        timing = bench.get("timing", {})
        comm = bench.get("communication_bytes", {})

        test_acc = mq.get("test_accuracy", {}).get("mean", 0)
        test_precision = mq.get("test_precision", {}).get("mean", 0)
        test_recall = mq.get("test_recall", {}).get("mean", 0)
        test_f1 = mq.get("test_f1", {}).get("mean", 0)

        if test_recall >= 99 and test_precision <= 1 and test_acc <= 5:
            diagnostics.append(
                "prediction_collapse_all_positive: model predicts mostly/only positive class"
            )

        if "he_concrete_tfhe" in mode:
            fit_total = timing.get("client_fit", {}).get("total", 0)
            enc_total = timing.get("encryption", {}).get("total", 0)
            if fit_total <= 0 and enc_total <= 0:
                diagnostics.append(
                    "tfhe_metrics_missing: benchmark hooks did not capture fit/encryption phases"
                )
            if test_f1 < 1:
                diagnostics.append(
                    "tfhe_unstable_quality: near-random or collapsed performance"
                )

        # Legacy `he_concrete` checks removed; concrete TFHE diagnostic handled above.

        result["diagnostics"] = diagnostics
