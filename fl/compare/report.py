"""
Text summary report for comparison results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _mean(d: Any) -> float:
    """Extract mean from a timing/metric sub-dict, or return 0."""
    if isinstance(d, dict):
        return float(d.get("mean", 0) or 0)
    return float(d or 0)


def _total(d: Any) -> float:
    """Extract total from a timing/metric sub-dict, or return 0."""
    if isinstance(d, dict):
        return float(d.get("total", 0) or 0)
    return float(d or 0)


def _fmt_upload(mb: float) -> str:
    """Format upload size: KB for small values, 1dp MB otherwise."""
    if mb == 0.0:
        return "0.0"
    if mb < 0.1:
        return f"{mb * 1024:.0f} KB"
    return f"{mb:.1f} MB"


def _fmt_crypto(seconds: float) -> str:
    """Format crypto overhead in seconds."""
    if seconds == 0.0:
        return "—"
    return f"{seconds:.3f}s"


def _round_mean(model_quality: Dict, key: str) -> Optional[float]:
    """Return the round-mean of one quality metric on a 0-100 scale, or None.

    Each metric gets its own column: an earlier helper returned AUPRC when
    present and printed it under an accuracy header.
    """
    v = model_quality.get(key)
    if v is None:
        return None
    val = _mean(v) if isinstance(v, dict) else float(v)
    if val <= 0:
        return None
    return val * 100 if val <= 1.0 else val


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a tabular summary of all mode results to stdout."""
    header = f"{'Mode':<20} {'Status':<10} {'Time(s)':>9} {'Fit/rnd(s)':>11} {'Crypto/rnd':>12} {'Upload':>11} {'AUPRC(%)':>9} {'Acc(%)':>7}"
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    rows = []  # collect per-row numeric values for totals

    for r in results:
        mode = r.get("mode", "?")
        status = "[OK] OK" if r.get("success") else "[FAIL] FAIL"
        bm = r.get("benchmark") or {}
        timing = bm.get("timing", {})
        model_quality = bm.get("model_quality", {})
        rounds = int(bm.get("rounds", 10))
        num_clients = max(int(bm.get("num_clients", 1)), 1)
        upload_bytes = (
            bm.get("communication_bytes", {}).get("upload", {}).get("mean", 0)
        ) or 0
        upload_mb = upload_bytes / (1024 * 1024)

        fit_mean = _mean(timing.get("client_fit"))

        def _tot(key: str) -> float:
            return float((timing.get(key) or {}).get("total", 0))

        # Wall-clock estimate:
        #  - Client-side timings (fit, encrypt, decrypt, proof) run in parallel
        #    across num_clients; sum / num_clients gives per-round wall-clock.
        #  - Server-side aggregation is serial; add its total directly.
        client_total = (
            _tot("client_fit")
            + _tot("encryption")
            + _tot("decryption")
            + _tot("proof_generation")
            + _tot("proof_verification")
            + _tot("dp_noise_addition")
        )
        total_time = client_total / num_clients + _tot("server_aggregate")

        # Crypto overhead per round (mean per call across all rounds × clients).
        # For HE modes: encrypt + decrypt (client-side) + server HE aggregation.
        # For ZKP modes: proof_generation + proof_verification.
        # For DP: noise addition.
        is_he = "he" in mode
        is_zkp = "zkp" in mode
        if is_he and is_zkp:
            crypto = (
                _mean(timing.get("encryption"))
                + _mean(timing.get("decryption"))
                + _mean(timing.get("server_aggregate"))
                + _mean(timing.get("proof_generation"))
                + _mean(timing.get("proof_verification"))
            )
        elif is_he:
            # Server aggregation is homomorphic arithmetic — real FHE cost.
            crypto = (
                _mean(timing.get("encryption"))
                + _mean(timing.get("decryption"))
                + _mean(timing.get("server_aggregate"))
            )
        elif is_zkp:
            crypto = _mean(timing.get("proof_generation")) + _mean(
                timing.get("proof_verification")
            )
        elif mode == "dp":
            crypto = _mean(timing.get("dp_noise_addition"))
        else:
            crypto = 0.0

        crypto_str = _fmt_crypto(crypto)

        auprc = _round_mean(model_quality, "test_auprc")
        acc = _round_mean(model_quality, "test_accuracy")

        # diagnostics warnings
        diag_str = ""
        for d in r.get("diagnostics", []):
            if isinstance(d, str):
                diag_str += f"\n    [WARN]  {d}"
            elif isinstance(d, dict) and d.get("type") == "warning":
                diag_str += f"\n    [WARN]  {d['message']}"

        upload_str = _fmt_upload(upload_mb)
        print(
            f"{mode:<20} {status:<10} {total_time:>9.1f} {fit_mean:>11.1f}"
            f" {crypto_str:>12} {upload_str:>11} {_fmt_pct(auprc):>9} {_fmt_pct(acc):>7}"
            + diag_str
        )

        rows.append(
            {
                "total_time": total_time,
                "fit_mean": fit_mean,
                "crypto": crypto,
                "upload_mb": upload_mb,
                "auprc": auprc,
                "acc": acc,
            }
        )

    # ── Totals / averages row ─────────────────────────────────────────────
    if rows:
        print(sep)
        sum_time = sum(row["total_time"] for row in rows)
        sum_fit = sum(row["fit_mean"] for row in rows)
        sum_crypto = sum(row["crypto"] for row in rows)
        sum_upload = sum(row["upload_mb"] for row in rows)

        def _avg(key: str) -> Optional[float]:
            vals = [row[key] for row in rows if row[key] is not None]
            return sum(vals) / len(vals) if vals else None

        crypto_tot_str = _fmt_crypto(sum_crypto)
        upload_tot_str = _fmt_upload(sum_upload)
        print(
            f"{'TOTAL/AVG':<20} {'':<10} {sum_time:>9.1f} {sum_fit:>11.1f}"
            f" {crypto_tot_str:>12} {upload_tot_str:>11}"
            f" {_fmt_pct(_avg('auprc')):>9} {_fmt_pct(_avg('acc')):>7}"
        )

    print(sep)
    print("AUPRC(%) and Acc(%) are test-set means over all evaluated rounds, not final-round scores.\n")
