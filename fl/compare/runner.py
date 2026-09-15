"""
Orchestrator — CompareConfig + run_comparison().
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from fl.compare.diagnostics import add_diagnostics
from fl.compare.experiment import run_experiment
from fl.compare.plots import create_plots
from fl.compare.registry import DATASETS, MODES, DatasetConfig, ModeConfig
from fl.compare.report import print_summary
from fl.compare.validation import (
    ZKP_INTERNAL_MODES,
    load_ledger_entries,
    sampled_coverage_warning,
    validate_run,
)


@dataclass
class CompareConfig:
    """All knobs for a single comparison run."""

    dataset: str = "healthcare"
    modes: List[str] = field(default_factory=list)
    num_clients: int = 3
    num_rounds: int = 20
    max_epochs: Optional[int] = None  # None → use dataset default
    batch_size: Optional[int] = None  # None → use dataset default
    device: str = "cpu"
    data_path: str = "data/"
    output_dir: str = "results/"
    use_simulation: bool = False
    # Blockchain audit ledger
    chain_backend: str = "mock"  # "mock" | "web3" | "none"
    chain_ledger_dir: Optional[str] = None  # None → <run_dir>/ledgers/
    # Non-IID Dirichlet partitioning
    dirichlet_alpha: Optional[float] = None  # None → uniform IID split
    # Reproducibility
    seed: int = 42
    extra_args: Dict[str, Any] = field(default_factory=dict)


def _merge_dataset_defaults(cfg: CompareConfig) -> CompareConfig:
    """Fill dataset defaults and use all registered modes when unset."""
    if cfg.dataset not in DATASETS:
        known = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{cfg.dataset}'. Known: {known}")

    ds: DatasetConfig = DATASETS[cfg.dataset]

    if cfg.max_epochs is None:
        cfg.max_epochs = ds.max_epochs
    if cfg.batch_size is None:
        cfg.batch_size = ds.batch_size
    if not cfg.modes:
        cfg.modes = list(MODES.keys())
    return cfg


def _validate_modes(modes: List[str]) -> List[str]:
    """Return the validated mode list; raises ValueError for unknown modes."""
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Known: {sorted(MODES.keys())}")
    return modes


def _warn_on_heavy_image_zkp(dataset: str, modes: List[str]) -> None:
    """Warn when full ZKP modes are requested on image datasets.

    Full gnark proofs over CNN layers are memory-intensive and can trigger OOM
    in both the PyTorch process and the gnark proof service.
    """
    if dataset not in {"mnist", "cifar", "cifar10"}:
        return

    risky_modes = {
        "zkp",
        "he_tenseal_zkp",
        "he_concrete_tfhe_zkp",
        "he_tenseal_zkp_dp",
        "he_concrete_tfhe_zkp_dp",
    }
    selected = [mode for mode in modes if mode in risky_modes]
    if not selected:
        return

    print(
        "[WARN]  Full ZKP modes on image datasets are memory-intensive and may "
        "trigger OOM in both pt_main_thread and gnark_service. "
        f"Requested: {selected}. Prefer 'zkp_sampled' and set "
        "FL_ZKP_PARALLELISM=1."
    )


def run_comparison(
    dataset: str = "healthcare",
    modes: Optional[List[str]] = None,
    num_clients: int = 3,
    num_rounds: int = 20,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    data_path: str = "data/",
    output_dir: str = "results/",
    use_simulation: bool = False,
    chain_backend: str = "mock",
    chain_ledger_dir: Optional[str] = None,
    dirichlet_alpha: Optional[float] = None,
    validate_zkp: bool = True,
    **extra_args,
) -> List[Dict]:
    """Run all requested privacy modes and return results.

    Parameters
    ----------
    dataset:
        Key in DATASETS registry (cifar, healthcare, creditcard, stock).
    modes:
        Explicit list of mode keys to run; ``None`` → use dataset defaults.
    num_clients / num_rounds:
        FL hyperparameters.
    max_epochs / batch_size:
        ``None`` → filled from dataset registry.
    device:
        ``cpu`` or ``cuda``.
    data_path:
        Root data directory passed to experiment subprocesses.
    output_dir:
        Root output directory; each mode creates a sub-directory here.
    use_simulation:
        ``False`` (default) → a SuperLink with SuperNode processes; ``True`` → Flower's Simulation Runtime.
    validate_zkp:
        ``True`` (default) → every ZKP-family mode must show, in its chain
        ledger, proofs from every aggregated client in every round; any
        failure marks that result unsuccessful and raises after the report is
        written. See fl/compare/validation.py for what is not checked.
    **extra_args:
        Forwarded verbatim to the experiment subprocess.

    Returns
    -------
    list of result dicts (one per mode), with 'diagnostics' attached.
    """
    # ── build and validate config ─────────────────────────────────────────
    cfg = _merge_dataset_defaults(
        CompareConfig(
            dataset=dataset,
            modes=list(modes) if modes else [],
            num_clients=num_clients,
            num_rounds=num_rounds,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            data_path=data_path,
            output_dir=output_dir,
            use_simulation=use_simulation,
            chain_backend=chain_backend,
            chain_ledger_dir=chain_ledger_dir,
            dirichlet_alpha=dirichlet_alpha,
            seed=extra_args.pop("seed", 42),
            extra_args=extra_args,
        )
    )
    _validate_modes(cfg.modes)
    # FedPrivate samples at least min-fit-clients=2 nodes, which the harness does
    # not override; fewer clients would wait until the node timeout.
    if cfg.num_clients < 2:
        raise ValueError(f"num_clients must be at least 2, got {cfg.num_clients}")
    if cfg.num_rounds < 1:
        raise ValueError(f"num_rounds must be at least 1, got {cfg.num_rounds}")
    _warn_on_heavy_image_zkp(cfg.dataset, cfg.modes)

    # ── prerequisites check ───────────────────────────────────────────────
    runnable: List[str] = []
    skipped: List[Dict] = []
    for mode_key in cfg.modes:
        mode_cfg = MODES[mode_key]
        err = mode_cfg.check_prerequisites()
        if err is None:
            runnable.append(mode_key)
        else:
            print(f"[WARN]  Skipping '{mode_key}': {err}")
            # Requested but not run: recorded as a failed result, not silently absent.
            skipped.append({"mode": mode_key, "success": False, "skipped": err, "benchmark": None})
    if not runnable:
        raise RuntimeError("No modes can run (prerequisites missing for all).")

    # ── output dir setup ─────────────────────────────────────────────────
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, cfg.dataset, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    # ── base args forwarded to subprocesses ──────────────────────────────
    base_args: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "data_path": cfg.data_path,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "number_clients": cfg.num_clients,
        "rounds": cfg.num_rounds,
        "device": cfg.device,
        "seed": cfg.seed,
        **cfg.extra_args,
    }
    if cfg.dirichlet_alpha is not None:
        base_args["dirichlet_alpha"] = cfg.dirichlet_alpha

    # ── chain ledger dir ─────────────────────────────────────────────────
    ledger_dir = cfg.chain_ledger_dir or os.path.join(run_dir, "ledgers")
    if cfg.chain_backend != "none":
        os.makedirs(ledger_dir, exist_ok=True)

    # ── run each mode ─────────────────────────────────────────────────────
    results: List[Dict] = list(skipped)
    for mode_key in runnable:
        mode_cfg: ModeConfig = MODES[mode_key]
        # Build per-mode args so each mode gets its own ledger file.
        mode_base_args = dict(base_args)
        if cfg.chain_backend != "none":
            mode_base_args["chain_backend"] = cfg.chain_backend
            mode_base_args["chain_ledger_path"] = os.path.join(
                ledger_dir, f"ledger_{mode_key}.json"
            )
        result = run_experiment(
            mode_cfg=mode_cfg,
            display_mode=mode_key,
            base_args=mode_base_args,
            output_dir=run_dir,
            use_simulation=cfg.use_simulation,
        )
        if cfg.chain_backend != "none":
            result["chain_ledger_path"] = mode_base_args["chain_ledger_path"]
        if validate_zkp and mode_cfg.internal_mode in ZKP_INTERNAL_MODES and result.get("benchmark"):
            report = validate_run(
                load_ledger_entries(result.get("chain_ledger_path")),
                result["benchmark"].get("round_outcomes"),
                expected_rounds=cfg.num_rounds,
            )
            result["zkp_validation"] = report
            if not report["ok"]:
                result["success"] = False
                print(f"[ZKP-VALIDATION] [FAIL] {mode_key}:")
                for err in report["errors"]:
                    print(f"    {err}")
        results.append(result)

    # ── post-processing ───────────────────────────────────────────────────
    add_diagnostics(results)

    by_mode = {r.get("mode"): r for r in results}
    sampled_warning = sampled_coverage_warning(
        (by_mode.get("zkp") or {}).get("zkp_validation"),
        (by_mode.get("zkp_sampled") or {}).get("zkp_validation"),
    )
    if sampled_warning:
        by_mode["zkp_sampled"]["diagnostics"].append(sampled_warning)
        print(f"[ZKP-VALIDATION] [WARN] {sampled_warning}")

    # Promote upload_size_bytes to top-level for easy notebook access.
    for r in results:
        bm = r.get("benchmark") or {}
        upload_mean = (
            bm.get("communication_bytes", {}).get("upload", {}).get("mean", None)
        )
        r["upload_size_bytes"] = int(upload_mean) if upload_mean else None

    report_path = os.path.join(run_dir, "comparison_report.json")
    _safe_results = []
    for r in results:
        safe = {
            k: v
            for k, v in r.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        _safe_results.append(safe)
    # Annotate each result with the seed used for this run
    for r in _safe_results:
        r["seed"] = cfg.seed
    with open(report_path, "w") as f:
        json.dump(_safe_results, f, indent=2)
    print(f"\n[OK] Report saved → {report_path}")

    # Merge results into the stable dataset-level report so the notebook
    # at results/<dataset>/comparison_report.json always stays up to date.
    _merge_into_dataset_report(_safe_results, cfg.output_dir, cfg.dataset)

    create_plots(results, run_dir)
    print_summary(results)

    # ── blockchain ledger merge + summary ──────────────────────────────────
    if cfg.chain_backend != "none":
        _merge_chain_ledgers(results, ledger_dir, run_dir)

    failed = {}
    for r in results:
        if r.get("success"):
            continue
        if r.get("skipped"):
            failed[r["mode"]] = "skipped: prerequisites missing"
        elif not (r.get("zkp_validation") or {}).get("ok", True):
            failed[r["mode"]] = "ZKP validation failed"
        else:
            failed[r["mode"]] = r.get("error") or "run failed"
    if failed:
        raise RuntimeError(
            f"{len(failed)} mode(s) did not produce valid results: {failed}; see {report_path}. "
            "These runs are not valid evidence."
        )

    return results


def _merge_chain_ledgers(
    results: List[Dict],
    ledger_dir: str,
    run_dir: str,
) -> None:
    """Merge per-mode ledger JSON files into one combined audit file and print a summary.

    Each mode produces a ``ledger_<mode>.json`` under *ledger_dir* (written by
    ``fl/server.py`` via ``MockChain.save()``).  This function:

    1. Loads each per-mode ledger (skip silently if missing/empty).
    2. Writes a combined ``ledger_comparison.json`` into *run_dir*.
    3. Prints a compact audit table to stdout.
    """
    combined: Dict[str, list] = {}
    total_events = 0

    for r in results:
        mode = r.get("mode", "?")
        ledger_path = r.get("chain_ledger_path", "")
        if not ledger_path or not os.path.exists(ledger_path):
            continue
        try:
            with open(ledger_path) as f:
                raw = json.load(f)
            # MockChain.save() writes {"ledger": [...]}; handle both formats
            if isinstance(raw, dict) and "ledger" in raw:
                entries = raw["ledger"]
            elif isinstance(raw, list):
                entries = raw
            else:
                entries = []
            combined[mode] = entries
            total_events += len(entries)
        except Exception as exc:
            print(f"[WARN]  Could not read ledger for '{mode}': {exc}")

    if not combined:
        print("\n[chain] No ledger data found — chain audit skipped.")
        return

    # Save combined ledger
    combined_path = os.path.join(run_dir, "ledger_comparison.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[OK] Combined chain ledger → {combined_path}  ({total_events} event(s))")

    # Print audit summary table
    _print_chain_summary(combined)


def _print_chain_summary(combined: Dict[str, list]) -> None:
    """Print a compact per-mode blockchain event table."""
    col_w = 24
    print("\n" + "─" * 72)
    print("  BLOCKCHAIN AUDIT SUMMARY")
    print("─" * 72)
    header = f"  {'Mode':<22} {'Events':>6}  {'ModelCommit':>12}  {'ProofAnchor':>12}  {'Last Block':>10}"
    print(header)
    print("─" * 72)

    for mode, entries in combined.items():
        n_commit = sum(1 for e in entries if e.get("type") == "ModelCommit")
        n_anchor = sum(1 for e in entries if e.get("type") == "ProofAnchor")
        last_block = entries[-1].get("block", "—") if entries else "—"
        print(
            f"  {mode:<22} {len(entries):>6}  {n_commit:>12}  {n_anchor:>12}  {str(last_block):>10}"
        )
    print("─" * 72 + "\n")


def _merge_into_dataset_report(
    new_results: List[Dict],
    output_dir: str,
    dataset: str,
) -> None:
    """Merge *new_results* into ``<output_dir>/<dataset>/comparison_report.json``.

    Mode entries from *new_results* replace any existing entry for the same
    mode key; modes not present in *new_results* are kept unchanged.  This
    lets the notebook always load a single up-to-date file regardless of
    how many partial ``compare.py`` runs have been done.
    """
    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_report_path = os.path.join(dataset_dir, "comparison_report.json")

    # Load existing entries (if any)
    existing: Dict[str, Dict] = {}
    if os.path.exists(dataset_report_path):
        try:
            with open(dataset_report_path) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("results", [])
            existing = {r["mode"]: r for r in items if "mode" in r}
        except Exception as exc:
            print(
                f"[WARN]  Could not read existing dataset report ({exc}); it will be overwritten."
            )

    # Overlay new results. Failed or skipped modes never replace a stored
    # entry: that would swap valid evidence for a failure record.
    for r in new_results:
        mode = r.get("mode")
        if not (mode and r.get("success")):
            continue
        # A simulated run (HE transports plaintext) never replaces a networked one.
        new_transport = (r.get("benchmark") or {}).get("transport")
        old_transport = ((existing.get(mode) or {}).get("benchmark") or {}).get("transport", "network")
        if mode in existing and new_transport == "simulated" and old_transport != "simulated":
            print(f"[WARN]  {mode}: simulated result not merged over a networked result in the dataset report")
            continue
        existing[mode] = r

    merged = list(existing.values())
    with open(dataset_report_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"[OK] Dataset report updated → {dataset_report_path}  ({len(merged)} mode(s))"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alpha sweep
# ─────────────────────────────────────────────────────────────────────────────

_ALPHA_SWEEP_VALUES = [0.1, 0.5, 1.0, 10.0]


def run_alpha_sweep(
    dataset: str = "healthcare",
    modes: Optional[List[str]] = None,
    num_clients: int = 3,
    num_rounds: int = 20,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    data_path: str = "data/",
    output_dir: str = "results/",
    use_simulation: bool = False,
    chain_backend: str = "mock",
    chain_ledger_dir: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    seed: int = 42,
    **extra_args,
) -> Dict[float, List[Dict]]:
    """Run a Dirichlet alpha sweep and return results keyed by alpha.

    Runs ``run_comparison`` once per alpha value in *alphas* (default:
    ``[0.1, 0.5, 1.0, 10.0]``), saving each run to a separate subdirectory:
    ``<output_dir>/<dataset>/alpha_<value>/``.

    This reveals how data heterogeneity interacts with each privacy mechanism.
    For example, DP noise typically hurts convergence more under extreme
    non-IID (α=0.1) because gradient variance is already high.

    Parameters
    ----------
    alphas:
        Dirichlet concentration values to sweep.  Defaults to
        ``[0.1, 0.5, 1.0, 10.0]``.

    Returns
    -------
    Dict mapping each alpha value to the list of per-mode result dicts.
    """
    if alphas is None:
        alphas = _ALPHA_SWEEP_VALUES

    sweep_results: Dict[float, List[Dict]] = {}

    print(f"\n{'='*60}")
    print(f"  Non-IID Dirichlet Alpha Sweep")
    print(f"  Dataset: {dataset}  |  Alphas: {alphas}")
    print(f"  Modes: {modes or 'dataset defaults'}")
    print(f"{'='*60}\n")

    for alpha in alphas:
        label = f"α={alpha}"
        alpha_dir = os.path.join(output_dir, dataset, f"alpha_{alpha}")
        print(f"\n{'─'*50}")
        print(f"  Running {label}  (output → {alpha_dir})")
        print(f"{'─'*50}")

        results = run_comparison(
            dataset=dataset,
            modes=modes,
            num_clients=num_clients,
            num_rounds=num_rounds,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            data_path=data_path,
            output_dir=alpha_dir,
            use_simulation=use_simulation,
            chain_backend=chain_backend,
            chain_ledger_dir=chain_ledger_dir,
            dirichlet_alpha=alpha,
            seed=seed,
            **extra_args,
        )
        # Annotate each result with the alpha value for later analysis
        for r in results:
            r["dirichlet_alpha"] = alpha
        sweep_results[alpha] = results

    # Save consolidated sweep summary
    sweep_summary_path = os.path.join(output_dir, dataset, "alpha_sweep_summary.json")
    os.makedirs(os.path.dirname(sweep_summary_path), exist_ok=True)
    flat = [r for results in sweep_results.values() for r in results]
    safe_flat = [
        {
            k: v
            for k, v in r.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        for r in flat
    ]
    with open(sweep_summary_path, "w") as f:
        json.dump(safe_flat, f, indent=2)
    print(f"\n[OK] Alpha sweep summary saved → {sweep_summary_path}")

    _print_alpha_sweep_table(sweep_results)
    return sweep_results


def _print_alpha_sweep_table(sweep_results: Dict[float, List[Dict]]) -> None:
    """Print a compact accuracy table: rows=alpha, cols=mode."""
    all_modes = []
    for results in sweep_results.values():
        for r in results:
            m = r.get("mode", "?")
            if m not in all_modes:
                all_modes.append(m)

    col_w = 16
    header = f"  {'Alpha':>6}  " + "".join(f"{m[:col_w-1]:<{col_w}}" for m in all_modes)
    print(f"\n{'='*60}")
    print("  NON-IID ALPHA SWEEP RESULTS  (test F1 score)")
    print(f"{'='*60}")
    print(header)
    print("─" * len(header))

    for alpha in sorted(sweep_results):
        results = sweep_results[alpha]
        mode_map = {r.get("mode"): r for r in results}
        row = f"  {alpha:>6.1f}   "
        for m in all_modes:
            r = mode_map.get(m, {})
            bm = r.get("benchmark", {})
            f1 = bm.get("test_f1", bm.get("f1", None))
            if f1 is not None:
                row += f"{f1:<{col_w}.4f}"
            else:
                row += f"{'N/A':<{col_w}}"
        print(row)
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DP Epsilon sweep
# ─────────────────────────────────────────────────────────────────────────────

_EPSILON_SWEEP_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def run_dp_epsilon_sweep(
    dataset: str = "healthcare",
    modes: Optional[List[str]] = None,
    num_clients: int = 3,
    num_rounds: int = 20,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    data_path: str = "data/",
    output_dir: str = "results/",
    use_simulation: bool = False,
    chain_backend: str = "mock",
    chain_ledger_dir: Optional[str] = None,
    epsilons: Optional[List[float]] = None,
    seed: int = 42,
    **extra_args,
) -> Dict[float, List[Dict]]:
    """Run a DP epsilon sweep and return results keyed by epsilon.

    Runs ``run_comparison`` with mode ``dp`` once per epsilon value in
    *epsilons* (default: ``[0.5, 1.0, 2.0, 3.0, 5.0, 8.0]``), saving each
    run to a separate subdirectory: ``<output_dir>/<dataset>/dp_eps_<value>/``.

    This generates the privacy-utility tradeoff curve: lower ε → stronger
    privacy but higher noise, typically lower accuracy.

    Parameters
    ----------
    epsilons:
        DP privacy budget values to sweep.  Defaults to
        ``[0.5, 1.0, 2.0, 3.0, 5.0, 8.0]``.

    Returns
    -------
    Dict mapping each epsilon value to the list of per-mode result dicts.
    """
    if epsilons is None:
        epsilons = _EPSILON_SWEEP_VALUES

    # Epsilon sweep is DP-only by design; ensure "dp" is present.
    if modes is None:
        modes = ["dp"]
    elif "dp" not in modes:
        modes = ["dp"] + list(modes)

    sweep_results: Dict[float, List[Dict]] = {}

    print(f"\n{'='*60}")
    print(f"  DP Epsilon Sweep  (privacy-utility tradeoff)")
    print(f"  Dataset: {dataset}  |  ε values: {epsilons}")
    print(f"  Modes: {modes}")
    print(f"{'='*60}\n")

    for eps in epsilons:
        label = f"ε={eps}"
        eps_dir = os.path.join(output_dir, dataset, f"dp_eps_{eps}")
        print(f"\n{'─'*50}")
        print(f"  Running {label}  (output → {eps_dir})")
        print(f"{'─'*50}")

        results = run_comparison(
            dataset=dataset,
            modes=modes,
            num_clients=num_clients,
            num_rounds=num_rounds,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            data_path=data_path,
            output_dir=eps_dir,
            use_simulation=use_simulation,
            chain_backend=chain_backend,
            chain_ledger_dir=chain_ledger_dir,
            dp_epsilon=eps,
            seed=seed,
            **extra_args,
        )
        # Annotate each result with the epsilon value for later analysis
        for r in results:
            r["dp_epsilon"] = eps
        sweep_results[eps] = results

    # Save consolidated sweep summary
    sweep_summary_path = os.path.join(
        output_dir, dataset, "dp_epsilon_sweep_summary.json"
    )
    os.makedirs(os.path.dirname(sweep_summary_path), exist_ok=True)
    flat = [r for results in sweep_results.values() for r in results]
    safe_flat = [
        {
            k: v
            for k, v in r.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        for r in flat
    ]
    with open(sweep_summary_path, "w") as f:
        json.dump(safe_flat, f, indent=2)
    print(f"\n[OK] Epsilon sweep summary saved → {sweep_summary_path}")

    _print_epsilon_sweep_table(sweep_results)
    return sweep_results


def _print_epsilon_sweep_table(sweep_results: Dict[float, List[Dict]]) -> None:
    """Print a compact accuracy/noise table: rows=epsilon."""
    col_w = 12
    print(f"\n{'='*60}")
    print("  DP EPSILON SWEEP RESULTS  (test accuracy / noise σ)")
    print(f"{'='*60}")
    header = f"  {'ε':>6}  {'σ (noise)':>10}  {'Accuracy':>10}  {'F1':>10}"
    print(header)
    print("─" * len(header))

    for eps in sorted(sweep_results):
        results = sweep_results[eps]
        dp_result = next((r for r in results if r.get("mode") == "dp"), None)
        if dp_result is None:
            continue
        bm = dp_result.get("benchmark") or {}
        mq = bm.get("model_quality") or {}
        acc = (mq.get("test_accuracy") or {}).get("max", None)
        f1 = bm.get("test_f1", bm.get("f1", None))
        # Recover noise_multiplier from benchmark stats if available
        import math

        sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / eps
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        print(f"  {eps:>6.2f}  {sigma:>10.4f}  {acc_str:>10}  {f1_str:>10}")
    print(f"{'='*60}\n")
