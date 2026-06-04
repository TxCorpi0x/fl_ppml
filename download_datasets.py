#!/usr/bin/env python3
"""
Centralized dataset downloader for the fhe-prediction project.

All datasets are placed under dataset/ at the project root:

  Kaggle datasets (require kaggle CLI to be configured):
    creditcard  — dataset/kaggle/input/mlg-ulb/creditcard.csv
                  Source: mlg-ulb/creditcardfraud
    healthcare  — dataset/kaggle/input/heart-disease-data/
                  Source: sid321axn/heart-statlog-cleveland-hungary-final-dataset
    stock       — dataset/kaggle/input/price-volume-data-for-all-us-stocks-etfs/
                  Source: borismarjanovic/price-volume-data-for-all-us-stocks-etfs
                  (~8 GB total; use --stock-minimal to fetch only cern.us.txt)

  Auto-downloaded via torchvision:
    mnist       — dataset/MNIST/           (~11 MB)
    cifar10     — dataset/cifar-10-batches-py/  (~170 MB)

Usage:
    # Download / verify all datasets
    python download_datasets.py

    # Specific datasets only
    python download_datasets.py --datasets mnist,cifar10
    python download_datasets.py --datasets creditcard,healthcare,stock

    # Show status without downloading
    python download_datasets.py --list

    # Re-download even if files already exist
    python download_datasets.py --force

    # Stock dataset: download only CERN ticker instead of full ~8 GB corpus
    python download_datasets.py --datasets stock --stock-minimal

Kaggle setup (one-time):
    pip install kaggle
    # Place your Kaggle API token at ~/.kaggle/kaggle.json
    # Or set env vars KAGGLE_USERNAME and KAGGLE_KEY
    # See: https://www.kaggle.com/docs/api

After downloading, point fl_ppml's comparison runner at this directory:
    cd fl_ppml
    python compare.py --dataset mnist --data-path ../dataset/ ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
#  Each entry maps a short name → metadata dict with:
#    kind        : "kaggle" | "torchvision"
#    kaggle_id   : "<owner>/<dataset-slug>" (kaggle only)
#    dest        : path relative to DATASET_DIR where files land
#    sentinel    : file/dir relative to DATASET_DIR whose existence = "done"
#    description : human-readable label
REGISTRY: dict[str, dict] = {
    "creditcard": {
        "kind": "kaggle",
        "kaggle_id": "mlg-ulb/creditcardfraud",
        "dest": "kaggle/input/mlg-ulb",
        "sentinel": "kaggle/input/mlg-ulb/creditcard.csv",
        "description": "Credit Card Fraud Detection (284 K transactions, 30 features)",
        "size_hint": "~150 MB",
    },
    "healthcare": {
        "kind": "kaggle",
        "kaggle_id": "sid321axn/heart-statlog-cleveland-hungary-final",
        "dest": "kaggle/input/heart-disease-data",
        "sentinel": "kaggle/input/heart-disease-data/heart_statlog_cleveland_hungary_final.csv",
        "description": "Heart Disease (StatLog + Cleveland + Hungary, 1190 samples)",
        "size_hint": "~50 KB",
    },
    "stock": {
        "kind": "kaggle",
        "kaggle_id": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
        "dest": "kaggle/input/price-volume-data-for-all-us-stocks-etfs",
        "sentinel": "kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cern.us.txt",
        "description": "Huge Stock Market Dataset — all US Stocks & ETFs (OHLCV)",
        "size_hint": "~8 GB (full) / ~2 MB (--stock-minimal: CERN only)",
    },
    "mnist": {
        "kind": "torchvision",
        "tv_class": "MNIST",
        "dest": "",  # torchvision places MNIST/ directly under dataset root
        "sentinel": "MNIST/raw/train-images-idx3-ubyte",
        "description": "MNIST Handwritten Digits (60 K train / 10 K test)",
        "size_hint": "~11 MB",
    },
    "cifar10": {
        "kind": "torchvision",
        "tv_class": "CIFAR10",
        "dest": "",  # torchvision places cifar-10-batches-py/ under root
        "sentinel": "cifar-10-batches-py/data_batch_1",
        "description": "CIFAR-10 Images (50 K train / 10 K test, 32×32 RGB)",
        "size_hint": "~170 MB",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if sys.stdout.isatty() else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if sys.stdout.isatty() else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if sys.stdout.isatty() else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def _check_kaggle_configured() -> bool:
    """Return True if the kaggle CLI is available and credentials are present."""
    if shutil.which("kaggle") is None:
        return False
    cred_file = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    return cred_file.exists() or env_ok


def _sentinel_present(info: dict) -> bool:
    sentinel = DATASET_DIR / info["sentinel"]
    return sentinel.exists()


def _download_kaggle(name: str, info: dict, force: bool) -> bool:
    """
    Download a Kaggle dataset using the kaggle CLI.

    Returns True on success, False on failure.
    """
    if not force and _sentinel_present(info):
        print(f"  {_green('[OK]')} Already present — skipping.")
        return True

    if not _check_kaggle_configured():
        print(
            _red("  [FAIL] kaggle CLI not configured.")
            + " Install it with  pip install kaggle\n"
            "    then place your API token at ~/.kaggle/kaggle.json\n"
            "    or export KAGGLE_USERNAME and KAGGLE_KEY."
        )
        return False

    dest = DATASET_DIR / info["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {_bold(info['kaggle_id'])} → {dest.relative_to(SCRIPT_DIR)}")
    print(f"  Expected size: {info['size_hint']}")

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        info["kaggle_id"],
        "-p",
        str(dest),
        "--unzip",
    ]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(_red(f"  [FAIL] kaggle CLI exited with code {result.returncode}."))
        return False

    if _sentinel_present(info):
        print(f"  {_green('[OK]')} Download complete.")
        return True

    # Some Kaggle datasets unzip into a subfolder named after the slug; try to
    # flatten one level if the sentinel is still missing.
    slug = info["kaggle_id"].split("/")[-1]
    slug_dir = dest / slug
    if slug_dir.is_dir():
        print(f"  Flattening {slug_dir.name}/ into {dest.name}/…")
        for item in slug_dir.iterdir():
            target = dest / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        slug_dir.rmdir() if not any(slug_dir.iterdir()) else None

    if _sentinel_present(info):
        print(f"  {_green('[OK]')} Download complete.")
        return True

    print(
        _yellow(
            f"  [WARN]  Download finished but sentinel file not found:\n"
            f"     {DATASET_DIR / info['sentinel']}\n"
            "     The dataset may have a different internal structure."
        )
    )
    return True  # Don't hard-fail; user can inspect manually.


def _download_stock_minimal(info: dict, force: bool) -> bool:
    """
    Fetch only CERN ticker file via direct HTTP (no kaggle auth required).
    Falls back to full kaggle download if URL is unavailable.
    """
    sentinel = DATASET_DIR / info["sentinel"]
    if not force and sentinel.exists():
        print(f"  {_green('[OK]')} cern.us.txt already present — skipping.")
        return True

    # The file is publicly mirrored on GitHub (borismarjanovic's repository).
    url = (
        "https://raw.githubusercontent.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
        "/master/Stocks/cern.us.txt"
    )
    print(f"  Fetching CERN ticker only from:\n    {url}")

    dest = DATASET_DIR / info["dest"] / "Stocks"
    dest.mkdir(parents=True, exist_ok=True)
    # Also keep the Data/Stocks/ mirror that notebooks use.
    data_dest = DATASET_DIR / info["dest"] / "Data" / "Stocks"
    data_dest.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        target = dest / "cern.us.txt"
        urllib.request.urlretrieve(url, target)
        # Copy to Data/Stocks/ as well (notebooks reference that path).
        shutil.copy(target, data_dest / "cern.us.txt")
        print(f"  {_green('[OK]')} cern.us.txt downloaded.")
        return True
    except Exception as exc:
        print(
            _yellow(
                f"  [WARN]  Direct download failed ({exc}). Trying full kaggle download…"
            )
        )
        return _download_kaggle("stock", info, force)


def _download_torchvision(name: str, info: dict, force: bool) -> bool:
    """Download MNIST or CIFAR-10 via torchvision into DATASET_DIR."""
    try:
        import torchvision.datasets as tvd
    except ImportError:
        print(_red("  [FAIL] torchvision not installed. Run: pip install torchvision"))
        return False

    if not force and _sentinel_present(info):
        print(f"  {_green('[OK]')} Already present — skipping.")
        return True

    dest = DATASET_DIR  # torchvision creates its own subdirectory here
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading to {dest.relative_to(SCRIPT_DIR)}/")

    cls = getattr(tvd, info["tv_class"])
    try:
        cls(root=str(dest), train=True, download=True)
        cls(root=str(dest), train=False, download=True)
    except Exception as exc:
        print(_red(f"  [FAIL] torchvision download failed: {exc}"))
        return False

    print(f"  {_green('[OK]')} Download complete.")
    return True


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def show_status(names: list[str]) -> None:
    print(
        f"\n{_bold('Dataset Status')}  (root: {DATASET_DIR.relative_to(SCRIPT_DIR)}/)\n"
    )
    col = max(len(n) for n in names) + 2
    for name in names:
        info = REGISTRY[name]
        present = _sentinel_present(info)
        mark = _green("[OK] present") if present else _yellow("[FAIL] missing")
        print(f"  {name:<{col}} {mark:30s}  {info['description']}")
        if not present:
            print(f"  {'':<{col}}   {_yellow('→')} {info['sentinel']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global DATASET_DIR  # may be overridden by --dataset-dir
    parser = argparse.ArgumentParser(
        description="Download datasets for the fhe-prediction project into dataset/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    all_names = list(REGISTRY)
    parser.add_argument(
        "--datasets",
        default=",".join(all_names),
        help=f"Comma-separated list of datasets to download. "
        f"Choices: {', '.join(all_names)}  (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show download status for each dataset and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset is already present.",
    )
    parser.add_argument(
        "--stock-minimal",
        action="store_true",
        help="For the stock dataset, download only cern.us.txt (~2 MB) "
        "instead of the full ~8 GB corpus.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DATASET_DIR),
        help=f"Override the target directory (default: {DATASET_DIR})",
    )
    args = parser.parse_args()

    # Allow override of target dir
    DATASET_DIR = Path(args.dataset_dir).resolve()

    # Resolve requested datasets
    requested = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in requested if n not in REGISTRY]
    if unknown:
        parser.error(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(all_names)}"
        )

    if args.list:
        show_status(requested)
        return

    print(f"\n{_bold('fhe-prediction — Dataset Downloader')}")
    print(f"Target directory: {DATASET_DIR}\n")

    results: dict[str, bool] = {}

    for name in requested:
        info = REGISTRY[name]
        print(f"{_bold(name)}  —  {info['description']}  [{info['size_hint']}]")

        if info["kind"] == "kaggle":
            if name == "stock" and args.stock_minimal:
                ok = _download_stock_minimal(info, args.force)
            else:
                ok = _download_kaggle(name, info, args.force)
        elif info["kind"] == "torchvision":
            ok = _download_torchvision(name, info, args.force)
        else:
            print(_red(f"  [FAIL] Unknown kind: {info['kind']}"))
            ok = False

        results[name] = ok
        print()

    # Summary
    passed = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if not ok]

    print(_bold("─" * 50))
    if passed:
        print(_green(f"  [OK] Ready:   {', '.join(passed)}"))
    if failed:
        print(_red(f"  [FAIL] Failed:  {', '.join(failed)}"))
    print()

    if failed:
        print(
            "To use kaggle datasets without the CLI, download manually from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final-dataset\n"
            "  https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs\n"
            "and place the files under dataset/kaggle/input/ as shown above."
        )
        sys.exit(1)

    print(
        "All datasets ready.\n\n"
        "To point the FL runner at this directory:\n"
        "  cd fl_ppml\n"
        "  python compare.py --dataset mnist   --data-path ../dataset/ ...\n"
        "  python compare.py --dataset cifar10 --data-path ../dataset/ ...\n"
        "  (Kaggle datasets are found automatically via relative-path lookup.)"
    )


if __name__ == "__main__":
    main()
