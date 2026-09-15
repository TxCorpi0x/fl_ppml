"""
Dataset and mode registries.

To add a new dataset
--------------------
Add one DatasetConfig entry to DATASETS.  The key becomes the --dataset value.

To add a new privacy mode
-------------------------
Add one ModeConfig entry to MODES.  The key is the mode string passed to
--modes (e.g. "he_tenseal", "zkp", "dp", "baseline").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DatasetConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetConfig:
    """Per-dataset defaults used by the comparison framework."""

    batch_size: int
    max_epochs: int
    output_dir: str
    description: str = ""
    # Optional: override timeout per-dataset (seconds); None → use mode default
    timeout_override: Optional[int] = None


DATASETS: Dict[str, DatasetConfig] = {
    # ── Image datasets ─────────────────────────────────────────────────────
    # Gradient inversion attacks are far more dangerous on image data: the
    # reconstructed images are visually recognisable at 28×28 / 32×32.
    "mnist": DatasetConfig(
        batch_size=64,
        max_epochs=3,
        output_dir="results/mnist_comparison",
        description="MNIST handwritten digits — 10 classes, grayscale 28×28",
    ),
    "cifar10": DatasetConfig(
        batch_size=64,
        max_epochs=5,
        output_dir="results/cifar10_comparison",
        description="CIFAR-10 colour images — 10 classes, RGB 32×32",
    ),
    # Legacy "cifar" key — kept for backwards compatibility; maps to cifar10 loader.
    "cifar": DatasetConfig(
        batch_size=32,
        max_epochs=1,
        output_dir="results/cifar_comparison",
        description="CIFAR-10 image classification (legacy key; prefer cifar10)",
    ),
    # ── Tabular datasets ───────────────────────────────────────────────────
    "healthcare": DatasetConfig(
        batch_size=16,
        max_epochs=5,
        output_dir="results/healthcare_comparison",
        description="Patient healthcare records (HIPAA-sensitive)",
    ),
    "creditcard": DatasetConfig(
        batch_size=32,
        max_epochs=5,
        output_dir="results/creditcard_comparison",
        description="Credit card fraud detection (PCI-DSS compliance)",
    ),
    "stock": DatasetConfig(
        batch_size=16,
        max_epochs=5,
        output_dir="results/stock_comparison",
        description="Stock market price prediction",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# ModeConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModeConfig:
    """
    Per-privacy-mode configuration.

    ``internal_mode`` and ``he_backend`` map to the legacy (mode, he_backend)
    tuple used for reporting and ZKP run validation.
    """

    # Legacy internal fields consumed by experiment.py
    internal_mode: str  # "he" | "zkp" | "dp" | "baseline"
    he_backend: Optional[str]  # "tenseal" | "concrete" | "concrete_tfhe" | None

    # Visual / reporting
    color: str

    # How long to wait for this mode before declaring timeout (seconds)
    timeout_s: int

    # Path that must exist before the mode can run.  None = no prerequisite.
    requires_key: Optional[str] = None

    # Human-readable name for plots / headers (auto-derived if not set)
    display_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.display_name is None:
            self.display_name = self.key  # set by registry after construction

    @property
    def key(self) -> str:  # convenience; overridden by registry lookup caller
        return (
            f"{self.internal_mode}_{self.he_backend}"
            if self.he_backend
            else self.internal_mode
        )

    def check_prerequisites(self) -> Optional[str]:
        """Return an error message if prerequisites are missing, else None."""
        if not self.requires_key:
            return None
        # ZKP pedersen: skip check when gnark backend is active
        if self.internal_mode == "zkp":
            zkp_backend = os.environ.get("FL_ZKP_BACKEND", "gnark").lower()
            if zkp_backend == "gnark":
                return None  # gnark doesn't need a pre-generated params file
        if not os.path.exists(self.requires_key):
            cmd_hint = {
                "keys/he_tenseal/secret_context.bin": "python -m fl.keys generate he_tenseal",
                "keys/zkp/zkp_params.json": "python -m fl.keys generate zkp",
                "keys/dp/dp_params.json": "python -m fl.keys generate dp",
            }.get(self.requires_key, f"python -m fl.keys generate <type>")
            return f"Missing prerequisite: {self.requires_key}\n" f"  Run: {cmd_hint}"
        return None


MODES: Dict[str, ModeConfig] = {
    "baseline": ModeConfig(
        internal_mode="baseline",
        he_backend=None,
        color="#2ecc71",
        timeout_s=600,
        requires_key=None,
        display_name="Baseline",
    ),
    "he_tenseal": ModeConfig(
        internal_mode="he",
        he_backend="tenseal",
        color="#e74c3c",
        timeout_s=3600,
        requires_key="keys/he_tenseal/secret_context.bin",
        display_name="HE-TenSEAL",
    ),
    # Legacy HE-Concrete mode removed; use he_concrete_tfhe instead.
    "he_concrete_tfhe": ModeConfig(
        internal_mode="he",
        he_backend="concrete_tfhe",
        color="#8e44ad",
        timeout_s=3600,
        requires_key=None,  # generates keys per-client at runtime
        display_name="HE-TFHE",
    ),
    "zkp": ModeConfig(
        internal_mode="zkp",
        he_backend=None,
        color="#3498db",
        timeout_s=3600,
        requires_key="keys/zkp/zkp_params.json",
        display_name="ZKP",
    ),
    "zkp_sampled": ModeConfig(
        internal_mode="zkp",
        he_backend=None,
        color="#2c7fb8",
        timeout_s=3600,
        requires_key="keys/zkp/zkp_params.json",
        display_name="ZKP (sampled)",
    ),
    "dp": ModeConfig(
        internal_mode="dp",
        he_backend=None,
        color="#f39c12",
        timeout_s=600,
        requires_key="keys/dp/dp_params.json",
        display_name="DP",
    ),
    # ── Hybrid FHE + ZKP modes ─────────────────────────────────────────────
    # internal_mode="he_zkp" groups the composites for reporting and ZKP run
    # validation; runs select the privacy mode by its registry key.
    "he_tenseal_zkp": ModeConfig(
        internal_mode="he_zkp",
        he_backend="tenseal",
        color="#c0392b",
        timeout_s=7200,
        requires_key="keys/he_tenseal/secret_context.bin",
        display_name="HE-TenSEAL + ZKP (unbound)",
    ),
    "he_concrete_tfhe_zkp": ModeConfig(
        internal_mode="he_zkp",
        he_backend="concrete_tfhe",
        color="#6c3483",
        timeout_s=7200,
        requires_key=None,  # Concrete TFHE generates keys per-client at runtime
        display_name="HE-TFHE + ZKP (unbound)",
    ),
    # Verifiable ElGamal: proofs are bound to the aggregated ciphertexts
    # (docs/ZKP.md, section 6.3).
    "he_elgamal_zkp": ModeConfig(
        internal_mode="he_zkp",
        he_backend="elgamal",
        color="#16a085",
        timeout_s=14400,
        requires_key="keys/he_elgamal/secret_key.json",
        display_name="HE-ElGamal + ZKP (bound)",
    ),
    # Commit–challenge coordinate sampling over committed ElGamal ciphertexts
    # (docs/ZKP.md, section 6.4). Two Flower rounds per federated round.
    "he_elgamal_zkp_sampled": ModeConfig(
        internal_mode="he_zkp",
        he_backend="elgamal",
        color="#1abc9c",
        timeout_s=14400,
        requires_key="keys/he_elgamal/secret_key.json",
        display_name="HE-ElGamal + sampled ZKP (commit-challenge)",
    ),
    # ── Triple: HE + ZKP + DP  ─────────────────────────────────────────────
    # internal_mode="he_zkp_dp": HE + ZKP + DP composites, selected by registry key.
    "he_tenseal_zkp_dp": ModeConfig(
        internal_mode="he_zkp_dp",
        he_backend="tenseal",
        color="#922b21",
        timeout_s=7200,
        requires_key="keys/he_tenseal/secret_context.bin",
        display_name="HE-TenSEAL + ZKP (unbound) + DP",
    ),
    "he_concrete_tfhe_zkp_dp": ModeConfig(
        internal_mode="he_zkp_dp",
        he_backend="concrete_tfhe",
        color="#4a235a",
        timeout_s=7200,
        requires_key=None,  # Concrete TFHE generates keys per-client at runtime
        display_name="HE-TFHE + ZKP (unbound) + DP",
    ),
}
