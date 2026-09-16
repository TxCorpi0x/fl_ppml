"""
ppflx.keys.dp — Differential Privacy parameter generation and loading.

Key material:
    dp_params.json   — serialised DifferentialPrivacyParams

Usage::

    from ppflx.keys.dp import generate, load

    generate(output="keys/dp/dp_params.json", epsilon=1.0, delta=1e-5)
    dp = load("keys/dp/dp_params.json")
    print(dp.noise_multiplier)
"""

from __future__ import annotations

import os


# ── Privacy-budget interpretation helpers ─────────────────────────────────────


def _privacy_label(epsilon: float) -> str:
    if epsilon < 0.5:
        return "Very Strong  (significant accuracy cost)"
    if epsilon < 1.0:
        return "Strong       (moderate accuracy cost)"
    if epsilon < 3.0:
        return "Moderate     (minimal accuracy cost)"
    return "Weak         (negligible accuracy cost)"


def _recommendation(epsilon: float) -> str:
    if epsilon < 0.5:
        return "High-security medical/financial data"
    if epsilon < 1.0:
        return "Sensitive personal data"
    if epsilon < 3.0:
        return "General-purpose privacy protection"
    return "Compliance requirements only"


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    output: str = "keys/dp/dp_params.json",
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    mechanism: str = "gaussian",
    noise_multiplier: float | None = None,
    overwrite: bool = False,
) -> "DifferentialPrivacyParams":  # noqa: F821
    """
    Create and save Differential Privacy parameters.

    Args:
        output:           Destination path for the JSON file.
        epsilon:          Privacy budget ε (smaller = more private, typical 0.1–10).
        delta:            Failure probability δ (typical 1e-5).
        max_grad_norm:    Gradient clipping bound C (typical 0.1–2.0).
        mechanism:        Noise mechanism: "gaussian" or "laplace".
        noise_multiplier: Override auto-computed noise multiplier.
        overwrite:        Raise if file exists and this is False.

    Returns:
        The constructed :class:`~ppflx.core.security.DifferentialPrivacyParams`.
    """
    from ppflx.core.security import DifferentialPrivacyParams, save_dp_params

    if os.path.exists(output) and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Pass overwrite=True to regenerate."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    params = DifferentialPrivacyParams(
        epsilon=epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        mechanism=mechanism,
    )
    save_dp_params(params, output)

    print(f"\n{'='*60}")
    print("Differential Privacy Parameters")
    print(f"{'='*60}")
    print(f"  ε (epsilon):       {params.epsilon}")
    print(f"  δ (delta):         {params.delta}")
    print(f"  Max grad norm C:   {params.max_grad_norm}")
    print(f"  Noise multiplier:  {params.noise_multiplier:.4f}")
    print(f"  Mechanism:         {params.mechanism}")
    print(f"\n  Privacy level:     {_privacy_label(epsilon)}")
    print(f"  Recommended for:   {_recommendation(epsilon)}")
    print(f"\n[OK] Saved → {output}")

    return params


def load(path: str = "dp_params.json") -> "DifferentialPrivacyParams":  # noqa: F821
    """Load and return a :class:`~ppflx.core.security.DifferentialPrivacyParams`."""
    from ppflx.core.security import load_dp_params

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DP params file not found: {path}\n" f"Run: python -m ppflx.keys generate dp"
        )
    return load_dp_params(path)
