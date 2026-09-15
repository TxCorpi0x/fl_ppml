"""
Update-norm bound for the ZKP modes (audit/norm.md, Step 7).

Every ZKP mode proves ‖w_local − w_global‖₂ ≤ B for the whole model, where B is
set by the server and sent to clients in the fit config. Honest clients clip
their update to B before proving, so a correct client is never rejected by the
bound; the bound limits how far one admitted client can move the global model
in a round, not in which direction.

Choosing B
    B = PER_EPOCH_UPDATE_NORM[dataset] × local_epochs

PER_EPOCH_UPDATE_NORM is KAPPA times the largest honest one-epoch update norm
observed by ``scripts/calibrate_update_norm.py`` (plain FedAvg from the seeded
initial model, harness hyperparameters). Scaling by epochs follows from the
triangle inequality over per-epoch updates. Re-run the calibration when the
model, learning rate, batch size or optimiser changes. ``FL_ZKP_MAX_NORM``
overrides the table (it is an update norm, not a weight norm).

Differential privacy adds noise to every step, so DP runs have larger honest
updates; calibrate them separately and set ``FL_ZKP_MAX_NORM``.
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import numpy as np

KAPPA = 1.5

# dataset → B for one local epoch. Source: scripts/calibrate_update_norm.py,
# results recorded in audit/norm.md ("Calibration").
PER_EPOCH_UPDATE_NORM = {
    # 3 clients, 5 rounds, seed 42, batch 16, lr 0.001: max 0.021748
    "healthcare": 0.032623,
}

FIT_CONFIG_KEY = "zkp_max_update_norm"


def max_update_norm(config) -> float:
    """The server's update-norm bound B for this run."""
    override = os.environ.get("FL_ZKP_MAX_NORM")
    if override:
        bound = float(override)
    else:
        dataset = getattr(config, "dataset", None)
        if dataset not in PER_EPOCH_UPDATE_NORM:
            raise RuntimeError(
                f"no calibrated ZKP update-norm bound for dataset {dataset!r}. Run "
                f"scripts/calibrate_update_norm.py --dataset {dataset} and add it to "
                "fl/core/update_bound.py, or set FL_ZKP_MAX_NORM"
            )
        bound = PER_EPOCH_UPDATE_NORM[dataset] * max(1, int(getattr(config, "local_epochs", 1)))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError(f"update-norm bound must be positive, got {bound}")
    return bound


def bound_from_fit_config(fit_config) -> float:
    """Client side: the bound the server sent. Missing means the server isn't enforcing one."""
    if FIT_CONFIG_KEY not in fit_config:
        raise RuntimeError("server sent no update-norm bound; refusing to prove without the server's policy")
    bound = float(fit_config[FIT_CONFIG_KEY])
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError(f"server sent an invalid update-norm bound {bound}")
    return bound


def clip_update(
    global_flat: np.ndarray,
    local_flat: np.ndarray,
    bound: float,
    fits: Callable[[np.ndarray], bool],
) -> Tuple[np.ndarray, float, bool]:
    """Scale Δ = local − global to norm ≤ bound until ``fits(new_local)`` holds.

    ``fits`` checks the exact integer statement the circuit will prove, so
    quantization can't push an honest client over the bound. Returns
    (new local values, norm of the original Δ, whether Δ was scaled).
    """
    global_flat = np.asarray(global_flat, dtype=np.float64)
    delta = np.asarray(local_flat, dtype=np.float64) - global_flat
    norm = float(np.linalg.norm(delta))
    factor = min(1.0, 0.999 * bound / norm) if norm > 0 else 1.0
    for _ in range(64):
        local = (global_flat + factor * delta).astype(np.float32)
        if fits(local):
            return local, norm, factor < 1.0
        factor *= 0.99
    raise RuntimeError(f"could not fit the update (norm {norm:.4g}) inside the bound {bound:.4g}")


def split_bound(energies, total: int):
    """Per-proof bounds declared by the client: each proof's actual energy (at least 1).

    The server accepts a proof set only if these sum to at most its total
    bound, so together the proofs bound the whole update.
    """
    bounds = [max(1, int(e)) for e in energies]
    if sum(bounds) > total:
        raise RuntimeError(f"update energy {sum(bounds)} exceeds the bound {total}")
    return bounds
