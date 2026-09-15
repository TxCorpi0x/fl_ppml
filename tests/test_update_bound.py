"""Update-norm bound helpers (fl/core/update_bound.py, audit/norm.md). No gnark service needed."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from fl.core import elgamal_gnark as eg
from fl.core import update_bound as ub
from fl.core import zkp_gnark


def test_bound_scales_with_local_steps_unless_overridden(monkeypatch):
    monkeypatch.delenv("FL_ZKP_MAX_NORM", raising=False)
    monkeypatch.setitem(ub.PER_STEP_UPDATE_NORM, "toy", 0.001)
    # N-3: fewer clients → larger shards → more batches → a proportionally larger bound.
    assert ub.max_update_norm(NS(dataset="toy", local_epochs=3, max_client_batches=10)) == pytest.approx(0.03)
    assert ub.max_update_norm(NS(dataset="toy", local_epochs=1, max_client_batches=15)) == pytest.approx(0.015)
    with pytest.raises(RuntimeError, match="batches per epoch"):
        ub.max_update_norm(NS(dataset="toy", local_epochs=1))
    with pytest.raises(RuntimeError, match="no calibrated"):
        ub.max_update_norm(NS(dataset="not-a-dataset", local_epochs=1, max_client_batches=5))
    monkeypatch.setenv("FL_ZKP_MAX_NORM", "0.25")
    assert ub.max_update_norm(NS(dataset="not-a-dataset", local_epochs=9)) == 0.25
    monkeypatch.setenv("FL_ZKP_MAX_NORM", "0")
    with pytest.raises(ValueError):
        ub.max_update_norm(NS(dataset="toy", local_epochs=1, max_client_batches=5))


def test_strategy_derives_client_batches_for_the_bound(monkeypatch):
    import torch

    from fl.config import FLConfig
    from fl.privacy.zkp import ZKPMode
    from fl.server import make_strategy

    monkeypatch.delenv("FL_ZKP_MAX_NORM", raising=False)
    monkeypatch.setitem(ub.PER_STEP_UPDATE_NORM, "healthcare", 0.001)
    config = FLConfig(dataset="healthcare", local_epochs=2, sim_mode=False, zkp_backend="gnark")
    config.num_classes = 2
    testloader = [(torch.zeros(4, 13), torch.zeros(4, dtype=torch.long))]
    mode = ZKPMode()

    strategy = make_strategy(config, mode, testloader, client_batches=12)

    assert mode._max_update_norm == pytest.approx(0.001 * 2 * 12)
    assert strategy.mode.fit_config(1) == {ub.FIT_CONFIG_KEY: str(mode._max_update_norm)}


def test_client_refuses_to_prove_without_the_servers_bound():
    with pytest.raises(RuntimeError, match="no update-norm bound"):
        ub.bound_from_fit_config({"server_round": 1})
    assert ub.bound_from_fit_config({ub.FIT_CONFIG_KEY: "0.5"}) == 0.5


def test_clip_update_scales_only_oversized_updates():
    rng = np.random.default_rng(0)
    g = rng.normal(size=100)
    small = g + 0.01 * rng.normal(size=100)
    out, norm, clipped = ub.clip_update(g, small, 1.0, lambda v: True)
    assert not clipped and np.allclose(out, small.astype(np.float32))
    big = g + 5 * rng.normal(size=100)
    out, norm, clipped = ub.clip_update(g, big, 1.0, lambda v: True)
    assert clipped and norm > 1.0 and np.linalg.norm(out.astype(np.float64) - g) <= 1.0 + 1e-5


def test_split_bound_refuses_energy_above_the_total():
    assert ub.split_bound([0, 5, 7], 13) == [1, 5, 7]
    with pytest.raises(RuntimeError):
        ub.split_bound([0, 5, 7], 12)


@pytest.mark.parametrize("seed", range(5))
def test_plaintext_clipped_update_always_fits_the_server_bound(seed):
    rng = np.random.default_rng(seed)
    n, bound = 3000, 0.03
    g = rng.normal(scale=0.3, size=n).astype(np.float32)
    local = (g + rng.normal(scale=1.0, size=n)).astype(np.float32)  # far over the bound
    total = zkp_gnark.policy_bound_sq(bound, n)
    g_q = zkp_gnark.quantize(g)
    new, _, clipped = ub.clip_update(g, local, bound, lambda v: True)
    assert clipped
    assert zkp_gnark.energy(zkp_gnark.quantize(new) - g_q) + 12 <= total  # 12 = proofs for 3000 values at n=256


@pytest.mark.parametrize("weight", [1, 3, 40])
def test_elgamal_clipped_update_always_fits_the_server_bound(weight):
    rng = np.random.default_rng(weight)
    n, bound, scale = 2914, 0.0326, 10_000
    sums = rng.integers(-3000 * weight, 3000 * weight, size=n)
    glob = eg.GlobalModel(weight=weight, ct=b"\0" * (n * eg.CIPHERTEXT_BYTES), sums=sums)
    local = sums / (weight * scale) + rng.normal(scale=0.5, size=n)
    q, norm, clipped = eg.quantize_update(glob, local, bound, scale)
    assert clipped and norm > bound
    assert eg.update_energy(q, sums, weight) + 23 <= eg.total_bound_sq(bound, scale, weight, n)
    # And an unclipped update of the same size can't be declared within it.
    raw = np.round(local * scale).astype(np.int64)
    assert eg.update_energy(raw, sums, weight) > eg.total_bound_sq(bound, scale, weight, n)
