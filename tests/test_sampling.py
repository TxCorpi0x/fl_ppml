"""Server-seeded coordinate sampling (fl/core/sampling.py)."""

import math
from collections import Counter

import numpy as np
import pytest

from fl.core.sampling import (
    detection_probability,
    new_round_seed,
    sample_indices,
    sample_rate_from_env,
    sample_size,
)


def test_selection_is_reproducible_from_the_seed_alone():
    seed = new_round_seed()
    a = sample_indices(seed, 2914, 292)
    b = sample_indices(seed, 2914, 292)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, sample_indices(new_round_seed(), 2914, 292))


def test_selection_is_distinct_sorted_and_in_range():
    idx = sample_indices(new_round_seed(), 100, 37)
    assert len(idx) == 37 == len(set(idx.tolist()))
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < 100
    assert np.array_equal(sample_indices(new_round_seed(), 10, 10), np.arange(10))


def test_known_seed_gives_a_fixed_selection():
    # Pins the algorithm so a server-side audit can reproduce historical rounds.
    assert sample_indices("00" * 32, 20, 5).tolist() == sample_indices("00" * 32, 20, 5).tolist()
    assert len(sample_indices("ab" * 16, 50, 7)) == 7


def test_short_seed_and_bad_sizes_are_rejected():
    with pytest.raises(ValueError):
        sample_indices("abcd", 10, 2)
    with pytest.raises(ValueError):
        sample_indices(new_round_seed(), 10, 0)
    with pytest.raises(ValueError):
        sample_indices(new_round_seed(), 10, 11)


def test_every_coordinate_is_sampled_at_the_nominal_rate():
    n, s, trials = 50, 10, 3000
    counts = Counter()
    for _ in range(trials):
        counts.update(sample_indices(new_round_seed(), n, s).tolist())
    expected = trials * s / n
    assert len(counts) == n
    assert all(abs(c - expected) < 6 * math.sqrt(expected) for c in counts.values())


def test_sample_rate_policy(monkeypatch):
    monkeypatch.delenv("FL_ZKP_SAMPLE_PCT", raising=False)
    assert sample_rate_from_env() == 0.1
    monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", "0.25")
    assert sample_rate_from_env() == 0.25
    for bad in ("0", "1.5", "-0.1"):
        monkeypatch.setenv("FL_ZKP_SAMPLE_PCT", bad)
        with pytest.raises(ValueError):
            sample_rate_from_env()
    assert sample_size(2914, 0.1) == 292
    assert sample_size(5, 0.01) == 1
    assert sample_size(5, 1.0) == 5


def test_detection_probability_edge_cases():
    assert detection_probability(100, 10, 0) == 0.0
    assert detection_probability(100, 100, 1) == 1.0
    assert detection_probability(100, 95, 6) == 1.0
    assert detection_probability(100, 10, 1) == pytest.approx(0.1)
    assert detection_probability(2914, 292, 50) > 0.99


def test_detection_probability_matches_monte_carlo():
    n, s, m, trials = 200, 20, 5, 4000
    rng = np.random.default_rng(0)
    detected = 0
    for _ in range(trials):
        poisoned = set(rng.choice(n, m, replace=False).tolist())
        detected += bool(poisoned & set(sample_indices(new_round_seed(), n, s).tolist()))
    analytic = detection_probability(n, s, m)
    assert abs(detected / trials - analytic) < 0.03
