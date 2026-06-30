"""Tests for the pure-numpy monitoring metrics (PSI)."""

import numpy as np

from src.monitoring.metrics import embedding_norms, population_stability_index


def test_psi_is_zero_for_identical_distributions():
    x = np.random.RandomState(0).normal(size=500)
    assert population_stability_index(x, x) < 1e-6


def test_psi_detects_a_clear_shift():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(3, 1, 500)
    assert population_stability_index(a, b) > 0.25


def test_embedding_norms():
    norms = embedding_norms([[3.0, 4.0], [0.0, 0.0]])
    assert list(norms) == [5.0, 0.0]


if __name__ == "__main__":
    test_psi_is_zero_for_identical_distributions()
    test_psi_detects_a_clear_shift()
    test_embedding_norms()
    print("✅ metrics tests passed")
