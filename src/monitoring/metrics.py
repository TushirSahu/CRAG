"""Lightweight, dependency-free monitoring metrics.

Currently the Population Stability Index (PSI), the standard way to quantify how
much a distribution has shifted between a reference window and a current one.
Kept pure-numpy so it runs anywhere (CI included) without the ML stack.
"""

from __future__ import annotations

from typing import List

import numpy as np


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """Return the PSI between a reference (`expected`) and current (`actual`) sample.

    Rule of thumb: < 0.1 no shift, 0.1–0.25 moderate, > 0.25 significant drift.
    """
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Bin on the reference quantiles so both samples share identical edges.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    exp_pct = np.histogram(expected, bins=edges)[0] / expected.size + eps
    act_pct = np.histogram(actual, bins=edges)[0] / actual.size + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def embedding_norms(vectors: List[List[float]]) -> np.ndarray:
    """Reduce a set of embeddings to a 1-D signal (L2 norm per vector) for PSI."""
    if not vectors:
        return np.array([])
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
