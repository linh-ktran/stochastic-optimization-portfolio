"""
Utility functions for stochastic optimization experiments.
"""

import numpy as np
from typing import Tuple


def monte_carlo_stats(results: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Monte Carlo mean and confidence interval.

    Parameters
    ----------
    results : np.ndarray
        Array of sample values.
    confidence : float
        Confidence level (default 95%).

    Returns
    -------
    Tuple[float, float]
        (mean, half_width of CI)
    """
    from scipy import stats

    n = len(results)
    mean = np.mean(results)
    se = np.std(results, ddof=1) / np.sqrt(n)
    z = stats.norm.ppf((1 + confidence) / 2)
    return mean, z * se


def generate_scenarios(n_mc: int, T: int, D: int, seed: int = 42) -> np.ndarray:
    """
    Generate demand scenarios: each demand is uniform over {1, ..., D}.

    Parameters
    ----------
    n_mc : int
        Number of scenarios.
    T : int
        Number of time periods.
    D : int
        Maximum demand.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Shape (n_mc, T) integer demand scenarios.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(1, D + 1, size=(n_mc, T))


def format_result(mean: float, ci: float, label: str = "") -> str:
    """Format a Monte Carlo result with confidence interval."""
    prefix = f"{label}: " if label else ""
    return f"{prefix}{mean:.2f} ± {ci:.2f} (95% CI)"

