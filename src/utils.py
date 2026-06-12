"""Utility helpers for Monte Carlo experiments."""

import numpy as np


def monte_carlo_stats(results, confidence=0.95):
    """Mean and confidence interval half-width for a sample."""
    from scipy import stats
    n = len(results)
    mean = np.mean(results)
    se = np.std(results, ddof=1) / np.sqrt(n)
    z = stats.norm.ppf((1 + confidence) / 2)
    return mean, z * se


def generate_scenarios(n_mc, T, D, seed=42):
    """Generate integer demand scenarios, each period ~ Uniform{1,...,D}."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, D + 1, size=(n_mc, T))


def format_result(mean, ci, label=""):
    """Pretty-print a Monte Carlo estimate."""
    prefix = f"{label}: " if label else ""
    return f"{prefix}{mean:.2f} ± {ci:.2f} (95% CI)"
