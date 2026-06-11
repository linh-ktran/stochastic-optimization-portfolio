"""
Newsvendor Problem - Stochastic Optimization

The newsvendor problem: buy u >= 0 units at cost c, sell at price p.
Demand d is uncertain. Unsold units are lost.

    min_{u >= 0}  E[ c*u - p*min(u, d) ]

For uniform demand d ~ U[a, b], the optimal solution is:
    u* = a + (b - a) * (p - c) / p   (critical fractile)
"""

import numpy as np
from typing import Tuple


def newsvendor_cost(u: float, d: np.ndarray, c: float = 1.0, p: float = 2.0) -> np.ndarray:
    """
    Compute the newsvendor cost for decision u and demand realizations d.

    Parameters
    ----------
    u : float
        Quantity ordered.
    d : np.ndarray
        Array of demand realizations.
    c : float
        Unit purchase cost.
    p : float
        Unit selling price.

    Returns
    -------
    np.ndarray
        Cost for each demand realization.
    """
    return c * u - p * np.minimum(u, d)


def optimal_newsvendor_uniform(c: float, p: float, a: float, b: float) -> Tuple[float, float]:
    """
    Analytical solution for the newsvendor problem with uniform demand U[a, b].

    Uses the critical fractile: P(d <= u*) = (p - c) / p

    Parameters
    ----------
    c : float
        Unit purchase cost.
    p : float
        Unit selling price.
    a : float
        Lower bound of demand distribution.
    b : float
        Upper bound of demand distribution.

    Returns
    -------
    Tuple[float, float]
        (optimal_order, optimal_cost)
    """
    critical_fractile = (p - c) / p
    u_star = a + (b - a) * critical_fractile
    # Compute expected cost analytically
    # E[cost] = c*u - p*E[min(u,d)]
    # For d ~ U[a,b]: E[min(u,d)] = integral from a to u of d/(b-a) dd + integral from u to b of u/(b-a) dd
    # = [d^2/(2(b-a))]_a^u + u*(b-u)/(b-a)
    # = (u^2 - a^2)/(2(b-a)) + u*(b-u)/(b-a)
    e_min = (u_star**2 - a**2) / (2 * (b - a)) + u_star * (b - u_star) / (b - a)
    optimal_cost = c * u_star - p * e_min
    return u_star, optimal_cost


def monte_carlo_evaluation(u: float, n_mc: int, c: float = 1.0, p: float = 2.0,
                           a: float = 5.0, b: float = 15.0,
                           seed: int = None) -> Tuple[float, float, float]:
    """
    Evaluate the expected cost of ordering u units using Monte Carlo simulation.

    Parameters
    ----------
    u : float
        Quantity ordered.
    n_mc : int
        Number of Monte Carlo samples.
    c, p : float
        Cost and price parameters.
    a, b : float
        Demand distribution bounds (uniform).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (mean_cost, std_cost, confidence_interval_half_width_95%)
    """
    rng = np.random.default_rng(seed)
    demand = rng.uniform(a, b, size=n_mc)
    costs = newsvendor_cost(u, demand, c, p)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs, ddof=1)
    ci_half = 1.96 * std_cost / np.sqrt(n_mc)
    return mean_cost, std_cost, ci_half


def saa_linear_program(n_scenarios: int, c: float = 1.0, p: float = 2.0,
                       a: float = 5.0, b: float = 15.0,
                       seed: int = None) -> Tuple[float, float]:
    """
    Solve the Sample Average Approximation (SAA) of the newsvendor problem
    as a Linear Program using PuLP.

    The SAA reformulation:
        min  c*u + (1/N) * sum_i z_i
        s.t. z_i >= -p * min(u, d_i)  =>  z_i >= -p*u, z_i >= -p*d_i  (epigraph)

    Equivalently (standard LP form):
        min  c*u - (p/N) * sum_i y_i
        s.t. y_i <= u       for all i
             y_i <= d_i     for all i
             u >= 0, y_i >= 0

    Parameters
    ----------
    n_scenarios : int
        Number of demand scenarios to sample.
    c, p : float
        Cost and price parameters.
    a, b : float
        Demand distribution bounds.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple[float, float]
        (optimal_value, optimal_order_quantity)
    """
    import pulp

    rng = np.random.default_rng(seed)
    demand = rng.uniform(a, b, size=n_scenarios)

    prob = pulp.LpProblem("Newsvendor_SAA", pulp.LpMinimize)

    u = pulp.LpVariable("u", lowBound=0)
    y = [pulp.LpVariable(f"y_{i}", lowBound=0) for i in range(n_scenarios)]

    # Objective: minimize c*u - (p/N)*sum(y_i)
    prob += c * u - (p / n_scenarios) * pulp.lpSum(y)

    # Constraints
    for i in range(n_scenarios):
        prob += y[i] <= u, f"sell_leq_order_{i}"
        prob += y[i] <= demand[i], f"sell_leq_demand_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return pulp.value(prob.objective), pulp.value(u)


def saa_convergence_study(n_range: np.ndarray, c: float = 1.0, p: float = 2.0,
                          a: float = 5.0, b: float = 15.0,
                          n_replications: int = 30) -> dict:
    """
    Study the convergence of SAA solutions as the number of scenarios increases.

    Parameters
    ----------
    n_range : np.ndarray
        Array of scenario counts to test.
    c, p, a, b : float
        Problem parameters.
    n_replications : int
        Number of independent SAA solves per scenario count.

    Returns
    -------
    dict
        Dictionary with keys 'n_scenarios', 'mean_values', 'std_values',
        'mean_controls', 'std_controls'.
    """
    mean_values = []
    std_values = []
    mean_controls = []
    std_controls = []

    for n in n_range:
        values = []
        controls = []
        for rep in range(n_replications):
            v, u = saa_linear_program(int(n), c, p, a, b, seed=rep * 1000 + int(n))
            values.append(v)
            controls.append(u)
        mean_values.append(np.mean(values))
        std_values.append(np.std(values))
        mean_controls.append(np.mean(controls))
        std_controls.append(np.std(controls))

    return {
        'n_scenarios': n_range,
        'mean_values': np.array(mean_values),
        'std_values': np.array(std_values),
        'mean_controls': np.array(mean_controls),
        'std_controls': np.array(std_controls),
    }

