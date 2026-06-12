"""
Newsvendor problem - analytical solution, Monte Carlo, and SAA.

Problem: min_{u >= 0}  E[ c*u - p*min(u, d) ]
For d ~ U[a, b]: u* = a + (b-a)*(p-c)/p  (critical fractile)
"""

import numpy as np


def newsvendor_cost(u, d, c=1.0, p=2.0):
    """Cost realization for order quantity u and demand vector d."""
    return c * u - p * np.minimum(u, d)


def optimal_newsvendor_uniform(c, p, a, b):
    """Analytical optimum for uniform demand U[a,b]. Returns (u*, V*)."""
    critical_fractile = (p - c) / p
    u_star = a + (b - a) * critical_fractile

    # E[min(u,d)] for d ~ U[a,b], a <= u <= b
    e_min = (u_star**2 - a**2) / (2 * (b - a)) + u_star * (b - u_star) / (b - a)
    optimal_cost = c * u_star - p * e_min
    return u_star, optimal_cost


def monte_carlo_evaluation(u, n_mc, c=1.0, p=2.0, a=5.0, b=15.0, seed=None):
    """Evaluate expected cost by Monte Carlo. Returns (mean, std, 95% CI half-width)."""
    rng = np.random.default_rng(seed)
    demand = rng.uniform(a, b, size=n_mc)
    costs = newsvendor_cost(u, demand, c, p)

    mean_cost = np.mean(costs)
    std_cost = np.std(costs, ddof=1)
    ci_half = 1.96 * std_cost / np.sqrt(n_mc)
    return mean_cost, std_cost, ci_half


def saa_linear_program(n_scenarios, c=1.0, p=2.0, a=5.0, b=15.0, seed=None):
    """
    Solve the SAA newsvendor as an LP:
        min  c*u - (p/N)*sum(y_i)
        s.t. y_i <= u, y_i <= d_i, u >= 0, y_i >= 0
    Returns (optimal_value, optimal_u).
    """
    import pulp

    rng = np.random.default_rng(seed)
    demand = rng.uniform(a, b, size=n_scenarios)

    prob = pulp.LpProblem("Newsvendor_SAA", pulp.LpMinimize)
    u = pulp.LpVariable("u", lowBound=0)
    y = [pulp.LpVariable(f"y_{i}", lowBound=0) for i in range(n_scenarios)]

    prob += c * u - (p / n_scenarios) * pulp.lpSum(y)

    for i in range(n_scenarios):
        prob += y[i] <= u
        prob += y[i] <= demand[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective), pulp.value(u)


def saa_convergence_study(n_range, c=1.0, p=2.0, a=5.0, b=15.0, n_replications=30):
    """Run SAA for different scenario counts, return convergence statistics."""
    mean_values, std_values = [], []
    mean_controls, std_controls = [], []

    for n in n_range:
        values, controls = [], []
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
