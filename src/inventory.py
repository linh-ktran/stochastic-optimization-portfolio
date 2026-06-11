"""
Multi-Stage Inventory Control (Flower Girl Problem)

A flower seller can buy flowers each morning at cost c, sell during the day at price p.
Unsold flowers carry over to the next day. After T days, remaining stock is lost.

This module implements:
- Extensive formulation (stochastic LP)
- Anticipative (perfect information) bound
- Two-stage relaxation bound
- Monte Carlo policy simulation
"""

import numpy as np
import pulp
from typing import Callable, Tuple, List


# Default parameters
DEFAULT_PARAMS = {
    'c': 1.0,    # purchase cost
    'p': 2.0,    # selling price
    'D': 10,     # max demand (demand ~ Uniform{1, ..., D})
    'U': 7,      # max purchase per day
    'T': 10,     # number of days
}


def solve_extensive_form(T: int, D: int, c: float = 1.0, p: float = 2.0,
                         U: int = 7) -> Tuple[float, np.ndarray]:
    """
    Solve the multi-stage inventory problem via extensive formulation.
    Only feasible for small T and D due to exponential scenario growth.

    Parameters
    ----------
    T : int
        Number of time periods.
    D : int
        Maximum demand (demand is uniform over {1, ..., D}).
    c, p : float
        Cost and selling price.
    U : int
        Maximum purchase quantity per period.

    Returns
    -------
    Tuple[float, np.ndarray]
        (optimal_expected_cost, first_stage_decision)
    """
    if T > 2:
        raise ValueError("Extensive form only practical for T <= 2 (exponential scenarios)")

    n_scenarios = D ** T
    prob = pulp.LpProblem("Inventory_Extensive", pulp.LpMinimize)

    # Generate all scenarios
    if T == 1:
        scenarios = np.arange(1, D + 1).reshape(-1, 1)
    else:  # T == 2
        scenarios = np.array([[s1, s2] for s1 in range(1, D + 1) for s2 in range(1, D + 1)])

    # Decision variables: u[t, scenario], v[t, scenario]
    u = {}
    v = {}
    for s in range(n_scenarios):
        for t in range(T):
            u[t, s] = pulp.LpVariable(f"u_{t}_{s}", lowBound=0, upBound=U)
            v[t, s] = pulp.LpVariable(f"v_{t}_{s}", lowBound=0)

    # Objective
    prob += (1.0 / n_scenarios) * pulp.lpSum(
        c * u[t, s] - p * v[t, s] for t in range(T) for s in range(n_scenarios)
    )

    # Constraints
    for s in range(n_scenarios):
        for t in range(T):
            # Can't sell more than demand
            prob += v[t, s] <= scenarios[s, t]
            # Can't sell more than cumulative stock
            prob += pulp.lpSum(v[tt, s] - u[tt, s] for tt in range(t + 1)) <= 0

    # Non-anticipativity constraints
    # At time t, decisions can only depend on demand realizations up to t-1
    if T == 1:
        # u[0] must be the same for all scenarios
        for s in range(1, n_scenarios):
            prob += u[0, s] == u[0, 0]
    elif T == 2:
        # u[0] deterministic (same for all)
        for s in range(1, n_scenarios):
            prob += u[0, s] == u[0, 0]
        # u[1] and v[0] depend only on d[0] (scenarios with same d[0] have same decision)
        for s1 in range(D):
            base_s = s1 * D  # first scenario with this d[0]
            for s2 in range(1, D):
                s = base_s + s2
                prob += u[1, s] == u[1, base_s]
                prob += v[0, s] == v[0, base_s]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    u_first = pulp.value(u[0, 0])
    return pulp.value(prob.objective), np.array([u_first])


def anticipative_bound(scenarios: np.ndarray, c: float = 1.0, p: float = 2.0,
                       U: int = 7) -> Tuple[float, float]:
    """
    Compute the anticipative (perfect information) lower bound.
    Solves a deterministic problem for each scenario.

    Parameters
    ----------
    scenarios : np.ndarray
        Shape (n_mc, T) - demand scenarios.
    c, p : float
        Cost and selling price.
    U : int
        Maximum purchase per period.

    Returns
    -------
    Tuple[float, float]
        (mean_bound, confidence_interval_half_width)
    """
    n_mc, T = scenarios.shape
    results = np.zeros(n_mc)

    for i in range(n_mc):
        demand = scenarios[i]
        prob = pulp.LpProblem(f"Anticipative_{i}", pulp.LpMinimize)

        u = [pulp.LpVariable(f"u_{t}", lowBound=0, upBound=U) for t in range(T)]
        v = [pulp.LpVariable(f"v_{t}", lowBound=0) for t in range(T)]

        prob += pulp.lpSum(c * u[t] - p * v[t] for t in range(T))

        for t in range(T):
            prob += v[t] <= demand[t]
            prob += pulp.lpSum(v[tt] - u[tt] for tt in range(t + 1)) <= 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        results[i] = pulp.value(prob.objective)

    mean_val = np.mean(results)
    ci = 1.96 * np.std(results, ddof=1) / np.sqrt(n_mc)
    return mean_val, ci


def simulate_policy(scenarios: np.ndarray, policy: Callable[[int, float], float],
                    c: float = 1.0, p: float = 2.0, U: int = 7) -> Tuple[float, float]:
    """
    Simulate a policy over demand scenarios.

    Parameters
    ----------
    scenarios : np.ndarray
        Shape (n_mc, T) - demand scenarios.
    policy : Callable[[int, float], float]
        Function policy(t, stock) -> purchase quantity.
    c, p : float
        Cost and selling price.
    U : int
        Maximum purchase per period.

    Returns
    -------
    Tuple[float, float]
        (mean_cost, confidence_interval_half_width)
    """
    n_mc, T = scenarios.shape
    costs = np.zeros(n_mc)

    for i in range(n_mc):
        stock = 0.0
        total_cost = 0.0
        for t in range(T):
            u = min(max(policy(t, stock), 0), U)
            available = stock + u
            sold = min(available, scenarios[i, t])
            total_cost += c * u - p * sold
            stock = available - sold
        costs[i] = total_cost

    mean_cost = np.mean(costs)
    ci = 1.96 * np.std(costs, ddof=1) / np.sqrt(n_mc)
    return mean_cost, ci


def constant_policy(u_const: float) -> Callable[[int, float], float]:
    """Create a policy that buys a constant amount each day."""
    def policy(t: int, stock: float) -> float:
        return u_const
    return policy


def best_constant_policy(scenarios: np.ndarray, c: float = 1.0, p: float = 2.0,
                         U: int = 7) -> Tuple[float, float, float]:
    """
    Find the best constant purchase policy by grid search.

    Returns
    -------
    Tuple[float, float, float]
        (best_u, best_cost, ci)
    """
    best_cost = np.inf
    best_u = 0
    best_ci = 0

    for u in np.arange(0, U + 0.5, 0.5):
        cost, ci = simulate_policy(scenarios, constant_policy(u), c, p, U)
        if cost < best_cost:
            best_cost = cost
            best_u = u
            best_ci = ci

    return best_u, best_cost, best_ci


def mpc_policy(T: int, D: int, c: float = 1.0, p: float = 2.0,
               U: int = 7) -> Callable[[int, float], float]:
    """
    Model Predictive Control policy: at each step, solve a deterministic problem
    using the expected demand for future periods.

    Returns a policy function.
    """
    expected_demand = (D + 1) / 2.0  # E[Uniform{1,...,D}]

    def policy(t: int, stock: float) -> float:
        remaining = T - t
        if remaining <= 0:
            return 0.0

        prob = pulp.LpProblem("MPC", pulp.LpMinimize)
        u_vars = [pulp.LpVariable(f"u_{tt}", lowBound=0, upBound=U) for tt in range(remaining)]
        v_vars = [pulp.LpVariable(f"v_{tt}", lowBound=0) for tt in range(remaining)]

        prob += pulp.lpSum(c * u_vars[tt] - p * v_vars[tt] for tt in range(remaining))

        for tt in range(remaining):
            prob += v_vars[tt] <= expected_demand
            cumulative_stock = stock + pulp.lpSum(u_vars[ttt] - v_vars[ttt] for ttt in range(tt))
            prob += v_vars[tt] <= cumulative_stock + u_vars[tt]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return pulp.value(u_vars[0]) if u_vars[0].varValue is not None else 0.0

    return policy

