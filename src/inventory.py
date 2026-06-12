"""
Multi-stage inventory control (flower girl problem).

Buy flowers each morning at cost c, sell at price p during the day.
Unsold flowers carry over. After T days, leftover stock is lost.
"""

import numpy as np
import pulp


def solve_extensive_form(T, D, c=1.0, p=2.0, U=7):
    """
    Exact solution via extensive formulation (only practical for T <= 2).
    Returns (optimal_value, first_stage_decision).
    """
    if T > 2:
        raise ValueError("Extensive form only practical for T <= 2")

    n_scenarios = D ** T

    if T == 1:
        scenarios = np.arange(1, D + 1).reshape(-1, 1)
    else:
        scenarios = np.array([[s1, s2] for s1 in range(1, D + 1) for s2 in range(1, D + 1)])

    prob = pulp.LpProblem("Inventory_Extensive", pulp.LpMinimize)

    # decision variables per scenario
    u, v = {}, {}
    for s in range(n_scenarios):
        for t in range(T):
            u[t, s] = pulp.LpVariable(f"u_{t}_{s}", lowBound=0, upBound=U)
            v[t, s] = pulp.LpVariable(f"v_{t}_{s}", lowBound=0)

    prob += (1.0 / n_scenarios) * pulp.lpSum(
        c * u[t, s] - p * v[t, s] for t in range(T) for s in range(n_scenarios)
    )

    for s in range(n_scenarios):
        for t in range(T):
            prob += v[t, s] <= scenarios[s, t]
            prob += pulp.lpSum(v[tt, s] - u[tt, s] for tt in range(t + 1)) <= 0

    # non-anticipativity
    if T == 1:
        for s in range(1, n_scenarios):
            prob += u[0, s] == u[0, 0]
    elif T == 2:
        for s in range(1, n_scenarios):
            prob += u[0, s] == u[0, 0]
        for s1 in range(D):
            base_s = s1 * D
            for s2 in range(1, D):
                s = base_s + s2
                prob += u[1, s] == u[1, base_s]
                prob += v[0, s] == v[0, base_s]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective), np.array([pulp.value(u[0, 0])])


def anticipative_bound(scenarios, c=1.0, p=2.0, U=7):
    """
    Perfect information lower bound: solve a deterministic LP per scenario.
    Returns (mean, 95% CI half-width).
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


def simulate_policy(scenarios, policy, c=1.0, p=2.0, U=7):
    """
    Simulate a policy(t, stock)->u over demand scenarios.
    Returns (mean_cost, 95% CI half-width).
    """
    n_mc, T = scenarios.shape
    costs = np.zeros(n_mc)

    for i in range(n_mc):
        stock = 0.0
        total_cost = 0.0
        for t in range(T):
            purchase = min(max(policy(t, stock), 0), U)
            available = stock + purchase
            sold = min(available, scenarios[i, t])
            total_cost += c * purchase - p * sold
            stock = available - sold
        costs[i] = total_cost

    mean_cost = np.mean(costs)
    ci = 1.96 * np.std(costs, ddof=1) / np.sqrt(n_mc)
    return mean_cost, ci


def constant_policy(u_const):
    """Buy the same amount each day."""
    return lambda t, stock: u_const


def best_constant_policy(scenarios, c=1.0, p=2.0, U=7):
    """Grid search over constant policies. Returns (best_u, cost, ci)."""
    best_cost = np.inf
    best_u, best_ci = 0, 0

    for u in np.arange(0, U + 0.5, 0.5):
        cost, ci = simulate_policy(scenarios, constant_policy(u), c, p, U)
        if cost < best_cost:
            best_cost, best_u, best_ci = cost, u, ci

    return best_u, best_cost, best_ci


def mpc_policy(T, D, c=1.0, p=2.0, U=7):
    """
    Model Predictive Control: at each step solve a deterministic LP
    using expected demand as forecast, apply only the first decision.
    """
    expected_demand = (D + 1) / 2.0

    def policy(t, stock):
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
