"""
Dynamic programming for the multi-stage inventory (flower girl) problem.

State: x = current stock
Control: u = quantity purchased (0 <= u <= U)
Dynamics: x_{t+1} = max(x_t + u_t - d_t, 0)
Stage cost: c*u - p*min(x+u, d)
"""

import numpy as np


def solve_dp(T, D, c=1.0, p=2.0, U=7, max_stock=None):
    """
    Backward induction for the inventory problem.
    Returns (value_functions, optimal_policy) arrays.
    """
    if max_stock is None:
        max_stock = D + U

    V = np.zeros((T + 1, max_stock + 1))  # terminal: V[T] = 0
    policy = np.zeros((T, max_stock + 1))
    prob_d = 1.0 / D

    for t in range(T - 1, -1, -1):
        for x in range(max_stock + 1):
            best_val = np.inf
            best_u = 0

            for u in range(min(U, max_stock - x) + 1):
                available = x + u
                expected_cost = 0.0
                for d in range(1, D + 1):
                    sold = min(available, d)
                    immediate = c * u - p * sold
                    next_stock = min(available - sold, max_stock)
                    expected_cost += prob_d * (immediate + V[t + 1][next_stock])

                if expected_cost < best_val:
                    best_val = expected_cost
                    best_u = u

            V[t][x] = best_val
            policy[t][x] = best_u

    return V, policy


def dp_policy_function(policy):
    """Wrap the policy table into a callable policy(t, stock) -> u."""
    T, n_states = policy.shape

    def pol(t, stock):
        if t >= T:
            return 0.0
        x = int(min(max(round(stock), 0), n_states - 1))
        return policy[t, x]

    return pol


def print_policy_table(policy, T, max_display_stock=10):
    """Print policy as a readable table."""
    n_cols = min(max_display_stock + 1, policy.shape[1])
    print(f"{'Time':<6}", end="")
    for x in range(n_cols):
        print(f"x={x:<4}", end="")
    print()
    print("-" * (6 + 6 * n_cols))

    for t in range(T):
        print(f"t={t+1:<4}", end="")
        for x in range(n_cols):
            print(f"{int(policy[t, x]):<6}", end="")
        print()


def value_of_perfect_information(V, scenarios, c=1.0, p=2.0, U=7):
    """EVPI = DP_value - anticipative_bound."""
    from .inventory import anticipative_bound
    dp_value = V[0, 0]
    ant_bound, _ = anticipative_bound(scenarios, c, p, U)
    return dp_value - ant_bound
