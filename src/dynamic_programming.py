"""
Dynamic Programming for Inventory Control

Implements backward induction (Bellman recursion) for the multi-stage
inventory problem (Flower Girl problem).

State: x = stock at beginning of day
Control: u = quantity purchased (0 <= u <= U, stock + u <= some bound)
Transition: x_{t+1} = x_t + u_t - min(x_t + u_t, d_t)  = max(x_t + u_t - d_t, 0)
Cost: c*u - p*min(x+u, d)
"""

import numpy as np
from typing import Tuple


def solve_dp(T: int, D: int, c: float = 1.0, p: float = 2.0,
             U: int = 7, max_stock: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the inventory problem by Dynamic Programming (backward induction).

    Parameters
    ----------
    T : int
        Number of time periods.
    D : int
        Maximum demand (demand ~ Uniform{1, ..., D}).
    c : float
        Unit purchase cost.
    p : float
        Unit selling price.
    U : int
        Maximum purchase per day.
    max_stock : int, optional
        Maximum possible stock level. Defaults to D + U.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        value_functions : shape (T+1, max_stock+1) - V_t(x)
        optimal_policy : shape (T, max_stock+1) - u*_t(x)
    """
    if max_stock is None:
        max_stock = D + U

    # Value functions: V[t][x] = expected cost-to-go from state x at time t
    V = np.zeros((T + 1, max_stock + 1))  # V[T] = 0 (terminal condition)
    policy = np.zeros((T, max_stock + 1))

    # Demand probabilities (uniform over {1, ..., D})
    prob_d = 1.0 / D

    # Backward induction
    for t in range(T - 1, -1, -1):
        for x in range(max_stock + 1):
            best_val = np.inf
            best_u = 0

            # Try all feasible controls
            max_u = min(U, max_stock - x)
            for u in range(max_u + 1):
                available = x + u
                # Expected immediate cost + future cost
                expected_cost = 0.0
                for d in range(1, D + 1):
                    sold = min(available, d)
                    immediate_cost = c * u - p * sold
                    next_stock = available - sold  # = max(available - d, 0)
                    next_stock = min(next_stock, max_stock)  # clip
                    expected_cost += prob_d * (immediate_cost + V[t + 1][next_stock])

                if expected_cost < best_val:
                    best_val = expected_cost
                    best_u = u

            V[t][x] = best_val
            policy[t][x] = best_u

    return V, policy


def dp_policy_function(policy: np.ndarray):
    """
    Create a callable policy from the DP policy table.

    Parameters
    ----------
    policy : np.ndarray
        Shape (T, max_stock+1) - optimal policy table.

    Returns
    -------
    Callable[[int, float], float]
        Policy function compatible with simulate_policy.
    """
    T, n_states = policy.shape

    def pol(t: int, stock: float) -> float:
        if t >= T:
            return 0.0
        x = int(min(max(round(stock), 0), n_states - 1))
        return policy[t, x]

    return pol


def print_policy_table(policy: np.ndarray, T: int, max_display_stock: int = 10):
    """Print the optimal policy in a readable table format."""
    print(f"{'Time':<6}", end="")
    for x in range(min(max_display_stock + 1, policy.shape[1])):
        print(f"x={x:<4}", end="")
    print()
    print("-" * (6 + 6 * min(max_display_stock + 1, policy.shape[1])))

    for t in range(T):
        print(f"t={t+1:<4}", end="")
        for x in range(min(max_display_stock + 1, policy.shape[1])):
            print(f"{int(policy[t, x]):<6}", end="")
        print()


def value_of_perfect_information(V: np.ndarray, scenarios: np.ndarray,
                                 c: float = 1.0, p: float = 2.0,
                                 U: int = 7) -> float:
    """
    Compute the Expected Value of Perfect Information (EVPI).

    EVPI = V*(x0) - E[V_anticipative]

    The EVPI measures how much one would pay for perfect demand forecasts.
    """
    from .inventory import anticipative_bound

    # DP optimal value (starting from stock = 0)
    dp_value = V[0, 0]

    # Anticipative bound
    ant_bound, _ = anticipative_bound(scenarios, c, p, U)

    return dp_value - ant_bound

