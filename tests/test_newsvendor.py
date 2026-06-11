"""Tests for the newsvendor module."""
import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from src.newsvendor import (
    newsvendor_cost,
    optimal_newsvendor_uniform,
    monte_carlo_evaluation,
    saa_linear_program,
)


class TestNewsvendorCost:
    def test_zero_order(self):
        """With zero order, cost is always 0."""
        d = np.array([5, 10, 15])
        costs = newsvendor_cost(0, d, c=1, p=2)
        np.testing.assert_array_equal(costs, np.zeros(3))

    def test_demand_exceeds_order(self):
        """When demand > order, all units are sold."""
        d = np.array([20.0])
        cost = newsvendor_cost(10, d, c=1, p=2)
        assert cost[0] == pytest.approx(1 * 10 - 2 * 10)  # = -10

    def test_order_exceeds_demand(self):
        """When order > demand, only demand units are sold."""
        d = np.array([5.0])
        cost = newsvendor_cost(10, d, c=1, p=2)
        assert cost[0] == pytest.approx(1 * 10 - 2 * 5)  # = 0


class TestOptimalSolution:
    def test_critical_fractile(self):
        """Check that the optimal solution matches the critical fractile formula."""
        u_star, _ = optimal_newsvendor_uniform(c=1, p=2, a=5, b=15)
        expected = 5 + 10 * 0.5  # = 10
        assert u_star == pytest.approx(expected)

    def test_symmetric_case(self):
        """When c = p/2, optimal is at the median."""
        u_star, _ = optimal_newsvendor_uniform(c=1, p=2, a=0, b=10)
        assert u_star == pytest.approx(5.0)


class TestMonteCarlo:
    def test_confidence_interval_covers_true(self):
        """95% CI should contain the true value (with high probability)."""
        u_star, v_star = optimal_newsvendor_uniform(1, 2, 5, 15)
        mean, std, ci = monte_carlo_evaluation(u_star, 100_000, seed=42)
        assert abs(mean - v_star) < ci * 2  # allow some slack

    def test_reproducibility(self):
        """Same seed should give same result."""
        r1 = monte_carlo_evaluation(10, 1000, seed=123)
        r2 = monte_carlo_evaluation(10, 1000, seed=123)
        assert r1[0] == r2[0]


class TestSAA:
    def test_saa_gives_reasonable_solution(self):
        """SAA with many scenarios should be close to analytical."""
        val, u = saa_linear_program(5000, c=1, p=2, a=5, b=15, seed=42)
        u_star, v_star = optimal_newsvendor_uniform(1, 2, 5, 15)
        assert abs(u - u_star) < 1.0  # within 1 unit
        assert abs(val - v_star) < 0.5  # within 0.5 cost

    def test_saa_negative_bias(self):
        """SAA value should be <= true optimal (negative bias)."""
        _, v_star = optimal_newsvendor_uniform(1, 2, 5, 15)
        saa_values = [saa_linear_program(50, seed=i)[0] for i in range(50)]
        mean_saa = np.mean(saa_values)
        assert mean_saa <= v_star + 0.1  # allow small numerical tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

