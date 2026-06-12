"""Basic tests for the newsvendor module."""
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


def test_cost_zero_order():
    d = np.array([5, 10, 15])
    assert all(newsvendor_cost(0, d) == 0)


def test_cost_all_sold():
    # demand > order: everything sells
    assert newsvendor_cost(10, np.array([20.0]))[0] == pytest.approx(-10.0)


def test_cost_excess_order():
    # order > demand: only demand sells
    assert newsvendor_cost(10, np.array([5.0]))[0] == pytest.approx(0.0)


def test_optimal_critical_fractile():
    u_star, _ = optimal_newsvendor_uniform(c=1, p=2, a=5, b=15)
    # (p-c)/p = 0.5, so u* = 5 + 10*0.5 = 10
    assert u_star == pytest.approx(10.0)


def test_optimal_value():
    _, v_star = optimal_newsvendor_uniform(c=1, p=2, a=5, b=15)
    # manual: E[min(10,d)] = 8.75, so V* = 10 - 2*8.75 = -7.5
    assert v_star == pytest.approx(-7.5)


def test_mc_covers_true_value():
    u_star, v_star = optimal_newsvendor_uniform(1, 2, 5, 15)
    mean, _, ci = monte_carlo_evaluation(u_star, 100_000, seed=42)
    assert abs(mean - v_star) < ci * 2


def test_saa_convergence():
    val, u = saa_linear_program(5000, c=1, p=2, a=5, b=15, seed=42)
    assert abs(u - 10.0) < 1.0
    assert abs(val - (-7.5)) < 0.5


def test_saa_negative_bias():
    """SAA value should be <= true optimum on average."""
    _, v_star = optimal_newsvendor_uniform(1, 2, 5, 15)
    values = [saa_linear_program(50, seed=i)[0] for i in range(50)]
    assert np.mean(values) <= v_star + 0.1
