# Stochastic Optimization Portfolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A collection of **stochastic optimization** projects demonstrating techniques widely used in **energy trading, finance, supply chain, and operations research**. 

---

## List Topics

| # | Notebook | Key Techniques |
|---|----------|---------------|
| 1 | [Newsvendor Problem](notebooks/01_newsvendor_problem.ipynb) | Sample Average Approximation (SAA), Monte Carlo simulation, confidence intervals |
| 2 | [Multi-Stage Inventory (Flower Girl)](notebooks/02_multi_stage_inventory.ipynb) | Multi-stage stochastic programming, extensive formulation, anticipative & two-stage bounds |
| 3 | [Dynamic Programming for Inventory Control](notebooks/03_dynamic_programming.ipynb) | Bellman equation, backward induction, optimal policy extraction |
| 4 | [Stochastic Unit Commitment (Energy)](notebooks/04_stochastic_unit_commitment.ipynb) | Two-stage stochastic LP, scenario generation, Value of Stochastic Solution (VSS) |
| 5 | [Portfolio Optimization under Uncertainty](notebooks/05_portfolio_optimization.ipynb) | CVaR optimization, mean-risk trade-off, scenario-based portfolio |

---

## Background

### Sample Average Approximation (SAA)
Given a stochastic program $\min_{x \in X} \mathbb{E}[f(x, \xi)]$, SAA replaces the expectation with an empirical average over $N$ samples:

$$\hat{v}_N = \min_{x \in X} \frac{1}{N} \sum_{i=1}^{N} f(x, \xi_i)$$

### Multi-Stage Stochastic Programming
For decisions made sequentially over time with evolving uncertainty:

$$\min_{x_1} c_1^T x_1 + \mathbb{E}\left[\min_{x_2(\xi_1)} c_2^T x_2 + \mathbb{E}\left[\cdots\right]\right]$$

subject to non-anticipativity constraints ensuring decisions only use available information.

### Dynamic Programming (Bellman Equation)
$$V_t(x) = \min_{u \in U(x)} \left\{ c(x, u) + \mathbb{E}\left[V_{t+1}(f(x, u, \xi))\right] \right\}$$

---

## References

- Birge, J.R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*. Springer.
- Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2021). *Lectures on Stochastic Programming*. SIAM.
- Kall, P. & Wallace, S.W. (1994). *Stochastic Programming*. Wiley.
