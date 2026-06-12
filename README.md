# Stochastic Optimization

Implementations of stochastic optimization methods I worked on during my studies and professional projects. The focus is on practical applications in energy and finance.

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01 - Newsvendor](notebooks/01_newsvendor_problem.ipynb) | Classical newsvendor, SAA, Monte Carlo |
| [02 - Multi-stage inventory](notebooks/02_multi_stage_inventory.ipynb) | Flower girl problem, extensive formulation, bounds |
| [03 - Dynamic programming](notebooks/03_dynamic_programming.ipynb) | Bellman recursion, policy computation |
| [04 - Unit commitment](notebooks/04_stochastic_unit_commitment.ipynb) | Two-stage stochastic LP, VSS, EVPI |
| [05 - CVaR portfolio](notebooks/05_portfolio_optimization.ipynb) | Mean-CVaR optimization, efficient frontier |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## References

- Birge & Louveaux, *Introduction to Stochastic Programming* (2011)
- Shapiro, Dentcheva & Ruszczyński, *Lectures on Stochastic Programming* (2021)
