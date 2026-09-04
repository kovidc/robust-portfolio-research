# Robust Markowitz Portfolio Research

This project asks a simple question: does robust portfolio optimization actually make portfolio decisions less sensitive to estimation error?

We compare standard mean-variance optimization with several uncertainty-aware variants on a universe of 147 ETFs. The results are mixed: robust methods make portfolio weights much more stable under perturbations in expected returns and covariance estimates, but they don't consistently outperform nominal MVO on realized returns.

This repository started as a CS361 class project. We kept the original version for reference in `legacy/`, then rebuilt it with correct portfolio accounting, better controls, and more careful comparisons.

## Main takeaway

Across our perturbation experiments, robust formulations reduced input-to-weight sensitivity substantially relative to nominal MVO.

The return results are more mixed. Once we account for estimation error, transaction costs, and risk calibration, we don't find strong evidence that robust optimization generally beats nominal MVO or simple diversification out of sample.

## What's in this repository

The rebuilt research pipeline adds:

- self-financing portfolio accounting
- drifted pre-trade weights and explicit turnover
- transaction-cost scenarios
- as-of return and covariance estimates
- robust expected-return and covariance formulations
- risk-attainment checks
- bootstrap perturbation experiments
- synthetic ETF-clone experiments
- clustering and redundancy tests
- dependence-aware statistical inference
- reproducible tables and figures

The original class-project version is preserved in `legacy/`. The current research code lives under `src/robust_portfolio/`.

## Portfolio models

All optimized portfolios are:

- fully invested
- long-only
- capped at 10% per asset

In other words, weights sum to 1 and each asset weight satisfies `0 <= w_i <= 0.10`.

### Nominal mean-variance optimization

The nominal optimizer uses the estimated return vector and covariance matrix directly. Under a fixed risk constraint, it chooses weights to maximize:

```text
mu_hat^T w
```

### Robust expected returns

We test two ways of handling uncertainty in expected returns.

**Box uncertainty** uses:

```text
mu_hat^T w - rho * ||S w||_1
```

Because the portfolios are long-only, this acts like shrinking estimated returns according to their uncertainty.

**Ellipsoidal uncertainty** uses:

```text
mu_hat^T w - rho * sqrt(w^T C_mu w)
```

This lets uncertainty in return estimates be correlated across assets.

### Covariance robustness

We also test a more conservative covariance estimate:

```text
Sigma_rob = Sigma_hat + kappa * diag(diag(Sigma_hat))
```

This increases estimated individual-asset risk while preserving the estimated cross-asset covariance structure.

### Risk calibration

Risk comparability is important here!

Using the same ex-ante volatility ceiling across methods doesn't guarantee that the resulting portfolios actually take the same predicted risk. A robust optimizer may naturally sit well below the ceiling.

So we run two kinds of comparisons:

1. portfolios solved under the same ex-ante volatility ceiling
2. portfolios calibrated toward the same predicted volatility target

If a robust formulation can’t reach the target risk, we mark the comparison as unattainable.

## Data

We use adjusted returns for a balanced panel of 147 ETFs.

The universe is survivor-conditioned. It was built from funds with sufficient current and historical data, rather than from a point-in-time historical ETF universe.

Because the universe is survivor-conditioned, results apply to this fixed ETF panel rather than a fully reconstructed historical universe.
Signals use only returns available before each quarterly decision date.

At each rebalance:

- existing holdings earn the return ending on the decision date
- targets execute at that close
- new weights first earn the following return

Turnover is computed from drifted pre-trade weights rather than assuming weights stay fixed between rebalances.

## Transaction costs

We test linear transaction-cost scenarios of:

- 0 bp
- 1 bp
- 5 bp
- 10 bp
- 25 bp

Note that the cost levels are scenario assumptions rather than historical ETF-specific estimates.

## Backtesting

The backtester is self-financing.

Between quarterly rebalances, weights drift with realized returns. At the next rebalance, we compare those drifted holdings with the new target portfolio to calculate turnover and transaction costs.

This fixes one of the main issues in the original class project, where returns were effectively computed as if target weights reset every day while turnover was only charged quarterly.

## Perturbation experiments

A central question in the project is how sensitive optimized portfolios are to small changes in estimated inputs.

To test that, we generate shared perturbations of expected returns and covariance estimates and then resolve each method under the same perturbed inputs.

This lets us compare how much the resulting portfolio weights move under nominal and robust optimization.

Across these perturbations, robust methods are substantially more stable than nominal MVO.

## ETF redundancy

ETF universes often contain funds with very similar exposures. That can make optimization unstable because small changes in inputs may cause the optimizer to jump between near-substitutes.

We study that in two ways:

- clustering ETFs using only information available before each evaluation period
- creating synthetic ETF clones to test near-redundancy directly

This lets us test whether instability comes from estimation error itself or from choosing between near-duplicate ETFs.

## Statistical inference

Because portfolio returns are serially dependent, we don't treat daily observations as independent.

We use stationary bootstrap inference to account for serial dependence in returns.

We also keep the stability experiments separate from the return comparisons. Weight stability and realized performance are evaluated separately.

## Installation

Python 3.13 was used for the final experiments.

From the repository root:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Reproducing the experiments

To reproduce the original CS361 baseline:

```bash
venv/bin/python scripts/reproduce_legacy.py
```

To run the rebuilt research pipeline:

```bash
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_core_experiment.py
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_final_analysis.py
```

## Tests

The repository includes tests for the legacy reproduction, the research foundation, the optimization experiments, and the final analysis.

```bash
venv/bin/python -m unittest discover -s tests/legacy -p 'test_*.py' -v
venv/bin/python -m unittest discover -s tests/research_foundation -p 'test_*.py' -v
venv/bin/python -m unittest discover -s tests/core_experiment -p 'test_*.py' -v
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python -m unittest discover -s tests/final_analysis -p 'test_*.py' -v
```

## Repository structure

```text
legacy/                            original CS361 experiment
src/robust_portfolio/data/         data handling and as-of universe construction
src/robust_portfolio/estimators/   return, covariance, and uncertainty estimates
src/robust_portfolio/optimizers/   nominal and robust portfolio optimizers
src/robust_portfolio/backtest/     holdings, rebalancing, turnover, and costs
src/robust_portfolio/calibration/  risk calibration
src/robust_portfolio/research/     experiments and diagnostics
src/robust_portfolio/inference/    bootstrap and statistical inference
src/robust_portfolio/reporting/    metrics, tables, and figures
tests/                             test suites
docs/                              detailed methods, report, and limitations
```

For the full methodology and results, see [`docs/FINAL_RESEARCH_REPORT.md`](docs/FINAL_RESEARCH_REPORT.md).

## Limitations

The main limitations are:

- The ETF universe is survivor-conditioned rather than point-in-time.
- We don't have historical data for funds that disappeared from the current universe.
- Transaction costs are scenario assumptions rather than ETF-specific historical estimates.
- The historical evaluation period was used during project development, so it shouldn't be treated as a completely untouched final holdout.
- We don't have a validated full-period risk-free series, so Sharpe comparisons under a zero risk-free assumption should be interpreted cautiously.

Overall, robust optimization substantially reduced sensitivity to estimation error, but did not consistently improve realized returns.
