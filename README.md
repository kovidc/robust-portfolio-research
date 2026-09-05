# Robust Portfolio Optimization

This project studies when robust mean-variance optimization changes portfolio decisions in useful ways. It compares nominal MVO, box and ellipsoidal mean uncertainty, and diagonal covariance uncertainty on historical adjusted returns for 147 ETFs, with quarterly rebalancing. The self-financing backtest lets holdings drift between trades and charges linear transaction costs of 0–25 bp per dollar traded.

On four prespecified perturbation dates, allocations from robust objectives were substantially less sensitive to resampled inputs than nominal MVO. This did not translate into consistent realized-return outperformance. ETF equal weight had the highest net CAGR among the eight headline strategies; ellipsoidal robustness had the highest zero-risk-free Sharpe, at much lower attained risk.

## Results

The evaluation runs from April 2, 2018 through December 30, 2025, with 31 quarterly decisions. At 5 bp per dollar traded:

| Strategy | Net CAGR | Realized volatility | Sharpe (zero RF) | Recurring one-way turnover |
|---|---:|---:|---:|---:|
| ETF equal weight | 9.16% | 14.32% | 0.684 | 2.69% |
| Asset-class equal weight | 8.72% | 14.00% | 0.668 | 2.73% |
| Inverse volatility (IEWMA) | 5.77% | 9.37% | 0.646 | 7.78% |
| GMV (IEWMA) | 1.88% | 3.40% | 0.564 | 14.46% |
| Nominal MVO, 10% ceiling | 1.34% | 16.00% | 0.164 | 47.48% |
| Box robust, 10% ceiling | 0.26% | 7.61% | 0.072 | 24.89% |
| Box + diagonal, 10% ceiling | 0.99% | 6.21% | 0.190 | 22.90% |
| Ellipsoidal, 10% ceiling | 2.83% | 3.58% | 0.798 | 13.81% |

CAGR compounds daily net returns and annualizes using 252 observations per year. Sharpe divides the annualized arithmetic daily mean by annualized daily volatility, assuming a zero risk-free rate. Turnover is the average half-L1 change from drifted pre-trade weights to target weights, excluding initial formation; costs also include initial formation.

The 10% predicted-volatility ceiling was binding on all 31 nominal decisions, but on only 6 box decisions and 14 box-plus-diagonal decisions; it was never binding for the ellipsoidal model. In the separate risk-aversion calibration, the robust optima were often already below the requested target. There was no common attained risk across all four model families in the tested grids, so these results do not support an all-model equal-risk performance ranking.

Mean L1 weight sensitivity on the four perturbation dates was 1.156 for nominal MVO, 0.649 for box, 0.552 for box plus diagonal covariance, and 0.174 for ellipsoidal robustness. Each date used 24 shared block-bootstrap training samples with the date's uncertainty geometry held fixed. Synthetic ETF clones and correlation clustering provide separate checks on exposure redundancy.

See the [research report](docs/FINAL_RESEARCH_REPORT.md) for formulations, forecasting comparisons, paired bootstrap intervals, and limitations. The [committed tables and figures](results/final/) contain the full reported results.

## Method

All optimized portfolios are fully invested, long-only, and capped at 10% per asset:

$$\mathcal{W}=\{w:\mathbf{1}^{\top}w=1,\ 0\leq w_i\leq0.10\}$$

Nominal MVO maximizes $\hat{\mu}^{\top}w$ under a predicted-risk ceiling. Box robustness subtracts $\rho_{\mathrm{box}}\|Sw\|_1$, where $S$ contains bootstrap standard errors. Ellipsoidal robustness subtracts $\rho_{\mathrm{ell}}\sqrt{w^{\top}C_\mu w}$, allowing correlated mean-estimation errors. The box-plus-diagonal model also uses $\hat{\Sigma}+\kappa\,\mathrm{diag}(\mathrm{diag}(\hat{\Sigma}))$ in its risk constraint.

The main estimates use the preceding 504 daily returns: an arithmetic sample mean and IEWMA covariance with 21-day volatility and 63-day correlation half-lives. Uncertainty parameters are calibrated from that training window. Sample, EWMA, and Ledoit–Wolf covariance estimates are also evaluated; Ledoit–Wolf had the lowest mean covariance forecast loss in this sample.

Forecasts use rows strictly before each decision date. Existing holdings earn the return ending on that date, targets execute at that close, and new holdings first earn the following return. Transaction costs are deducted from NAV using actual risky-asset dollar trades.

## Reproduction

The recorded experiments used Python 3.13.7. From the repository root:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_core_experiment.py
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_final_analysis.py
```

Both commands use the stored CSV data and write to configuration-hash directories under `artifacts/`. Run the core experiment first; the final analysis reads its outputs. The committed `results/final/` snapshot is separate from generated runs.

The tests use Python's standard-library `unittest`; the final-analysis suite requires the core outputs above:

```bash
venv/bin/python -m unittest discover -s tests/research_foundation -p 'test_*.py' -v
venv/bin/python -m unittest discover -s tests/core_experiment -p 'test_*.py' -v
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python -m unittest discover -s tests/final_analysis -p 'test_*.py' -v
```

## Repository

- `configs/`: experiment settings, seeds, solver choices, and cost scenarios.
- `data/`: stored prices, returns, ETF metadata, and rebalance dates. `src/download_data.py` records the data-preparation procedure; downloading again may change the historical data.
- `src/robust_portfolio/`: estimators, optimizers, accounting, experiments, inference, and reporting.
- `scripts/`: entry points for the core experiment and final analysis.
- `tests/`: test suites listed under Reproduction.
- `results/final/`: reported tables, figures, and the original run manifest.
- `docs/FINAL_RESEARCH_REPORT.md`: detailed methods and results.

## Limitations

The ETF panel is survivor-conditioned, with static asset-class labels and no inactive-fund history. The evaluation period was observed during project development, so there is no untouched holdout. Same-close execution is an approximation, and transaction-cost rates are scenarios rather than estimated historical ETF execution costs. Cash earns zero and Sharpe uses zero RF because a validated full-period risk-free series is unavailable. The direct sensitivity and clone experiments cover four selected dates, and the optimized strategies attain different risks.
