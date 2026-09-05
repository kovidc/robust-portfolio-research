# Robust Portfolio Optimization

This project studies how robust mean-variance optimization changes portfolio decisions under estimation error. It compares nominal MVO with box and ellipsoidal mean uncertainty and diagonal covariance uncertainty using historical daily returns for 147 ETFs with quarterly rebalancing. The backtest is self-financing, lets holdings drift between trades, and charges linear transaction costs of 0–25 bp on traded notional.

On four dates selected in advance for the perturbation test, the robust portfolios moved substantially less under resampled inputs than nominal MVO. That stability did not consistently improve realized returns. ETF equal weight had the highest net CAGR among the eight strategies in the main comparison, while the ellipsoidal portfolio had the highest Sharpe under a zero risk-free-rate assumption.

## Results

The backtest covers April 2, 2018 through December 30, 2025, with 31 quarterly portfolio decisions. At 5 bp per dollar traded:

| Strategy | Net CAGR | Realized volatility | Sharpe (zero risk-free rate) | Recurring one-way turnover |
|---|---:|---:|---:|---:|
| ETF equal weight | 9.16% | 14.32% | 0.684 | 2.69% |
| Asset-class equal weight | 8.72% | 14.00% | 0.668 | 2.73% |
| Inverse volatility (IEWMA) | 5.77% | 9.37% | 0.646 | 7.78% |
| GMV (IEWMA) | 1.88% | 3.40% | 0.564 | 14.46% |
| Nominal MVO, 10% ceiling | 1.34% | 16.00% | 0.164 | 47.48% |
| Box robust, 10% ceiling | 0.26% | 7.61% | 0.072 | 24.89% |
| Box + diagonal, 10% ceiling | 0.99% | 6.21% | 0.190 | 22.90% |
| Ellipsoidal, 10% ceiling | 2.83% | 3.58% | 0.798 | 13.81% |

Net CAGR compounds daily net returns and annualizes them using 252 trading days per year. Sharpe is the annualized arithmetic mean of daily net returns divided by annualized daily volatility, using a zero risk-free rate. Turnover is the average half-L1 change from drifted pre-trade weights to target weights, excluding initial formation; transaction costs include initial formation.

The 10% predicted-volatility ceiling was binding on all 31 nominal decisions, but on only 6 box decisions and 14 box-plus-diagonal decisions. It was never binding for the ellipsoidal model. In a separate risk-aversion calibration, the robust optima were often already below the requested risk target. The tested grids did not contain a common range of average attained predicted risk across all four model families, so these results are not an equal-risk ranking.

Mean L1 weight sensitivity across the four perturbation dates was 1.156 for nominal MVO, 0.649 for box, 0.552 for box plus diagonal covariance, and 0.174 for ellipsoidal robustness. Each method used the same 24 block-bootstrap resamples on each date, with that date's uncertainty-set parameters held fixed. I also used synthetic ETF clones and correlation-based clustering to test how near-duplicate exposures affected the optimizer.

See the [research report](docs/FINAL_RESEARCH_REPORT.md) for the full formulations, forecasting comparisons, bootstrap intervals, and limitations. The [tables and figures](results/final/) contain the complete reported results.

## Method

All optimized portfolios are fully invested, long-only, and capped at 10% per asset:

$$\mathcal{W}=\lbrace w:\mathbf{1}^{\top}w=1,\ 0\leq w_i\leq0.10\rbrace$$

Nominal MVO maximizes $\hat{\mu}^{\top}w$ under a predicted-risk ceiling. Box robustness subtracts $\rho_{\mathrm{box}}\lVert Sw\rVert_1$, where $S=\mathrm{diag}(s)$ and $s$ contains bootstrap standard errors of annualized mean returns. Ellipsoidal robustness subtracts $\rho_{\mathrm{ell}}\sqrt{w^{\top}C_\mu w}$, allowing correlated mean-estimation errors. The box-plus-diagonal model also uses $\hat{\Sigma}+\kappa\,\mathrm{diag}(\mathrm{diag}(\hat{\Sigma}))$ in its risk constraint.

At each rebalance, expected returns and covariance are estimated from the preceding 504 daily returns. Expected returns use an arithmetic sample mean, and the main covariance estimator is IEWMA with 21-day volatility and 63-day correlation half-lives. The uncertainty parameters are calibrated from the same training window. Sample, EWMA, and Ledoit–Wolf covariance estimates are also evaluated; Ledoit–Wolf had the lowest mean covariance forecast loss in this sample.

Forecasts use only rows strictly before each decision date. Existing holdings earn the return ending on that date, target weights execute at that close, and the new holdings first earn the following return. Transaction costs are deducted from NAV using actual risky-asset dollar trades.

## Reproduction

The recorded experiments used Python 3.13.7. From the repository root:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_core_experiment.py
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python scripts/run_final_analysis.py
```

Both commands use the stored CSV data. Generated runs are written under `artifacts/`; the committed results are in `results/final/`. Run the core experiment first because the final analysis reads its outputs.

The repository includes unit and regression tests. The final-analysis tests require the core outputs above:

```bash
venv/bin/python -m unittest discover -s tests/research_foundation -p 'test_*.py' -v
venv/bin/python -m unittest discover -s tests/core_experiment -p 'test_*.py' -v
MPLCONFIGDIR=/tmp/robust_portfolio_mpl venv/bin/python -m unittest discover -s tests/final_analysis -p 'test_*.py' -v
```

## Repository

- `configs/`: experiment settings, seeds, solver choices, and cost scenarios.
- `data/`: stored prices, returns, ETF metadata, and rebalance dates. `src/download_data.py` records the data-preparation procedure; downloading the data again may produce different historical files.
- `src/robust_portfolio/`: estimators, optimizers, accounting, experiments, inference, and reporting.
- `scripts/`: entry points for the core experiment and final analysis.
- `tests/`: unit and regression tests.
- `results/final/`: reported tables, figures, and run manifest.
- `docs/FINAL_RESEARCH_REPORT.md`: detailed methods and results.

## Limitations

The ETF panel is survivor-conditioned, with static asset-class labels and no inactive-fund history. The same historical period was used during development, so it should not be treated as an untouched holdout. Same-close execution is an approximation, and the transaction-cost rates are scenarios rather than historical ETF-specific execution estimates. Cash earns zero, and Sharpe uses a zero risk-free rate because a validated full-period risk-free series was not used. The sensitivity and clone experiments cover four selected dates, and the optimized strategies end up at different risk levels.
