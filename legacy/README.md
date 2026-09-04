# Historical CS361 Baseline

This directory contains the configuration and reproduction metadata for the original CS361 experiment. The original code, data, and stored outputs remain in their existing locations in the repository. This legacy path is kept so the original results can be reproduced and compared with the rebuilt research pipeline.

The results reproduced here are the original class-project results, not the corrected results reported elsewhere in the repository.

## Reproducing the original experiment

From the repository root:

```bash
venv/bin/python scripts/reproduce_legacy.py
```

The script verifies the cached inputs and legacy source before running, does not download new market data, and writes reproduced outputs to:

```text
artifacts/legacy_cs361/reproduced/
```

The reproduction includes:

- `legacy_metrics.json`: reproduced metrics and comparison with the stored baseline
- `accounting_contradiction.json`: shows the mismatch between the original daily return calculation and quarterly turnover calculation
- `legacy_run.json`: runtime information, hash checks, solver information, and output paths

The repository's historical outputs are stored in [`../outputs/`](../outputs/) and are not overwritten by the reproduction script.

## Original methodology

[`config.json`](config.json) records the settings used by the original experiment, including the ETF universe, estimation window, rebalance dates, covariance estimators, optimizer parameters, constraints, and accounting assumptions.

[`baseline_manifest.json`](baseline_manifest.json) contains the hashes, environment information, expected metrics, and numerical tolerances used to verify reproduction.

## Known issues in the original project

The original experiment has several methodological problems that are corrected in the rebuilt version:

- Portfolio returns use daily constant target weights, while turnover is calculated from quarterly drifted weights.
- The ETF universe is survivor-conditioned.
- Classical and robust Markowitz use different covariance estimators.
- Classical and robust portfolios use different risk-aversion parameters.
- Several hyperparameters were selected using the reported evaluation period.
- CM-IEWMA expert scoring reuses data involved in constructing the covariance forecasts.
- The reported Sharpe ratio uses CAGR divided by annualized volatility with a zero risk-free rate.
- Transaction costs are omitted.
- The signal and execution timing convention is not fully specified.

These choices are left unchanged so the original experiment remains reproducible. The rebuilt implementation under [`../src/robust_portfolio/`](../src/robust_portfolio/) addresses them separately.

## Tests

From the repository root:

```bash
venv/bin/python -m unittest discover -s tests/legacy -p 'test_*.py' -v
```

The tests reproduce the stored baseline, verify portfolio weights and output tables within numerical tolerances, check optimizer behavior, and confirm that the original accounting inconsistency remains detectable.
