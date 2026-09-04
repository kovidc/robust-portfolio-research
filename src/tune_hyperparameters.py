from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import TRAINING_WINDOW_DAYS, _compute_drifted_weights, _load_backtest_inputs
from strategies import classical_markowitz_strategy, robust_markowitz_strategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRADING_DAYS_PER_YEAR = 252

CLASSICAL_GAMMA_GRID = [3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
ROBUST_GAMMA_GRID = [5.0, 10.0, 15.0, 20.0, 30.0]
ROBUST_RHO_GRID = [0.25, 0.50, 0.75, 1.0, 1.5]
ROBUST_COV_UNCERTAINTY_GRID = [0.05, 0.10, 0.15, 0.20]
MAX_WEIGHT = 0.10


def _compute_metrics(strategy_returns, turnover_values):
    cumulative_growth = (1.0 + strategy_returns).prod()
    annualized_return = cumulative_growth ** (TRADING_DAYS_PER_YEAR / len(strategy_returns)) - 1.0
    annualized_volatility = strategy_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan

    strategy_values = (1.0 + strategy_returns).cumprod()
    running_max = strategy_values.cummax()
    max_drawdown = (strategy_values / running_max - 1.0).min()

    return {
        "cumulative_return": cumulative_growth - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "average_turnover": np.nanmean(turnover_values) if len(turnover_values) else np.nan,
        "final_portfolio_value": strategy_values.iloc[-1],
    }


def _simulate_strategy(returns, rebalance_dates, strategy_function):
    first_rebalance_date = rebalance_dates[0]
    backtest_returns_index = returns.loc[first_rebalance_date:].index
    portfolio_returns = pd.Series(0.0, index=backtest_returns_index, dtype=float)

    ending_weights = None
    turnover_values = []

    for rebalance_number, rebalance_date in enumerate(rebalance_dates, start=1):
        training_window = returns.loc[returns.index < rebalance_date].tail(TRAINING_WINDOW_DAYS)
        if len(training_window) < TRAINING_WINDOW_DAYS:
            continue

        if rebalance_number < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[rebalance_number]
            holding_period_returns = returns.loc[
                (returns.index >= rebalance_date) & (returns.index < next_rebalance_date)
            ]
        else:
            holding_period_returns = returns.loc[returns.index >= rebalance_date]

        if holding_period_returns.empty:
            continue

        weights = strategy_function(training_window).reindex(returns.columns).fillna(0.0)

        if ending_weights is not None:
            turnover_values.append(0.5 * np.abs(weights - ending_weights).sum())

        strategy_daily_returns = holding_period_returns.mul(weights, axis=1).sum(axis=1)
        portfolio_returns.loc[strategy_daily_returns.index] = strategy_daily_returns
        ending_weights = _compute_drifted_weights(weights, holding_period_returns)

    return _compute_metrics(portfolio_returns, turnover_values)


def tune_hyperparameters():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns, rebalance_dates = _load_backtest_inputs()

    records = []

    print("Tuning Classical Markowitz gamma...")
    for gamma in CLASSICAL_GAMMA_GRID:
        print(f"  Classical gamma={gamma}")
        metrics = _simulate_strategy(
            returns,
            rebalance_dates,
            lambda window, gamma=gamma: classical_markowitz_strategy(
                window,
                gamma=gamma,
                max_weight=MAX_WEIGHT,
            ),
        )
        records.append({"strategy": "classical_markowitz", "gamma": gamma, **metrics})

    print("Tuning Robust Markowitz gamma/rho/covariance uncertainty...")
    robust_combinations = list(
        product(ROBUST_GAMMA_GRID, ROBUST_RHO_GRID, ROBUST_COV_UNCERTAINTY_GRID)
    )
    for combo_number, (gamma, rho, cov_uncertainty) in enumerate(robust_combinations, start=1):
        print(
            "  Robust combo "
            f"{combo_number}/{len(robust_combinations)}: "
            f"gamma={gamma}, rho={rho}, cov_uncertainty={cov_uncertainty}"
        )
        metrics = _simulate_strategy(
            returns,
            rebalance_dates,
            lambda window, gamma=gamma, rho=rho, cov_uncertainty=cov_uncertainty: (
                robust_markowitz_strategy(
                    window,
                    gamma=gamma,
                    rho=rho,
                    cov_uncertainty=cov_uncertainty,
                    max_weight=MAX_WEIGHT,
                )
            ),
        )
        records.append(
            {
                "strategy": "robust_markowitz",
                "gamma": gamma,
                "rho": rho,
                "cov_uncertainty": cov_uncertainty,
                **metrics,
            }
        )

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_DIR / "hyperparameter_tuning.csv", index=False)

    classical_results = (
        results[results["strategy"] == "classical_markowitz"]
        .sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False])
    )
    robust_results = (
        results[results["strategy"] == "robust_markowitz"]
        .sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False])
    )

    print()
    print("Best Classical Markowitz settings:")
    print(classical_results.head(5).round(4).to_string(index=False))

    print()
    print("Best Robust Markowitz settings:")
    print(robust_results.head(5).round(4).to_string(index=False))

    print()
    print(f"Saved full tuning results to {OUTPUT_DIR / 'hyperparameter_tuning.csv'}")

    return classical_results, robust_results


if __name__ == "__main__":
    tune_hyperparameters()
