from pathlib import Path

import numpy as np
import pandas as pd

from strategies import (
    classical_markowitz_strategy,
    equal_weight_strategy,
    robust_markowitz_strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAINING_WINDOW_DAYS = 504


def _compute_drifted_weights(target_weights, holding_period_returns):
    """Estimate portfolio weights right before the next rebalance."""
    if holding_period_returns.empty:
        return target_weights.copy()

    asset_growth = (1.0 + holding_period_returns).cumprod().iloc[-1]
    ending_values = target_weights * asset_growth
    total_value = ending_values.sum()

    if total_value <= 0:
        return target_weights.copy()

    return ending_values / total_value


def _load_backtest_inputs():
    returns_path = DATA_DIR / "returns_clean.csv"
    rebalance_path = DATA_DIR / "quarterly_rebalance_dates.csv"

    if not returns_path.exists():
        raise FileNotFoundError(f"Missing return file: {returns_path}")
    if not rebalance_path.exists():
        raise FileNotFoundError(f"Missing rebalance file: {rebalance_path}")

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    rebalance_dates = pd.read_csv(rebalance_path, parse_dates=["rebalance_date"])

    if returns.empty:
        raise ValueError("Return data file is empty.")
    if rebalance_dates.empty:
        raise ValueError("Rebalance date file is empty.")

    return returns, pd.DatetimeIndex(rebalance_dates["rebalance_date"])


def run_backtest(
    classical_gamma=10,
    robust_gamma=10,
    rho=1.0,
    cov_uncertainty=0.10,
    max_weight=0.10,
    initial_value=1.0,
):
    """Run quarterly backtests for all three strategies."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean returns and rebalance dates...")
    returns, rebalance_dates = _load_backtest_inputs()

    strategy_functions = {
        "equal_weight": lambda window: equal_weight_strategy(window),
        "classical_markowitz": lambda window: classical_markowitz_strategy(
            window, gamma=classical_gamma, max_weight=max_weight
        ),
        "robust_markowitz": lambda window: robust_markowitz_strategy(
            window,
            gamma=robust_gamma,
            rho=rho,
            cov_uncertainty=cov_uncertainty,
            max_weight=max_weight,
        ),
    }

    asset_names = returns.columns.tolist()
    first_rebalance_date = rebalance_dates[0]
    backtest_returns_index = returns.loc[first_rebalance_date:].index

    portfolio_returns = pd.DataFrame(
        0.0,
        index=backtest_returns_index,
        columns=strategy_functions.keys(),
    )
    weights_history = {
        strategy_name: pd.DataFrame(index=rebalance_dates, columns=asset_names, dtype=float)
        for strategy_name in strategy_functions
    }
    turnover_history = pd.DataFrame(
        np.nan,
        index=rebalance_dates,
        columns=strategy_functions.keys(),
    )

    ending_weights_by_strategy = {strategy_name: None for strategy_name in strategy_functions}

    print(f"Running backtest over {len(rebalance_dates)} quarterly rebalances...")

    for rebalance_number, rebalance_date in enumerate(rebalance_dates, start=1):
        print(
            f"Processing rebalance {rebalance_number}/{len(rebalance_dates)} on "
            f"{rebalance_date.date()}..."
        )

        training_window = returns.loc[returns.index < rebalance_date].tail(TRAINING_WINDOW_DAYS)
        if len(training_window) < TRAINING_WINDOW_DAYS:
            print("  Skipping this rebalance because there is not enough training data.")
            continue

        if rebalance_number < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[rebalance_number]
            holding_period_returns = returns.loc[
                (returns.index >= rebalance_date) & (returns.index < next_rebalance_date)
            ]
        else:
            holding_period_returns = returns.loc[returns.index >= rebalance_date]

        if holding_period_returns.empty:
            print("  No holding-period returns were found for this rebalance.")
            continue

        for strategy_name, strategy_function in strategy_functions.items():
            weights = strategy_function(training_window).reindex(asset_names).fillna(0.0)
            weights_history[strategy_name].loc[rebalance_date] = weights.values

            previous_ending_weights = ending_weights_by_strategy[strategy_name]
            if previous_ending_weights is not None:
                turnover = 0.5 * np.abs(weights - previous_ending_weights).sum()
                turnover_history.loc[rebalance_date, strategy_name] = turnover

            strategy_daily_returns = holding_period_returns.mul(weights, axis=1).sum(axis=1)
            portfolio_returns.loc[strategy_daily_returns.index, strategy_name] = strategy_daily_returns

            ending_weights_by_strategy[strategy_name] = _compute_drifted_weights(
                weights, holding_period_returns
            )

    portfolio_values = (1.0 + portfolio_returns).cumprod() * initial_value

    portfolio_returns.to_csv(OUTPUT_DIR / "portfolio_returns.csv")
    portfolio_values.to_csv(OUTPUT_DIR / "portfolio_values.csv")
    weights_history["equal_weight"].to_csv(OUTPUT_DIR / "weights_equal_weight.csv")
    weights_history["classical_markowitz"].to_csv(OUTPUT_DIR / "weights_classical_markowitz.csv")
    weights_history["robust_markowitz"].to_csv(OUTPUT_DIR / "weights_robust_markowitz.csv")
    turnover_history.to_csv(OUTPUT_DIR / "turnover.csv")

    print("Saved backtest outputs:")
    print(f"  {OUTPUT_DIR / 'portfolio_returns.csv'}")
    print(f"  {OUTPUT_DIR / 'portfolio_values.csv'}")
    print(f"  {OUTPUT_DIR / 'weights_equal_weight.csv'}")
    print(f"  {OUTPUT_DIR / 'weights_classical_markowitz.csv'}")
    print(f"  {OUTPUT_DIR / 'weights_robust_markowitz.csv'}")
    print(f"  {OUTPUT_DIR / 'turnover.csv'}")

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_values": portfolio_values,
        "weights_history": weights_history,
        "turnover_history": turnover_history,
    }


if __name__ == "__main__":
    run_backtest()
