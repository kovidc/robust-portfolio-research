from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRADING_DAYS_PER_YEAR = 252


def _max_drawdown(portfolio_values):
    running_max = portfolio_values.cummax()
    drawdown_series = portfolio_values / running_max - 1.0
    return drawdown_series.min()


def evaluate_performance():
    """Calculate summary performance metrics for each strategy."""
    returns_path = OUTPUT_DIR / "portfolio_returns.csv"
    values_path = OUTPUT_DIR / "portfolio_values.csv"
    turnover_path = OUTPUT_DIR / "turnover.csv"
    metrics_path = OUTPUT_DIR / "performance_metrics.csv"

    if not returns_path.exists() or not values_path.exists() or not turnover_path.exists():
        raise FileNotFoundError("Run the backtest before evaluating performance.")

    print("Evaluating backtest performance...")
    portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    portfolio_values = pd.read_csv(values_path, index_col=0, parse_dates=True)
    turnover = pd.read_csv(turnover_path, index_col=0, parse_dates=True)

    metrics = []

    for strategy_name in portfolio_returns.columns:
        strategy_returns = portfolio_returns[strategy_name].dropna()
        strategy_values = portfolio_values[strategy_name].dropna()

        if strategy_returns.empty or strategy_values.empty:
            print(f"Skipping performance metrics for {strategy_name} because no data was found.")
            continue

        cumulative_growth = (1.0 + strategy_returns).prod()
        cumulative_return = cumulative_growth - 1.0
        annualized_return = cumulative_growth ** (TRADING_DAYS_PER_YEAR / len(strategy_returns)) - 1.0
        annualized_volatility = strategy_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_ratio = (
            annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
        )
        max_drawdown = _max_drawdown(strategy_values)
        average_turnover = turnover[strategy_name].dropna().mean()
        final_portfolio_value = strategy_values.iloc[-1]

        metrics.append(
            {
                "strategy": strategy_name,
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "average_turnover": average_turnover,
                "final_portfolio_value": final_portfolio_value,
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("strategy")
    metrics_df.to_csv(metrics_path)

    print("Saved performance metrics:")
    print(f"  {metrics_path}")
    print()
    print(metrics_df.round(4))

    return metrics_df


if __name__ == "__main__":
    evaluate_performance()
