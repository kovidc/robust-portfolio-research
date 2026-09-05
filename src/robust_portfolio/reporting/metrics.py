"""Core experiment metrics with explicit provisional zero-risk-free labeling."""

from __future__ import annotations

import numpy as np


def maximum_drawdown(wealth) -> float:
    return float((wealth / wealth.cummax() - 1.0).min())


def annualized_cagr(returns, annualization_factor: int) -> float:
    """Compound daily returns and express growth per annualization_factor rows."""
    growth = float((1.0 + returns).prod())
    return growth ** (annualization_factor / len(returns)) - 1.0


def scenario_metrics(path, targets, *, annualization_factor: int) -> dict:
    gross_returns = path.daily["gross_return"]
    net_returns = path.daily["net_return"]
    net_volatility = float(net_returns.std(ddof=1) * np.sqrt(annualization_factor))
    provisional_sharpe = (
        float(net_returns.mean() * annualization_factor / net_volatility)
        if net_volatility > 0.0
        else np.nan
    )
    recurring = [item for item in path.net_executions if not item.initial_formation]
    initial = [item for item in path.net_executions if item.initial_formation]
    effective = [1.0 / float((weights**2).sum()) for weights in targets.values()]
    recurring_one_way = (
        float(np.mean([x.one_way_turnover for x in recurring])) if recurring else np.nan
    )
    recurring_gross = (
        float(np.mean([x.gross_traded_fraction for x in recurring])) if recurring else np.nan
    )
    return {
        "gross_final_wealth": float(path.daily["gross_wealth"].iloc[-1]),
        "net_final_wealth": float(path.daily["net_wealth"].iloc[-1]),
        "gross_cumulative_return": float(path.daily["gross_wealth"].iloc[-1] - 1.0),
        "net_cumulative_return": float(path.daily["net_wealth"].iloc[-1] - 1.0),
        # Keep CSV field names stable; both annualized-return fields contain CAGR.
        "gross_annualized_return": annualized_cagr(gross_returns, annualization_factor),
        "net_annualized_return": annualized_cagr(net_returns, annualization_factor),
        "realized_volatility": net_volatility,
        "provisional_zero_rf_sharpe": provisional_sharpe,
        "max_drawdown": maximum_drawdown(path.daily["net_wealth"]),
        "recurring_one_way_turnover": recurring_one_way,
        "recurring_gross_traded_fraction": recurring_gross,
        "total_gross_traded_fraction": float(sum(x.gross_traded_fraction for x in path.net_executions)),
        "initial_one_way_turnover": float(initial[0].one_way_turnover),
        "total_transaction_cost": float(path.daily["transaction_cost"].sum()),
        "cost_drag_final_wealth": float(
            path.daily["gross_wealth"].iloc[-1] - path.daily["net_wealth"].iloc[-1]
        ),
        "average_effective_holdings": float(np.mean(effective)),
    }
