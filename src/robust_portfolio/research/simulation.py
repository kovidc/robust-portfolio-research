"""Core experiment simulation using the research foundation accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robust_portfolio.backtest.costs import LinearCostModel, ZeroCostModel
from robust_portfolio.backtest.execution import ExecutionResult, execute_target
from robust_portfolio.backtest.state import PortfolioState
from robust_portfolio.data.providers import FrozenCsvReturnProvider


@dataclass(frozen=True)
class ScenarioPath:
    strategy: str
    cost_bps: float
    daily: pd.DataFrame
    gross_executions: tuple[ExecutionResult, ...]
    net_executions: tuple[ExecutionResult, ...]


def simulate_targets(
    provider: FrozenCsvReturnProvider,
    target_by_date: dict[pd.Timestamp, pd.Series],
    *,
    strategy: str,
    cost_bps: float,
    maximum_weight: float,
    cash_daily_return: float = 0.0,
    market_returns: pd.DataFrame | None = None,
) -> ScenarioPath:
    if not target_by_date:
        raise ValueError("A strategy must supply at least one target.")
    schedule = pd.DatetimeIndex(sorted(pd.Timestamp(date) for date in target_by_date))
    simulation_dates = provider.dates[provider.dates >= schedule[0]]
    timestamp = schedule[0] - pd.Timedelta(nanoseconds=1)
    gross_state = PortfolioState.all_cash(timestamp, 1.0, provider.assets)
    net_state = PortfolioState.all_cash(timestamp, 1.0, provider.assets)
    gross_holdings = gross_state.holdings.to_numpy(copy=True)
    net_holdings = net_state.holdings.to_numpy(copy=True)
    gross_cash = gross_state.cash
    net_cash = net_state.cash
    gross_executions = []
    net_executions = []
    records = []
    rate = float(cost_bps) / 10_000.0
    schedule_set = set(schedule)

    if market_returns is None:
        return_matrix = np.vstack(
            [provider.return_ending_at(date).reindex(provider.assets).to_numpy() for date in simulation_dates]
        )
    else:
        return_matrix = market_returns.reindex(index=simulation_dates, columns=provider.assets).to_numpy()
    if not np.isfinite(return_matrix).all():
        raise ValueError("Core experiment simulation requires complete finite held-asset returns.")

    for row_number, date in enumerate(simulation_dates):
        gross_start = float(gross_holdings.sum() + gross_cash)
        net_start = float(net_holdings.sum() + net_cash)
        growth = 1.0 + return_matrix[row_number]
        gross_holdings *= growth
        net_holdings *= growth
        gross_cash *= 1.0 + cash_daily_return
        net_cash *= 1.0 + cash_daily_return
        gross_nav = float(gross_holdings.sum() + gross_cash)
        net_nav = float(net_holdings.sum() + net_cash)
        cost = 0.0
        if date in schedule_set:
            gross_state = PortfolioState(
                timestamp=date,
                nav=gross_nav,
                holdings=pd.Series(gross_holdings, index=provider.assets),
                cash=gross_cash,
            )
            net_state = PortfolioState(
                timestamp=date,
                nav=net_nav,
                holdings=pd.Series(net_holdings, index=provider.assets),
                cash=net_cash,
            )
            target = target_by_date[pd.Timestamp(date)]
            gross_execution = execute_target(
                gross_state,
                target,
                execution_date=date,
                cost_model=ZeroCostModel(),
                maximum_weight=maximum_weight,
                initial_formation=not gross_executions,
            )
            net_execution = execute_target(
                net_state,
                target,
                execution_date=date,
                cost_model=LinearCostModel(rate),
                maximum_weight=maximum_weight,
                initial_formation=not net_executions,
            )
            gross_state = gross_execution.post_trade_state
            net_state = net_execution.post_trade_state
            gross_holdings = gross_state.holdings.to_numpy(copy=True)
            net_holdings = net_state.holdings.to_numpy(copy=True)
            gross_cash = gross_state.cash
            net_cash = net_state.cash
            gross_nav = gross_state.nav
            net_nav = net_state.nav
            cost = net_execution.transaction_cost
            gross_executions.append(gross_execution)
            net_executions.append(net_execution)
        records.append(
            {
                "date": date,
                "gross_wealth": gross_nav,
                "net_wealth": net_nav,
                "gross_return": gross_nav / gross_start - 1.0,
                "net_return": net_nav / net_start - 1.0,
                "transaction_cost": cost,
            }
        )
    return ScenarioPath(
        strategy=strategy,
        cost_bps=float(cost_bps),
        daily=pd.DataFrame(records).set_index("date"),
        gross_executions=tuple(gross_executions),
        net_executions=tuple(net_executions),
    )
