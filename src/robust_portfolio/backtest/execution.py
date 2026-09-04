"""Self-financing conversion of target weights into actual trades."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .accounting import turnover_from_weights
from .costs import CostModel, ZeroCostModel
from .state import ACCOUNTING_TOLERANCE, PortfolioState


@dataclass(frozen=True)
class ExecutionResult:
    execution_date: pd.Timestamp
    pre_trade_state: PortfolioState
    target_weights: pd.Series
    weight_trades: pd.Series
    dollar_trades: pd.Series
    gross_traded_fraction: float
    one_way_turnover: float
    transaction_cost: float
    post_trade_state: PortfolioState
    initial_formation: bool

    def __post_init__(self):
        expected_post_nav = self.pre_trade_state.nav - self.transaction_cost
        if not np.isclose(
            self.post_trade_state.nav,
            expected_post_nav,
            atol=ACCOUNTING_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("Post-trade NAV does not equal pre-trade NAV less cost.")
        cash_change = self.post_trade_state.cash - self.pre_trade_state.cash
        financing_residual = float(self.dollar_trades.sum() + cash_change + self.transaction_cost)
        if abs(financing_residual) > 5 * ACCOUNTING_TOLERANCE:
            raise ValueError("Trades, cash, and costs do not satisfy self-financing continuity.")


def _validated_target(
    target_weights: pd.Series,
    assets: pd.Index,
    maximum_weight: float | None,
) -> pd.Series:
    target = target_weights.reindex(assets, fill_value=0.0).astype(float)
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Target weights must be finite.")
    if float(target.min()) < -ACCOUNTING_TOLERANCE:
        raise ValueError("Research foundation target weights cannot be negative.")
    target = target.clip(lower=0.0)
    if float(target.sum()) > 1.0 + ACCOUNTING_TOLERANCE:
        raise ValueError("Risky-asset target weights cannot sum above one.")
    if maximum_weight is not None and float(target.max()) > maximum_weight + ACCOUNTING_TOLERANCE:
        raise ValueError("A target weight exceeds the configured maximum weight.")
    return target


def _post_cost_nav(
    pre_trade_state: PortfolioState,
    target_weights: pd.Series,
    cost_model: CostModel,
) -> float:
    """Solve V+ + cost(target * V+ - holdings-) = V-."""
    pre_nav = pre_trade_state.nav
    pre_holdings = pre_trade_state.holdings.reindex(target_weights.index, fill_value=0.0)

    def residual(post_nav: float) -> float:
        trades = target_weights * post_nav - pre_holdings
        return post_nav + cost_model.cost(trades) - pre_nav

    high_residual = residual(pre_nav)
    if abs(high_residual) <= ACCOUNTING_TOLERANCE:
        return pre_nav
    low_residual = residual(0.0)
    if low_residual > ACCOUNTING_TOLERANCE:
        raise ValueError("Transaction costs exhaust available NAV; no self-financing trade exists.")

    low, high = 0.0, pre_nav
    for _ in range(200):
        midpoint = 0.5 * (low + high)
        midpoint_residual = residual(midpoint)
        if abs(midpoint_residual) <= ACCOUNTING_TOLERANCE:
            return midpoint
        if midpoint_residual > 0:
            high = midpoint
        else:
            low = midpoint
    return 0.5 * (low + high)


def execute_target(
    pre_trade_state: PortfolioState,
    target_weights: pd.Series,
    *,
    execution_date,
    cost_model: CostModel | None = None,
    maximum_weight: float | None = None,
    initial_formation: bool = False,
) -> ExecutionResult:
    """Execute a post-cost target at close without creating wealth."""
    cost_model = cost_model or ZeroCostModel()
    assets = pre_trade_state.holdings.index.union(target_weights.index, sort=False)
    pre_state = pre_trade_state.reindex(assets)
    target = _validated_target(target_weights, assets, maximum_weight)
    pre_weights = pre_state.weights
    gross, one_way, weight_trades = turnover_from_weights(pre_weights, target)

    post_nav = _post_cost_nav(pre_state, target, cost_model)
    post_holdings = target * post_nav
    dollar_trades = post_holdings - pre_state.holdings
    transaction_cost = cost_model.cost(dollar_trades)
    post_cash = (1.0 - float(target.sum())) * post_nav
    post_state = PortfolioState(
        timestamp=pd.Timestamp(execution_date),
        nav=post_nav,
        holdings=post_holdings,
        cash=post_cash,
    )
    return ExecutionResult(
        execution_date=pd.Timestamp(execution_date),
        pre_trade_state=pre_state,
        target_weights=target,
        weight_trades=weight_trades,
        dollar_trades=dollar_trades,
        gross_traded_fraction=gross,
        one_way_turnover=one_way,
        transaction_cost=transaction_cost,
        post_trade_state=post_state,
        initial_formation=initial_formation,
    )
