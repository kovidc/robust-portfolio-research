"""Pure accounting operations for drift and turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .state import ACCOUNTING_TOLERANCE, PortfolioState


def apply_close_to_close_return(
    state: PortfolioState,
    asset_returns: pd.Series,
    *,
    return_date,
    cash_return: float = 0.0,
) -> PortfolioState:
    """Let existing holdings earn the return ending at return_date."""
    if cash_return <= -1.0 or not np.isfinite(cash_return):
        raise ValueError("cash_return must be finite and greater than -1.")
    aligned_returns = asset_returns.reindex(state.holdings.index)
    held = state.holdings > ACCOUNTING_TOLERANCE
    missing_held = aligned_returns.index[held & aligned_returns.isna()].tolist()
    if missing_held:
        raise ValueError(f"Missing return for held assets: {missing_held}")
    aligned_returns = aligned_returns.fillna(0.0).astype(float)
    if not np.isfinite(aligned_returns.to_numpy()).all():
        raise ValueError("Asset returns must be finite for held assets.")
    if bool((aligned_returns[held] < -1.0).any()):
        raise ValueError("An asset return cannot be less than -100%.")

    holdings = state.holdings * (1.0 + aligned_returns)
    cash = state.cash * (1.0 + cash_return)
    nav = float(holdings.sum() + cash)
    return PortfolioState(
        timestamp=pd.Timestamp(return_date),
        nav=nav,
        holdings=holdings,
        cash=float(cash),
    )


def turnover_from_weights(
    pre_trade_weights: pd.Series,
    target_weights: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Return gross L1 fraction, one-way turnover, and weight trade vector."""
    assets = pre_trade_weights.index.union(target_weights.index, sort=False)
    pre = pre_trade_weights.reindex(assets, fill_value=0.0).astype(float)
    target = target_weights.reindex(assets, fill_value=0.0).astype(float)
    weight_trades = target - pre
    gross = float(np.abs(weight_trades).sum())
    return gross, 0.5 * gross, weight_trades
