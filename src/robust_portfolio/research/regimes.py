"""As-of market trend and volatility regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regimes(
    market_returns: pd.Series,
    rebalance_dates,
    *,
    trend_lookback: int,
    volatility_lookback: int,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    series = market_returns.sort_index().astype(float)
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("Regime classification requires finite market returns.")
    trailing_volatility = series.rolling(volatility_lookback).std(ddof=1) * np.sqrt(
        annualization_factor
    )
    records = []
    for date in pd.DatetimeIndex(rebalance_dates):
        history = series.loc[series.index < date]
        if len(history) < max(trend_lookback, volatility_lookback):
            raise ValueError(f"Insufficient as-of regime history at {date.date()}.")
        trend_return = float((1.0 + history.iloc[-trend_lookback:]).prod() - 1.0)
        current_volatility = float(history.iloc[-volatility_lookback:].std(ddof=1) * np.sqrt(annualization_factor))
        prior_volatility = trailing_volatility.loc[trailing_volatility.index < date].dropna()
        threshold = float(prior_volatility.median())
        positive = trend_return > 0.0
        high = current_volatility > threshold
        if positive and not high:
            label = "calm risk-on"
        elif positive and high:
            label = "volatile risk-on"
        elif not positive and not high:
            label = "weak/cooling"
        else:
            label = "stress/risk-off"
        records.append(
            {
                "decision_date": date,
                "trend_trailing_return": trend_return,
                "current_trailing_volatility": current_volatility,
                "expanding_median_volatility_threshold": threshold,
                "trend_state": "positive" if positive else "negative",
                "volatility_state": "high" if high else "low",
                "regime": label,
                "information_end": history.index[-1],
            }
        )
    return pd.DataFrame(records)
