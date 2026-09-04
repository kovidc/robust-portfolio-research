"""Expected-return estimators with one explicit annualization convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robust_portfolio.data.schemas import ReturnPanel


@dataclass(frozen=True)
class MeanForecast:
    method: str
    as_of: pd.Timestamp
    annualized_mean: pd.Series
    observations: int
    annualization_factor: int


def _validated_values(panel: ReturnPanel) -> pd.DataFrame:
    values = panel.values.astype(float)
    if len(values) < 2:
        raise ValueError("At least two observations are required for a mean estimate.")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Mean estimators require a complete finite return window.")
    return values


def _ewma_weights(length: int, half_life: float) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("EWMA half-life must be positive.")
    beta = 2.0 ** (-1.0 / float(half_life))
    weights = beta ** np.arange(length - 1, -1, -1, dtype=float)
    return weights / weights.sum()


def estimate_mean(
    panel: ReturnPanel,
    method: str,
    *,
    annualization_factor: int = 252,
    ewma_half_life: float = 63.0,
    shrinkage_intensity: float = 0.5,
) -> MeanForecast:
    """Estimate an annualized arithmetic mean using only ``panel`` rows."""
    values = _validated_values(panel)
    if annualization_factor < 1:
        raise ValueError("annualization_factor must be positive.")
    if not 0.0 <= shrinkage_intensity <= 1.0:
        raise ValueError("shrinkage_intensity must lie in [0, 1].")

    sample = values.mean(axis=0)
    if method == "sample":
        daily = sample
    elif method == "ewma":
        daily = pd.Series(
            _ewma_weights(len(values), ewma_half_life) @ values.to_numpy(),
            index=values.columns,
        )
    elif method == "shrink_zero":
        daily = (1.0 - shrinkage_intensity) * sample
    elif method == "shrink_grand_mean":
        prior = float(sample.mean())
        daily = (1.0 - shrinkage_intensity) * sample + shrinkage_intensity * prior
    else:
        raise ValueError(f"Unsupported mean estimator: {method}")

    return MeanForecast(
        method=method,
        as_of=panel.as_of,
        annualized_mean=(daily * annualization_factor).astype(float),
        observations=len(values),
        annualization_factor=annualization_factor,
    )


def bootstrap_mean(values: np.ndarray, indices: np.ndarray, annualization_factor: int) -> np.ndarray:
    """Vectorized sample means for a batch of bootstrap index rows."""
    return values[indices].mean(axis=1) * float(annualization_factor)
