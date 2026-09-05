"""Transparent simple portfolio benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _check_capacity(asset_count: int, maximum_weight: float) -> None:
    if asset_count < 1:
        raise ValueError("At least one eligible asset is required.")
    if asset_count * maximum_weight < 1.0 - 1e-12:
        raise ValueError("The maximum-weight constraint makes full investment infeasible.")


def _cap_proportional(raw: pd.Series, maximum_weight: float) -> pd.Series:
    """Normalize positive scores to the capped simplex without changing their ordering."""
    scores = raw.astype(float).clip(lower=0.0)
    _check_capacity(len(scores), maximum_weight)
    if not np.isfinite(scores.to_numpy()).all() or float(scores.sum()) <= 0.0:
        raise ValueError("Heuristic scores must be finite with a positive sum.")
    remaining = 1.0
    available = list(scores.index)
    result = pd.Series(0.0, index=scores.index)
    while available:
        weights = scores.loc[available] / scores.loc[available].sum() * remaining
        capped = weights[weights > maximum_weight]
        if capped.empty:
            result.loc[available] = weights
            break
        result.loc[capped.index] = maximum_weight
        remaining -= maximum_weight * len(capped)
        available = [asset for asset in available if asset not in capped.index]
    if abs(float(result.sum()) - 1.0) > 1e-10:
        raise RuntimeError("Capped-simplex normalization failed.")
    return result


def equal_weight(assets, maximum_weight: float = 1.0) -> pd.Series:
    index = pd.Index(list(assets))
    _check_capacity(len(index), maximum_weight)
    weights = pd.Series(1.0 / len(index), index=index, dtype=float)
    if float(weights.max()) > maximum_weight + 1e-12:
        raise ValueError("Equal weight violates the maximum-weight constraint.")
    return weights


def asset_class_equal_weight(
    assets,
    asset_classes: pd.Series,
    maximum_weight: float = 1.0,
) -> pd.Series:
    index = pd.Index(list(assets))
    _check_capacity(len(index), maximum_weight)
    classes = asset_classes.reindex(index)
    if classes.isna().any():
        raise ValueError(f"Missing asset-class labels: {classes.index[classes.isna()].tolist()}")
    class_count = int(classes.nunique())
    weights = pd.Series(0.0, index=index)
    for members in classes.groupby(classes).groups.values():
        weights.loc[list(members)] = 1.0 / (class_count * len(members))
    if float(weights.max()) > maximum_weight + 1e-12:
        raise ValueError("Asset-class equal weight violates the maximum-weight constraint.")
    return weights


def inverse_volatility(covariance: pd.DataFrame, maximum_weight: float = 1.0) -> pd.Series:
    variances = np.diag(covariance.to_numpy(dtype=float))
    if not np.isfinite(variances).all() or bool((variances <= 0.0).any()):
        raise ValueError("Inverse volatility requires finite positive variances.")
    scores = pd.Series(1.0 / np.sqrt(variances), index=covariance.index)
    return _cap_proportional(scores, maximum_weight)


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    marginal = covariance @ weights
    return weights * marginal


def risk_parity(
    covariance: pd.DataFrame,
    maximum_weight: float = 1.0,
    *,
    tolerance: float = 1e-10,
) -> pd.Series:
    """Long-only equal-risk-contribution solution, cap-constrained if needed."""
    sigma = covariance.to_numpy(dtype=float)
    assets = covariance.index
    _check_capacity(len(assets), maximum_weight)
    initial = np.repeat(1.0 / len(assets), len(assets))

    def objective(weights):
        contributions = risk_contributions(weights, sigma)
        target = contributions.sum() / len(weights)
        scale = max(abs(float(contributions.sum())), 1e-12)
        return float(np.sum(((contributions - target) / scale) ** 2))

    solution = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(1e-12, maximum_weight)] * len(assets),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": tolerance, "maxiter": 2000, "disp": False},
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError(f"Risk-parity optimization failed explicitly: {solution.message}")
    weights = pd.Series(solution.x, index=assets)
    if abs(float(weights.sum()) - 1.0) > 1e-7 or float(weights.max()) > maximum_weight + 1e-7:
        raise ValueError("Risk-parity solver returned an infeasible portfolio.")
    return weights
