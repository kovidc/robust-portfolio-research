"""Covariance forecasts evaluated as predictions, not selected by portfolio P&L."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robust_portfolio.data.schemas import ReturnPanel


@dataclass(frozen=True)
class CovarianceForecast:
    method: str
    as_of: pd.Timestamp
    annualized_covariance: pd.DataFrame
    observations: int
    annualization_factor: int
    ridge_added: float
    minimum_eigenvalue_before: float
    shrinkage_intensity: float | None = None


def nearest_psd(
    matrix: np.ndarray,
    *,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Symmetrize and floor eigenvalues, returning matrix, ridge, and raw minimum."""
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    raw_minimum = float(eigenvalues.min())
    scale = max(float(np.trace(symmetric)) / len(symmetric), 0.0)
    floor = max(float(absolute_floor), float(relative_floor) * scale)
    clipped = np.maximum(eigenvalues, floor)
    adjusted = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    adjusted = (adjusted + adjusted.T) / 2.0
    return adjusted, max(0.0, floor - raw_minimum), raw_minimum


def _values(panel: ReturnPanel) -> pd.DataFrame:
    values = panel.values.astype(float)
    if len(values) < 2:
        raise ValueError("At least two observations are required for covariance estimation.")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Covariance estimators require a complete finite return window.")
    return values


def sample_covariance(values: np.ndarray) -> np.ndarray:
    return np.cov(values, rowvar=False, ddof=1)


def ewma_covariance(values: np.ndarray, half_life: float) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("EWMA half-life must be positive.")
    length = len(values)
    beta = 2.0 ** (-1.0 / float(half_life))
    weights = beta ** np.arange(length - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    mean = weights @ values
    centered = values - mean
    return (centered * weights[:, None]).T @ centered


def iewma_covariance(
    values: np.ndarray,
    *,
    volatility_half_life: float,
    correlation_half_life: float,
    winsorize_clip: float,
    variance_floor: float = 1e-14,
) -> np.ndarray:
    """IEWMA recursion whose standardized row uses the prior volatility forecast."""
    if volatility_half_life <= 0 or correlation_half_life <= 0:
        raise ValueError("IEWMA half-lives must be positive.")
    if winsorize_clip <= 0:
        raise ValueError("winsorize_clip must be positive.")
    x = np.asarray(values, dtype=float)
    variance = np.maximum(x[0] ** 2, variance_floor)
    standardized_outer = np.eye(x.shape[1], dtype=float)
    beta_vol = 2.0 ** (-1.0 / float(volatility_half_life))
    beta_cor = 2.0 ** (-1.0 / float(correlation_half_life))

    for row in x[1:]:
        prior_sigma = np.sqrt(np.maximum(variance, variance_floor))
        standardized = np.clip(row / prior_sigma, -winsorize_clip, winsorize_clip)
        standardized_outer = (
            beta_cor * standardized_outer
            + (1.0 - beta_cor) * np.outer(standardized, standardized)
        )
        variance = beta_vol * variance + (1.0 - beta_vol) * row**2

    diagonal = np.sqrt(np.maximum(np.diag(standardized_outer), variance_floor))
    correlation = standardized_outer / np.outer(diagonal, diagonal)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    sigma = np.sqrt(np.maximum(variance, variance_floor))
    return np.outer(sigma, sigma) * correlation


def iewma_diagonal_batch(
    samples: np.ndarray,
    *,
    volatility_half_life: float,
    variance_floor: float = 1e-14,
) -> np.ndarray:
    """Final IEWMA variance diagonals for bootstrap paths shaped (B, T, N)."""
    beta = 2.0 ** (-1.0 / float(volatility_half_life))
    variance = np.maximum(samples[:, 0, :] ** 2, variance_floor)
    for position in range(1, samples.shape[1]):
        variance = beta * variance + (1.0 - beta) * samples[:, position, :] ** 2
    return variance


def ledoit_wolf_covariance(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf linear shrinkage toward a scaled identity target."""
    x = np.asarray(values, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    observations, assets = x.shape
    empirical = x.T @ x / observations
    empirical_trace = np.diag(empirical)
    mu = float(empirical_trace.sum() / assets)
    x_squared = x**2
    beta_term = float((x_squared.T @ x_squared).sum())
    delta_term = float(((x.T @ x) ** 2).sum())
    beta = (beta_term / observations - delta_term / observations**2) / (assets * observations)
    delta = (
        delta_term / observations**2
        - 2.0 * mu * float(empirical_trace.sum())
        + assets * mu**2
    ) / assets
    beta = min(max(beta, 0.0), max(delta, 0.0))
    shrinkage = 0.0 if delta <= 0.0 else beta / delta
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: assets + 1] += shrinkage * mu
    return covariance, float(shrinkage)


def estimate_covariance(
    panel: ReturnPanel,
    method: str,
    *,
    annualization_factor: int = 252,
    ewma_half_life: float = 63.0,
    iewma_volatility_half_life: float = 21.0,
    iewma_correlation_half_life: float = 63.0,
    iewma_winsorize_clip: float = 4.2,
    iewma_variance_floor: float = 1e-14,
    absolute_eigenvalue_floor: float = 1e-10,
    relative_eigenvalue_floor: float = 1e-8,
) -> CovarianceForecast:
    values = _values(panel)
    x = values.to_numpy()
    shrinkage = None
    if method == "sample":
        daily = sample_covariance(x)
    elif method == "ewma":
        daily = ewma_covariance(x, ewma_half_life)
    elif method == "iewma":
        daily = iewma_covariance(
            x,
            volatility_half_life=iewma_volatility_half_life,
            correlation_half_life=iewma_correlation_half_life,
            winsorize_clip=iewma_winsorize_clip,
            variance_floor=iewma_variance_floor,
        )
    elif method == "ledoit_wolf":
        daily, shrinkage = ledoit_wolf_covariance(x)
    else:
        raise ValueError(f"Unsupported covariance estimator: {method}")

    annualized = daily * float(annualization_factor)
    psd, ridge, raw_minimum = nearest_psd(
        annualized,
        absolute_floor=absolute_eigenvalue_floor,
        relative_floor=relative_eigenvalue_floor,
    )
    return CovarianceForecast(
        method=method,
        as_of=panel.as_of,
        annualized_covariance=pd.DataFrame(psd, index=values.columns, columns=values.columns),
        observations=len(values),
        annualization_factor=annualization_factor,
        ridge_added=ridge,
        minimum_eigenvalue_before=raw_minimum,
        shrinkage_intensity=shrinkage,
    )
