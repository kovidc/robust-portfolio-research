"""Dependence-aware bootstrap calibration for robust uncertainty sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robust_portfolio.data.schemas import ReturnPanel

from .covariance import iewma_diagonal_batch, nearest_psd
from .means import bootstrap_mean


@dataclass(frozen=True)
class UncertaintyCalibration:
    as_of: pd.Timestamp
    standard_errors: pd.Series
    mean_error_covariance: pd.DataFrame
    box_rho: float
    ellipsoid_rho: float
    diagonal_kappa: float
    bootstrap_seed: int
    bootstrap_replications: int
    block_length: int
    coverage_probability: float
    mean_covariance_ridge: float


def circular_block_bootstrap_indices(
    observations: int,
    replications: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Generate fixed-length circular moving-block bootstrap paths."""
    if observations < 2 or replications < 2:
        raise ValueError("Bootstrap requires at least two rows and two replications.")
    if not 1 <= block_length <= observations:
        raise ValueError("block_length must lie between one and the sample length.")
    generator = np.random.default_rng(int(seed))
    blocks = int(np.ceil(observations / block_length))
    starts = generator.integers(0, observations, size=(replications, blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % observations
    return indices.reshape(replications, -1)[:, :observations]


def calibrate_uncertainty(
    panel: ReturnPanel,
    *,
    bootstrap_seed: int,
    bootstrap_replications: int,
    block_length: int,
    coverage_probability: float,
    annualization_factor: int = 252,
    standard_error_floor: float = 1e-10,
    relative_eigenvalue_floor: float = 1e-6,
    absolute_eigenvalue_floor: float = 1e-10,
    iewma_volatility_half_life: float = 21.0,
    iewma_variance_floor: float = 1e-14,
    batch_size: int = 25,
) -> UncertaintyCalibration:
    """Calibrate box/ellipsoid mean uncertainty and diagonal risk uncertainty."""
    if not 0.0 < coverage_probability < 1.0:
        raise ValueError("coverage_probability must lie strictly between zero and one.")
    values = panel.values.astype(float)
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Uncertainty calibration requires a complete finite window.")
    x = values.to_numpy()
    indices = circular_block_bootstrap_indices(
        len(values), bootstrap_replications, block_length, bootstrap_seed
    )
    base_mean = x.mean(axis=0) * float(annualization_factor)
    bootstrap_means = bootstrap_mean(x, indices, annualization_factor)
    errors = bootstrap_means - base_mean
    standard_errors = np.maximum(errors.std(axis=0, ddof=1), standard_error_floor)
    max_standardized = np.max(np.abs(errors) / standard_errors, axis=1)
    box_rho = float(np.quantile(max_standardized, coverage_probability))

    raw_mean_covariance = np.cov(errors, rowvar=False, ddof=1)
    mean_covariance, ridge, _ = nearest_psd(
        raw_mean_covariance,
        absolute_floor=absolute_eigenvalue_floor,
        relative_floor=relative_eigenvalue_floor,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    coordinates = errors @ eigenvectors
    mahalanobis_squared = np.sum(coordinates**2 / eigenvalues, axis=1)
    ellipsoid_rho = float(np.sqrt(np.quantile(mahalanobis_squared, coverage_probability)))

    base_variance = iewma_diagonal_batch(
        x[None, :, :],
        volatility_half_life=iewma_volatility_half_life,
        variance_floor=iewma_variance_floor,
    )[0]
    k_statistics = []
    for start in range(0, bootstrap_replications, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_variance = iewma_diagonal_batch(
            x[batch_indices],
            volatility_half_life=iewma_volatility_half_life,
            variance_floor=iewma_variance_floor,
        )
        relative_increase = np.maximum(
            (batch_variance - base_variance) / np.maximum(base_variance, 1e-14),
            0.0,
        )
        k_statistics.extend(np.max(relative_increase, axis=1).tolist())
    diagonal_kappa = float(np.quantile(k_statistics, coverage_probability))

    assets = values.columns
    return UncertaintyCalibration(
        as_of=panel.as_of,
        standard_errors=pd.Series(standard_errors, index=assets, dtype=float),
        mean_error_covariance=pd.DataFrame(
            mean_covariance, index=assets, columns=assets, dtype=float
        ),
        box_rho=box_rho,
        ellipsoid_rho=ellipsoid_rho,
        diagonal_kappa=diagonal_kappa,
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_replications=int(bootstrap_replications),
        block_length=int(block_length),
        coverage_probability=float(coverage_probability),
        mean_covariance_ridge=ridge,
    )
