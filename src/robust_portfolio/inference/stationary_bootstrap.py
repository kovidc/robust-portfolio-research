"""Joint stationary bootstrap for paired portfolio-return inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    observations: int,
    replications: int,
    expected_block_length: float,
    seed: int,
) -> np.ndarray:
    """Return Politis-Romano stationary-bootstrap indices shaped ``(B, T)``."""
    if observations < 2 or replications < 2:
        raise ValueError("Stationary bootstrap requires at least two rows and replications.")
    if expected_block_length < 1.0:
        raise ValueError("expected_block_length must be at least one.")
    generator = np.random.default_rng(int(seed))
    restart_probability = 1.0 / float(expected_block_length)
    indices = np.empty((replications, observations), dtype=np.int64)
    indices[:, 0] = generator.integers(0, observations, size=replications)
    for position in range(1, observations):
        restart = generator.random(replications) < restart_probability
        continuation = (indices[:, position - 1] + 1) % observations
        fresh = generator.integers(0, observations, size=replications)
        indices[:, position] = np.where(restart, fresh, continuation)
    return indices


def _statistics(samples: np.ndarray, annualization: int, ce_gamma: float) -> dict[str, np.ndarray]:
    means = samples.mean(axis=1) * annualization
    volatility = samples.std(axis=1, ddof=1) * np.sqrt(annualization)
    sharpe = np.divide(means, volatility, out=np.full_like(means, np.nan), where=volatility > 0)
    certainty_equivalent = means - 0.5 * ce_gamma * volatility**2
    return {
        "annualized_zero_rf_mean": means,
        "annualized_volatility": volatility,
        "provisional_zero_rf_sharpe": sharpe,
        "zero_rf_certainty_equivalent": certainty_equivalent,
    }


def bootstrap_headline_statistics(
    returns: pd.DataFrame,
    *,
    replications: int,
    expected_block_length: float,
    seed: int,
    confidence_level: float,
    annualization_factor: int,
    certainty_equivalent_risk_aversion: float,
    comparators: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Bootstrap all strategy columns with the same time indices.

    Returns metric intervals, paired strategy-minus-comparator intervals, and
    the shared index matrix used to pair the samples.
    """
    if returns.empty or returns.isna().any().any():
        raise ValueError("Inference requires a complete joint strategy-return panel.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Inference returns must be finite.")
    indices = stationary_bootstrap_indices(
        len(returns), replications, expected_block_length, seed
    )
    samples = values[indices]
    statistics = _statistics(
        samples, annualization_factor, certainty_equivalent_risk_aversion
    )
    point = _statistics(
        values[None, :, :], annualization_factor, certainty_equivalent_risk_aversion
    )
    alpha = (1.0 - confidence_level) / 2.0
    records = []
    for metric, draws in statistics.items():
        for position, strategy in enumerate(returns.columns):
            records.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "estimate": float(point[metric][0, position]),
                    "ci_lower": float(np.quantile(draws[:, position], alpha)),
                    "ci_upper": float(np.quantile(draws[:, position], 1.0 - alpha)),
                    "confidence_level": confidence_level,
                    "method": "joint_stationary_bootstrap_percentile",
                }
            )
    differences = []
    for comparator in comparators:
        if comparator not in returns.columns:
            raise ValueError(f"Comparator {comparator!r} is not in the joint panel.")
        comparator_position = returns.columns.get_loc(comparator)
        for position, strategy in enumerate(returns.columns):
            if strategy == comparator:
                continue
            for metric in ("provisional_zero_rf_sharpe", "zero_rf_certainty_equivalent"):
                draws = statistics[metric][:, position] - statistics[metric][:, comparator_position]
                estimate = point[metric][0, position] - point[metric][0, comparator_position]
                differences.append(
                    {
                        "strategy": strategy,
                        "comparator": comparator,
                        "metric": f"delta_{metric}",
                        "estimate": float(estimate),
                        "ci_lower": float(np.quantile(draws, alpha)),
                        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
                        "confidence_level": confidence_level,
                        "method": "paired_joint_stationary_bootstrap_percentile",
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(differences), indices
