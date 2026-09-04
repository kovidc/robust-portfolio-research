"""Secondary Deflated Sharpe Ratio diagnostic."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew


def deflated_sharpe_probability(
    returns,
    trial_annualized_sharpes,
    *,
    annualization_factor: int = 252,
) -> dict[str, float | int]:
    """Compute the Bailey-Lopez de Prado DSR approximation.

    The observed dispersion across the finite candidate set estimates the
    expected maximum Sharpe under selection. This is reported as a secondary
    diagnostic because correlated trials weaken the independent-trial
    approximation.
    """
    values = np.asarray(returns, dtype=float)
    trials = np.asarray(trial_annualized_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("DSR requires a finite one-dimensional return series.")
    if len(trials) < 2:
        raise ValueError("DSR requires at least two finite candidate Sharpes.")
    daily_std = float(values.std(ddof=1))
    if daily_std <= 0.0:
        raise ValueError("DSR is undefined for zero-volatility returns.")
    daily_sharpe = float(values.mean() / daily_std)
    annualized_sharpe = daily_sharpe * np.sqrt(annualization_factor)
    trial_std = float(trials.std(ddof=1))
    euler_gamma = 0.5772156649015329
    count = len(trials)
    expected_maximum = trial_std * (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / count)
        + euler_gamma * norm.ppf(1.0 - 1.0 / (count * np.e))
    )
    daily_benchmark = expected_maximum / np.sqrt(annualization_factor)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator = np.sqrt(
        max(
            1.0
            - sample_skew * daily_sharpe
            + 0.25 * (sample_kurtosis - 1.0) * daily_sharpe**2,
            1e-16,
        )
    )
    statistic = (
        (daily_sharpe - daily_benchmark) * np.sqrt(len(values) - 1.0) / denominator
    )
    return {
        "candidate_count": int(count),
        "annualized_sharpe": annualized_sharpe,
        "annualized_expected_maximum_sharpe": float(expected_maximum),
        "return_skewness": sample_skew,
        "return_kurtosis": sample_kurtosis,
        "deflated_sharpe_probability": float(norm.cdf(statistic)),
    }
