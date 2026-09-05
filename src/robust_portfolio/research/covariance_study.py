"""Sequential out-of-sample covariance forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_portfolio.estimators.covariance import CovarianceForecast
from robust_portfolio.optimizers import global_minimum_variance


def gaussian_nll(covariance: np.ndarray, observations: np.ndarray) -> float:
    """Average Gaussian covariance loss per asset, omitting the common constant."""
    sigma = np.asarray(covariance, dtype=float)
    x = np.asarray(observations, dtype=float)
    cholesky = np.linalg.cholesky(sigma)
    log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
    solved = np.linalg.solve(cholesky, x.T)
    quadratic = np.sum(solved**2, axis=0)
    return float(np.mean(log_determinant + quadratic) / sigma.shape[0])


def effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    positive = probabilities > 0.0
    return float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))


def evaluation_rows_after_forecast(
    returns: pd.DataFrame,
    forecast_date,
    next_rebalance_date=None,
) -> pd.DataFrame:
    """Rows strictly after a forecast, through the next execution row if supplied."""
    mask = returns.index > pd.Timestamp(forecast_date)
    if next_rebalance_date is not None:
        mask &= returns.index <= pd.Timestamp(next_rebalance_date)
    return returns.loc[mask]


def run_covariance_study(
    forecasts_by_date: dict[pd.Timestamp, dict[str, CovarianceForecast]],
    returns: pd.DataFrame,
    outer_dates: pd.DatetimeIndex,
    *,
    annualization_factor: int,
    maximum_weight: float,
    solver_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_records = []
    gmv_returns: dict[str, list[pd.Series]] = {}
    for position, forecast_date in enumerate(outer_dates):
        next_date = outer_dates[position + 1] if position + 1 < len(outer_dates) else None
        future = evaluation_rows_after_forecast(returns, forecast_date, next_date)
        if future.empty:
            continue
        for method, forecast in forecasts_by_date[forecast_date].items():
            assets = forecast.annualized_covariance.index
            future_aligned = future.loc[:, assets]
            annualized_covariance = forecast.annualized_covariance.to_numpy()
            daily_covariance = annualized_covariance / float(annualization_factor)
            equal = np.repeat(1.0 / len(assets), len(assets))
            predicted_volatility = float(np.sqrt(equal @ annualized_covariance @ equal))
            equal_returns = future_aligned.to_numpy() @ equal
            realized_volatility = float(
                np.std(equal_returns, ddof=1) * np.sqrt(annualization_factor)
            )
            gmv = global_minimum_variance(
                forecast.annualized_covariance,
                maximum_weight=maximum_weight,
                solver_order=solver_order,
            )
            gmv_period_returns = future_aligned @ gmv.weights
            gmv_returns.setdefault(method, []).append(gmv_period_returns)
            eigenvalues = np.linalg.eigvalsh(annualized_covariance)
            condition = float(eigenvalues.max() / max(eigenvalues.min(), 1e-18))
            period_records.append(
                {
                    "forecast_date": forecast_date,
                    "evaluation_start": future.index[0],
                    "evaluation_end": future.index[-1],
                    "estimator": method,
                    "oos_gaussian_nll_per_asset": gaussian_nll(
                        daily_covariance, future_aligned.to_numpy()
                    ),
                    "predicted_equal_weight_volatility": predicted_volatility,
                    "realized_equal_weight_volatility": realized_volatility,
                    "volatility_error": predicted_volatility - realized_volatility,
                    "condition_number": condition,
                    "effective_rank": effective_rank(annualized_covariance),
                    "gmv_predicted_volatility": gmv.predicted_volatility,
                    "gmv_solver": gmv.solver,
                }
            )
    periods = pd.DataFrame(period_records)
    summaries = []
    for method, group in periods.groupby("estimator", sort=True):
        concatenated = pd.concat(gmv_returns[method]).sort_index()
        summaries.append(
            {
                "estimator": method,
                "oos_covariance_loss": float(group["oos_gaussian_nll_per_asset"].mean()),
                "vol_forecast_bias": float(group["volatility_error"].mean()),
                "vol_forecast_rmse": float(np.sqrt(np.mean(group["volatility_error"] ** 2))),
                "median_condition_number": float(group["condition_number"].median()),
                "median_effective_rank": float(group["effective_rank"].median()),
                "gmv_oos_volatility": float(
                    concatenated.std(ddof=1) * np.sqrt(annualization_factor)
                ),
                "forecast_periods": len(group),
            }
        )
    return pd.DataFrame(summaries).set_index("estimator"), periods
