"""
Covariance forecasting utilities for the portfolio backtest.

This file adapts the IEWMA / CM-IEWMA idea from:
  Johansson, Ogut, Pelger, Schmelzer, Boyd (2023)
  "A Simple Method for Predicting Covariance Matrices of Financial Returns"

It also mirrors the practical implementation style used in the external
`cvx_options` repo the user referenced. The main idea is:

1. Estimate per-asset volatility with a fast EWMA.
2. Standardize returns by that volatility estimate.
3. Estimate correlations with a slower EWMA on standardized returns.
4. Recompose covariance as D_vol @ R @ D_vol.

For the robust strategy we use a combined-multiple IEWMA forecast (CM-IEWMA),
which blends several IEWMA experts with different half-life pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINSORIZE_CLIP = 4.2

# Single IEWMA half-lives used for the classical Markowitz strategy.
CLASSICAL_IEWMA_HALFLIFE_VOL = 21
CLASSICAL_IEWMA_HALFLIFE_COR = 63

# Multiple IEWMA half-life pairs used for the robust strategy.
ROBUST_CM_IEWMA_HALFLIFE_PAIRS = [
    (10, 21),
    (21, 63),
    (63, 125),
]

ROBUST_CM_IEWMA_LOOKBACK = 21
ROBUST_CM_IEWMA_TEMPERATURE = 1.0


def _halflife_to_beta(halflife: float) -> float:
    """Convert an EWMA half-life into the corresponding forgetting factor."""
    return 2.0 ** (-1.0 / halflife)


class IEWMAPredictor:
    """Single IEWMA covariance predictor."""

    def __init__(self, n_assets: int, halflife_vol: float, halflife_cor: float):
        self.n_assets = n_assets
        self.beta_vol = _halflife_to_beta(halflife_vol)
        self.beta_cor = _halflife_to_beta(halflife_cor)
        self._variance = None
        self._standardized_outer = None
        self._observation_count = 0

    def update(self, returns_row):
        """Update the volatility and correlation estimates with one return vector."""
        returns_row = np.asarray(returns_row, dtype=float).ravel()
        self._observation_count += 1

        squared_returns = returns_row ** 2
        if self._variance is None:
            self._variance = squared_returns.copy()
        else:
            self._variance = (
                self.beta_vol * self._variance + (1.0 - self.beta_vol) * squared_returns
            )

        sigma = np.sqrt(np.maximum(self._variance, 1e-14))
        standardized_returns = np.clip(
            returns_row / sigma,
            -WINSORIZE_CLIP,
            WINSORIZE_CLIP,
        )

        outer_product = np.outer(standardized_returns, standardized_returns)
        if self._standardized_outer is None:
            self._standardized_outer = outer_product.copy()
        else:
            self._standardized_outer = (
                self.beta_cor * self._standardized_outer
                + (1.0 - self.beta_cor) * outer_product
            )

    def predict(self):
        """Return the next-period covariance forecast."""
        if (
            self._variance is None
            or self._standardized_outer is None
            or self._observation_count < 2
        ):
            return None

        diagonal_scale = np.sqrt(np.maximum(np.diag(self._standardized_outer), 1e-14))
        inverse_diagonal = np.diag(1.0 / diagonal_scale)
        correlation = inverse_diagonal @ self._standardized_outer @ inverse_diagonal
        correlation = (correlation + correlation.T) / 2.0
        np.fill_diagonal(correlation, 1.0)

        sigma = np.sqrt(np.maximum(self._variance, 1e-14))
        diagonal_volatility = np.diag(sigma)
        covariance = diagonal_volatility @ correlation @ diagonal_volatility
        covariance = (covariance + covariance.T) / 2.0
        return covariance


class CMIEWMAPredictor:
    """
    Combined-multiple IEWMA predictor.

    Several IEWMA experts are maintained in parallel and blended using
    a trailing log-likelihood score over recent returns.
    """

    def __init__(
        self,
        n_assets: int,
        halflife_pairs=None,
        lookback=ROBUST_CM_IEWMA_LOOKBACK,
        temperature=ROBUST_CM_IEWMA_TEMPERATURE,
    ):
        halflife_pairs = halflife_pairs or ROBUST_CM_IEWMA_HALFLIFE_PAIRS
        self.predictors = [
            IEWMAPredictor(n_assets, halflife_vol, halflife_cor)
            for halflife_vol, halflife_cor in halflife_pairs
        ]
        self.lookback = lookback
        self.temperature = temperature
        self.n_assets = n_assets
        self._recent_returns = []
        self._last_weights = None

    def update(self, returns_row):
        returns_row = np.asarray(returns_row, dtype=float).ravel()
        for predictor in self.predictors:
            predictor.update(returns_row)

        self._recent_returns.append(returns_row.copy())
        if len(self._recent_returns) > self.lookback + 1:
            self._recent_returns.pop(0)

    def predict(self):
        """Return the blended covariance forecast."""
        predictions = [predictor.predict() for predictor in self.predictors]
        valid_predictions = [(index, sigma) for index, sigma in enumerate(predictions) if sigma is not None]

        if not valid_predictions:
            self._last_weights = None
            return None

        if len(valid_predictions) == 1:
            self._last_weights = np.ones(1)
            return valid_predictions[0][1]

        weights = self._score_weights(valid_predictions)
        self._last_weights = weights
        covariance = sum(weight * sigma for weight, (_, sigma) in zip(weights, valid_predictions))
        covariance = (covariance + covariance.T) / 2.0
        return covariance

    def get_last_weights(self):
        return self._last_weights

    def _score_weights(self, valid_predictions):
        """
        Score experts using trailing Gaussian log-likelihood and
        softmax the scores into convex combination weights.
        """
        recent_returns = self._recent_returns[-self.lookback :]
        if len(recent_returns) < 2:
            return np.ones(len(valid_predictions)) / len(valid_predictions)

        scores = np.zeros(len(valid_predictions))
        for prediction_index, (_, sigma) in enumerate(valid_predictions):
            sigma_regularized = sigma + np.eye(self.n_assets) * 1e-8

            try:
                cholesky = np.linalg.cholesky(sigma_regularized)
                log_determinant = 2.0 * np.sum(np.log(np.diag(cholesky)))
                sigma_inverse = np.linalg.inv(sigma_regularized)
            except np.linalg.LinAlgError:
                scores[prediction_index] = -1e12
                continue

            log_likelihood_sum = 0.0
            for returns_row in recent_returns:
                mahalanobis = returns_row @ sigma_inverse @ returns_row
                log_likelihood_sum += -0.5 * (log_determinant + mahalanobis)

            scores[prediction_index] = log_likelihood_sum / len(recent_returns)

        scores = scores - scores.max()
        weights = np.exp(scores / self.temperature)
        weight_sum = weights.sum()
        if weight_sum <= 1e-15:
            return np.ones(len(valid_predictions)) / len(valid_predictions)
        return weights / weight_sum


def estimate_iewma_covariance(
    returns_window: pd.DataFrame,
    halflife_vol: float = CLASSICAL_IEWMA_HALFLIFE_VOL,
    halflife_cor: float = CLASSICAL_IEWMA_HALFLIFE_COR,
):
    """
    Estimate the next covariance matrix using one IEWMA predictor.

    The returned covariance is daily. The caller can annualize it if needed.
    """
    predictor = IEWMAPredictor(
        n_assets=returns_window.shape[1],
        halflife_vol=halflife_vol,
        halflife_cor=halflife_cor,
    )

    for _, returns_row in returns_window.iterrows():
        predictor.update(returns_row.to_numpy(dtype=float))

    covariance = predictor.predict()
    if covariance is None:
        raise ValueError("IEWMA covariance estimation failed because not enough data was available.")
    return covariance


def estimate_cmiewma_covariance(
    returns_window: pd.DataFrame,
    halflife_pairs=None,
    lookback: int = ROBUST_CM_IEWMA_LOOKBACK,
    temperature: float = ROBUST_CM_IEWMA_TEMPERATURE,
):
    """
    Estimate the next covariance matrix using a CM-IEWMA predictor.

    The returned covariance is daily. The caller can annualize it if needed.
    """
    predictor = CMIEWMAPredictor(
        n_assets=returns_window.shape[1],
        halflife_pairs=halflife_pairs or ROBUST_CM_IEWMA_HALFLIFE_PAIRS,
        lookback=lookback,
        temperature=temperature,
    )

    for _, returns_row in returns_window.iterrows():
        predictor.update(returns_row.to_numpy(dtype=float))

    covariance = predictor.predict()
    if covariance is None:
        raise ValueError(
            "CM-IEWMA covariance estimation failed because not enough data was available."
        )
    return covariance
