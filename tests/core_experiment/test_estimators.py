"""Core experiment mean/covariance estimator and scoring tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.data.schemas import ReturnPanel
from robust_portfolio.estimators import estimate_covariance, estimate_mean
from robust_portfolio.estimators.covariance import (
    ewma_covariance,
    ledoit_wolf_covariance,
)
from robust_portfolio.research.covariance_study import (
    evaluation_rows_after_forecast,
)


def panel(values, as_of="2020-01-10"):
    frame = pd.DataFrame(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
        columns=["A", "B"],
        dtype=float,
    )
    return ReturnPanel(pd.Timestamp(as_of), frame, "synthetic")


class TestEstimators(unittest.TestCase):
    def test_sample_covariance_correctness(self):
        bounded = panel([[0.01, 0.02], [0.03, 0.00], [-0.01, 0.04]])
        forecast = estimate_covariance(
            bounded,
            "sample",
            annualization_factor=1,
            absolute_eigenvalue_floor=0.0,
            relative_eigenvalue_floor=0.0,
        )
        np.testing.assert_allclose(
            forecast.annualized_covariance,
            bounded.values.cov(),
            atol=1e-14,
        )

    def test_ewma_matches_deterministic_weighted_example(self):
        values = np.array([[1.0, 2.0], [2.0, 0.0], [4.0, 1.0]])
        half_life = 2.0
        beta = 2.0 ** (-1.0 / half_life)
        weights = beta ** np.array([2.0, 1.0, 0.0])
        weights /= weights.sum()
        mean = weights @ values
        centered = values - mean
        expected = (centered * weights[:, None]).T @ centered
        np.testing.assert_allclose(ewma_covariance(values, half_life), expected, atol=1e-14)

    def test_iewma_is_symmetric_psd(self):
        bounded = panel([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.04], [0.02, -0.01]])
        covariance = estimate_covariance(bounded, "iewma", annualization_factor=1)
        matrix = covariance.annualized_covariance.to_numpy()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(matrix).min()), -1e-12)

    def test_iewma_as_of_forecast_does_not_change_when_future_rows_exist(self):
        prefix = panel([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.04]], as_of="2020-01-04")
        extended_values = pd.DataFrame(
            [[0.01, -0.02], [0.03, 0.01], [-0.01, 0.04], [9.0, -9.0]],
            index=pd.date_range("2020-01-01", periods=4, freq="D"),
            columns=["A", "B"],
        )
        same_information = ReturnPanel(
            pd.Timestamp("2020-01-04"), extended_values.iloc[:3], "extended"
        )
        first = estimate_covariance(prefix, "iewma", annualization_factor=1)
        second = estimate_covariance(same_information, "iewma", annualization_factor=1)
        np.testing.assert_allclose(first.annualized_covariance, second.annualized_covariance)

    def test_ledoit_wolf_is_symmetric_psd_with_valid_shrinkage(self):
        values = np.array(
            [[1.0, 1.5], [2.0, 2.5], [0.0, -0.5], [3.0, 2.0], [-1.0, -0.5]]
        )
        covariance, shrinkage = ledoit_wolf_covariance(values)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)
        self.assertGreaterEqual(shrinkage, 0.0)
        self.assertLessEqual(shrinkage, 1.0)

    def test_covariance_scoring_selects_strictly_future_rows(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        returns = pd.DataFrame({"A": range(6)}, index=dates)
        selected = evaluation_rows_after_forecast(returns, dates[2], dates[4])
        self.assertEqual(selected.index.tolist(), [dates[3], dates[4]])
        self.assertTrue(bool((selected.index > dates[2]).all()))

    def test_mean_estimators_use_consistent_annualization(self):
        bounded = panel([[0.01, 0.02], [0.03, 0.04], [0.02, 0.00]])
        sample = estimate_mean(bounded, "sample", annualization_factor=252)
        np.testing.assert_allclose(
            sample.annualized_mean.to_numpy(),
            bounded.values.mean().to_numpy() * 252,
        )
        shrink = estimate_mean(
            bounded, "shrink_zero", annualization_factor=252, shrinkage_intensity=0.5
        )
        np.testing.assert_allclose(shrink.annualized_mean, 0.5 * sample.annualized_mean)


if __name__ == "__main__":
    unittest.main()
