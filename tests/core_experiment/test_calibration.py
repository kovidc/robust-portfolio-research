"""Core experiment bootstrap reproducibility and nested-scope tests."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.calibration import derive_outer_schedule  # noqa: E402
from robust_portfolio.data import FrozenCsvReturnProvider  # noqa: E402
from robust_portfolio.data.schemas import ReturnPanel  # noqa: E402
from robust_portfolio.estimators import calibrate_uncertainty  # noqa: E402
from robust_portfolio.estimators.uncertainty import (  # noqa: E402
    circular_block_bootstrap_indices,
)


class TestCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(7)
        values = pd.DataFrame(
            generator.normal(0.0, 0.01, size=(80, 3)),
            index=pd.date_range("2020-01-01", periods=80, freq="D"),
            columns=["A", "B", "C"],
        )
        cls.panel = ReturnPanel(pd.Timestamp("2020-04-01"), values, "synthetic")

    def calibration(self):
        return calibrate_uncertainty(
            self.panel,
            bootstrap_seed=99,
            bootstrap_replications=40,
            block_length=5,
            coverage_probability=0.90,
            annualization_factor=252,
        )

    def test_block_bootstrap_reproducibility(self):
        first = circular_block_bootstrap_indices(20, 10, 4, 123)
        second = circular_block_bootstrap_indices(20, 10, 4, 123)
        np.testing.assert_array_equal(first, second)

    def test_rho_bootstrap_calibration_reproducibility(self):
        first, second = self.calibration(), self.calibration()
        self.assertEqual(first.box_rho, second.box_rho)
        self.assertEqual(first.ellipsoid_rho, second.ellipsoid_rho)
        np.testing.assert_allclose(first.standard_errors, second.standard_errors)

    def test_kappa_calibration_reproducibility(self):
        first, second = self.calibration(), self.calibration()
        self.assertEqual(first.diagonal_kappa, second.diagonal_kappa)
        self.assertGreaterEqual(first.diagonal_kappa, 0.0)

    def test_mean_error_covariance_is_psd(self):
        calibration = self.calibration()
        self.assertGreater(
            float(np.linalg.eigvalsh(calibration.mean_error_covariance).min()), 0.0
        )

    def test_inner_folds_cannot_access_outer_period(self):
        dates = pd.bdate_range("2019-01-01", periods=120)
        schedule = pd.DatetimeIndex([dates[30], dates[50], dates[70], dates[90], dates[110]])
        outer, folds = derive_outer_schedule(
            dates,
            schedule,
            estimation_window=20,
            minimum_prior_inner_folds=4,
        )
        self.assertEqual(outer[0], schedule[4])
        self.assertTrue(all(fold.validation_end < outer[0] for fold in folds[outer[0]]))
        self.assertTrue(all(fold.fit_end < fold.decision_date for fold in folds[outer[0]]))

    def test_repository_history_implies_2018_04_02_outer_start(self):
        provider = FrozenCsvReturnProvider(REPOSITORY_ROOT / "data" / "returns_clean.csv")
        schedule = pd.DatetimeIndex(
            pd.read_csv(
                REPOSITORY_ROOT / "data" / "quarterly_rebalance_dates.csv",
                parse_dates=["rebalance_date"],
            )["rebalance_date"]
        )
        outer, _ = derive_outer_schedule(
            provider.dates,
            schedule,
            estimation_window=504,
            minimum_prior_inner_folds=4,
        )
        self.assertEqual(outer[0], pd.Timestamp("2018-04-02"))


if __name__ == "__main__":
    unittest.main()
