import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robust_portfolio.calibration import calibrate_risk_aversion


class TestRiskAttainment(unittest.TestCase):
    def setUp(self):
        self.assets = pd.Index(["a", "b", "c"])
        self.mean = pd.Series([0.12, 0.07, 0.03], index=self.assets)
        self.covariance = pd.DataFrame(
            np.diag([0.04, 0.01, 0.0025]), index=self.assets, columns=self.assets
        )

    def test_bisection_attains_requested_predicted_risk(self):
        result = calibrate_risk_aversion(
            self.mean, self.covariance, target_volatility=0.10,
            maximum_weight=1.0, solver_order=["CLARABEL", "SCS"],
            volatility_tolerance=1e-4,
        )
        self.assertEqual(result.status, "ATTAINED")
        self.assertAlmostEqual(result.solution.predicted_volatility, 0.10, delta=1e-4)
        self.assertGreaterEqual(result.solution.risk_aversion, 0.0)

    def test_target_below_gmv_is_not_forced(self):
        result = calibrate_risk_aversion(
            self.mean, self.covariance, target_volatility=0.01,
            maximum_weight=1.0, solver_order=["CLARABEL", "SCS"],
        )
        self.assertEqual(result.status, "TARGET_NOT_ATTAINABLE")
        self.assertIsNone(result.solution)

    def test_target_above_zero_gamma_risk_is_not_forced(self):
        result = calibrate_risk_aversion(
            self.mean, self.covariance, target_volatility=0.30,
            maximum_weight=1.0, solver_order=["CLARABEL", "SCS"],
        )
        self.assertEqual(result.status, "TARGET_NOT_ATTAINABLE")
        self.assertIn("zero-risk-aversion", result.reason)
