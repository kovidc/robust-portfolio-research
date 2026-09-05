"""Core experiment benchmark feasibility and robust-counterpart tests."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.optimizers import (
    OptimizationFailure,
    asset_class_equal_weight,
    box_worst_case_mean,
    diagonal_robust_covariance,
    ellipsoid_worst_case_mean,
    global_minimum_variance,
    inverse_volatility,
    risk_parity,
    solve_target_risk,
)
from robust_portfolio.optimizers.robust import (
    explicit_box_worst_case_mean,
    explicit_diagonal_worst_case_variance,
    explicit_ellipsoid_worst_case_mean,
)

SOLVERS = ["CLARABEL", "SCS"]


class TestOptimizers(unittest.TestCase):
    def test_asset_class_equal_weight(self):
        classes = pd.Series({"A": "equity", "B": "equity", "C": "bonds"})
        weights = asset_class_equal_weight(["A", "B", "C"], classes)
        self.assertAlmostEqual(weights["A"], 0.25)
        self.assertAlmostEqual(weights["B"], 0.25)
        self.assertAlmostEqual(weights["C"], 0.50)

    def test_inverse_volatility(self):
        covariance = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
        weights = inverse_volatility(covariance)
        np.testing.assert_allclose(weights.to_numpy(), [1.0 / 3.0, 2.0 / 3.0])

    def test_gmv_hand_checkable_diagonal_solution(self):
        covariance = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
        result = global_minimum_variance(covariance, maximum_weight=1.0, solver_order=SOLVERS)
        np.testing.assert_allclose(result.weights.to_numpy(), [0.2, 0.8], atol=2e-5)

    def test_risk_parity_equalizes_contributions(self):
        covariance = pd.DataFrame(np.diag([0.04, 0.01]), index=["A", "B"], columns=["A", "B"])
        weights = risk_parity(covariance)
        contributions = weights.to_numpy() * (covariance.to_numpy() @ weights.to_numpy())
        self.assertAlmostEqual(contributions[0], contributions[1], delta=2e-7)

    def test_nominal_mvo_is_feasible_and_hits_binding_target(self):
        covariance = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
        mean = pd.Series({"A": 0.01, "B": 0.20})
        result = solve_target_risk(
            mean,
            covariance,
            target_volatility=0.15,
            maximum_weight=1.0,
            solver_order=SOLVERS,
        )
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=7)
        self.assertLessEqual(result.predicted_volatility, 0.15001)
        self.assertTrue(result.target_binding)

    def test_box_explicit_and_closed_form_are_equivalent(self):
        mu = np.array([0.06, 0.03, 0.08])
        s = np.array([0.01, 0.02, 0.015])
        weights = np.array([0.2, 0.3, 0.5])
        closed = box_worst_case_mean(mu, s, 1.7, weights)
        explicit = explicit_box_worst_case_mean(mu, s, 1.7, weights)
        self.assertAlmostEqual(closed, explicit, places=8)

    def test_box_short_case_retains_absolute_weights(self):
        mu = np.array([0.06, 0.03])
        s = np.array([0.01, 0.02])
        weights = np.array([1.2, -0.2])
        correct = box_worst_case_mean(mu, s, 2.0, weights)
        incorrect_long_only_simplification = float((mu - 2.0 * s) @ weights)
        self.assertNotAlmostEqual(correct, incorrect_long_only_simplification)

    def test_diagonal_covariance_explicit_and_closed_form_are_equivalent(self):
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = np.array([0.4, 0.6])
        closed = float(weights @ diagonal_robust_covariance(covariance, 0.3) @ weights)
        explicit = explicit_diagonal_worst_case_variance(covariance, 0.3, weights)
        self.assertAlmostEqual(closed, explicit, places=8)

    def test_ellipsoid_explicit_and_closed_form_are_equivalent(self):
        mu = np.array([0.05, 0.07])
        covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        weights = np.array([0.4, 0.6])
        closed = ellipsoid_worst_case_mean(mu, covariance, 1.5, weights)
        explicit = explicit_ellipsoid_worst_case_mean(mu, covariance, 1.5, weights)
        self.assertAlmostEqual(closed, explicit, places=8)

    def test_zero_box_radius_reproduces_nominal(self):
        covariance = pd.DataFrame(np.diag([0.01, 0.04]), index=["A", "B"], columns=["A", "B"])
        mean = pd.Series({"A": 0.01, "B": 0.20})
        nominal = solve_target_risk(mean, covariance, target_volatility=0.15, maximum_weight=1.0, solver_order=SOLVERS)
        box = solve_target_risk(
            mean,
            covariance,
            target_volatility=0.15,
            maximum_weight=1.0,
            solver_order=SOLVERS,
            standard_errors=pd.Series({"A": 0.02, "B": 0.03}),
            box_rho=0.0,
        )
        np.testing.assert_allclose(nominal.weights, box.weights, atol=1e-6)

    def test_infeasible_target_fails_explicitly(self):
        covariance = pd.DataFrame(np.diag([0.04, 0.04]), index=["A", "B"], columns=["A", "B"])
        with self.assertRaisesRegex(OptimizationFailure, "below feasible minimum"):
            solve_target_risk(
                pd.Series({"A": 0.1, "B": 0.2}),
                covariance,
                target_volatility=0.05,
                maximum_weight=1.0,
                solver_order=SOLVERS,
            )

    def test_solver_failure_never_returns_equal_weight_fallback(self):
        covariance = pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])
        with self.assertRaisesRegex(OptimizationFailure, "All configured solvers failed"):
            global_minimum_variance(
                covariance, maximum_weight=1.0, solver_order=["NOT_A_SOLVER"]
            )


if __name__ == "__main__":
    unittest.main()
