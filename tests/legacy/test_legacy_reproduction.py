"""Golden-master tests for the intentionally uncorrected CS361 baseline."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
LEGACY_SOURCE_DIR = REPOSITORY_ROOT / "src"
for path in (SCRIPTS_DIR, LEGACY_SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import backtest as legacy_backtest  # noqa: E402
import covariance as legacy_covariance  # noqa: E402
import reproduce_legacy  # noqa: E402
import strategies as legacy_strategies  # noqa: E402
import tune_hyperparameters as legacy_tuning  # noqa: E402


def _load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class TestLegacyReproduction(unittest.TestCase):
    """Run one offline legacy reproduction, then validate its requirements."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = _load_json(REPOSITORY_ROOT / "legacy" / "baseline_manifest.json")
        cls.config = _load_json(REPOSITORY_ROOT / "legacy" / "config.json")
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="cs361_legacy_test_")
        cls.output_dir = Path(cls.temporary_directory.name) / "reproduced"

        network_error = AssertionError("The LEGACY reproduction attempted network access.")
        with (
            mock.patch("socket.create_connection", side_effect=network_error),
            mock.patch.object(socket.socket, "connect", side_effect=network_error),
        ):
            cls.result = reproduce_legacy.reproduce_legacy(
                output_dir=cls.output_dir,
                create_plots=False,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_frozen_inputs_source_config_and_historical_evidence_match_hashes(self):
        verification = reproduce_legacy.verify_frozen_evidence(self.manifest)
        self.assertTrue(verification["all_match"], verification["mismatches"])
        for group in verification["groups"].values():
            self.assertTrue(all(check["matches"] for check in group.values()))

    def test_manifest_and_artifacts_are_explicitly_legacy(self):
        self.assertEqual(self.manifest["result_label"], "LEGACY")
        self.assertEqual(self.config["result_label"], "LEGACY")
        self.assertEqual(self.result["metrics_artifact"]["result_label"], "LEGACY")
        self.assertEqual(self.result["accounting_diagnostic"]["result_label"], "LEGACY")
        self.assertIn("not corrected research", self.manifest["warning"].lower())

    def test_complete_selected_configuration_is_frozen(self):
        optimization = self.config["optimization"]
        estimation = self.config["estimation"]
        walk_forward = self.config["walk_forward"]
        self.assertEqual(walk_forward["training_window_trading_days"], 504)
        self.assertEqual(walk_forward["rebalance_frequency"], "quarterly")
        self.assertEqual(walk_forward["rebalance_count"], 35)
        self.assertEqual(optimization["maximum_weight"], 0.10)
        self.assertEqual(optimization["gamma_classical"], 5.0)
        self.assertEqual(optimization["gamma_robust"], 20.0)
        self.assertEqual(optimization["rho"], 0.25)
        self.assertEqual(optimization["kappa"], 0.20)
        self.assertEqual(optimization["solver_order"], ["OSQP", "CLARABEL", "SCS"])
        self.assertEqual(
            estimation["classical_covariance"],
            {"method": "single IEWMA", "volatility_half_life": 21, "correlation_half_life": 63},
        )
        self.assertEqual(
            estimation["robust_covariance"]["expert_half_life_pairs"],
            [[10, 21], [21, 63], [63, 125]],
        )
        self.assertEqual(estimation["statistics_covariance_diagonal_ridge"], 1e-6)
        self.assertEqual(estimation["nearest_psd_minimum_eigenvalue"], 1e-8)
        self.assertEqual(len(self.config["data"]["requested_tickers"]), 148)
        self.assertEqual(len(self.config["data"]["retained_tickers"]), 147)
        self.assertEqual(self.config["data"]["dropped_tickers"], ["JO"])

    def test_original_entry_point_default_differences_are_preserved(self):
        defaults = self.config["entry_point_defaults"]
        self.assertEqual(defaults["src/main.py"]["gamma_classical"], 5.0)
        self.assertEqual(defaults["src/main.py"]["gamma_robust"], 20.0)
        signature = inspect.signature(legacy_backtest.run_backtest)
        self.assertEqual(signature.parameters["classical_gamma"].default, 10)
        self.assertEqual(signature.parameters["robust_gamma"].default, 10)
        self.assertEqual(signature.parameters["rho"].default, 1.0)
        self.assertEqual(signature.parameters["cov_uncertainty"].default, 0.10)
        self.assertEqual(signature.parameters["max_weight"].default, 0.10)

    def test_estimator_and_tuning_constants_are_preserved(self):
        self.assertEqual(legacy_covariance.WINSORIZE_CLIP, 4.2)
        self.assertEqual(legacy_covariance.CLASSICAL_IEWMA_HALFLIFE_VOL, 21)
        self.assertEqual(legacy_covariance.CLASSICAL_IEWMA_HALFLIFE_COR, 63)
        self.assertEqual(
            legacy_covariance.ROBUST_CM_IEWMA_HALFLIFE_PAIRS,
            [(10, 21), (21, 63), (63, 125)],
        )
        self.assertEqual(legacy_covariance.ROBUST_CM_IEWMA_LOOKBACK, 21)
        self.assertEqual(legacy_covariance.ROBUST_CM_IEWMA_TEMPERATURE, 1.0)
        self.assertEqual(legacy_tuning.CLASSICAL_GAMMA_GRID, [3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
        self.assertEqual(legacy_tuning.ROBUST_GAMMA_GRID, [5.0, 10.0, 15.0, 20.0, 30.0])
        self.assertEqual(legacy_tuning.ROBUST_RHO_GRID, [0.25, 0.50, 0.75, 1.0, 1.5])
        self.assertEqual(legacy_tuning.ROBUST_COV_UNCERTAINTY_GRID, [0.05, 0.10, 0.15, 0.20])

    def test_reproduced_metrics_match_expected_legacy_metrics(self):
        tolerance = self.manifest["numerical_tolerances"]["metric_absolute"]
        actual = self.result["metrics"]
        for strategy, expected_metrics in self.manifest["expected_legacy_metrics"].items():
            for metric, expected in expected_metrics.items():
                self.assertAlmostEqual(
                    float(actual.loc[strategy, metric]),
                    float(expected),
                    delta=tolerance,
                    msg=f"{strategy} {metric}",
                )
        self.assertTrue(self.result["metrics_artifact"]["golden_master_validation"]["passed"])

    def test_reproduced_returns_values_and_turnover_match_stored_golden_master(self):
        tolerances = self.manifest["numerical_tolerances"]
        comparisons = (
            ("portfolio_returns.csv", tolerances["daily_return_absolute"]),
            ("portfolio_values.csv", tolerances["portfolio_value_absolute"]),
            ("turnover.csv", tolerances["turnover_absolute"]),
        )
        for filename, tolerance in comparisons:
            expected = pd.read_csv(
                REPOSITORY_ROOT / "outputs" / filename, index_col=0, parse_dates=True
            )
            actual = pd.read_csv(self.output_dir / filename, index_col=0, parse_dates=True)
            pd.testing.assert_index_equal(actual.index, expected.index)
            pd.testing.assert_index_equal(actual.columns, expected.columns)
            np.testing.assert_allclose(
                actual.to_numpy(),
                expected.to_numpy(),
                atol=tolerance,
                rtol=0.0,
                equal_nan=True,
                err_msg=filename,
            )

    def test_all_35_target_weight_rows_match_and_are_feasible(self):
        tolerance = self.manifest["numerical_tolerances"]["solver_weight_absolute"]
        sum_tolerance = self.manifest["numerical_tolerances"]["weight_sum_absolute"]
        bound_tolerance = self.manifest["numerical_tolerances"]["weight_bound_absolute"]
        max_weight = self.config["optimization"]["maximum_weight"]

        for strategy, filename in reproduce_legacy.WEIGHT_FILES.items():
            expected = pd.read_csv(
                REPOSITORY_ROOT / "outputs" / filename, index_col=0, parse_dates=True
            )
            actual = pd.read_csv(self.output_dir / filename, index_col=0, parse_dates=True)
            self.assertEqual(len(actual), 35, strategy)
            pd.testing.assert_index_equal(actual.index, expected.index)
            pd.testing.assert_index_equal(actual.columns, expected.columns)
            np.testing.assert_allclose(
                actual.to_numpy(), expected.to_numpy(), atol=tolerance, rtol=0.0, equal_nan=True
            )
            np.testing.assert_allclose(
                actual.sum(axis=1).to_numpy(), 1.0, atol=sum_tolerance, rtol=0.0
            )
            self.assertGreaterEqual(float(actual.min().min()), -bound_tolerance)
            self.assertLessEqual(float(actual.max().max()), max_weight + bound_tolerance)

    def test_no_optimizer_fallback_was_recorded(self):
        run = self.result["run_artifact"]
        self.assertFalse(run["optimizer_fallback_detected"])
        log = (self.output_dir / "legacy_console.log").read_text(encoding="utf-8")
        self.assertNotIn("falling back to equal-weight portfolios", log)
        self.assertNotIn("failed, using equal weight fallback", log)

    def test_accounting_contradiction_is_explicitly_reproduced(self):
        diagnostic = self.result["accounting_diagnostic"]
        self.assertEqual(diagnostic["artifact_type"], "LEGACY_ACCOUNTING_CONTRADICTION")
        self.assertTrue(diagnostic["all_strategies_exhibit_contradiction"])
        for strategy, result in diagnostic["strategies"].items():
            self.assertTrue(
                result["reported_returns_match_daily_fixed_target_weights"], strategy
            )
            self.assertTrue(
                result["reported_turnover_matches_quarterly_drifted_pretrade_weights"],
                strategy,
            )
            self.assertTrue(result["return_and_turnover_accounting_are_contradictory"], strategy)
            self.assertGreaterEqual(
                result["absolute_cumulative_return_path_difference"],
                diagnostic["required_cumulative_path_difference"],
            )

    def test_reproduction_is_offline_and_namespace_isolated(self):
        run = self.result["run_artifact"]
        self.assertFalse(run["network_access_used"])
        self.assertNotEqual(self.output_dir.resolve(), (REPOSITORY_ROOT / "outputs").resolve())
        self.assertTrue(run["hash_verification"]["all_match"])
        with self.assertRaises(ValueError):
            reproduce_legacy._validate_output_directory(REPOSITORY_ROOT / "outputs")
        with self.assertRaises(ValueError):
            reproduce_legacy._validate_output_directory(REPOSITORY_ROOT)
        with self.assertRaises(ValueError):
            reproduce_legacy._validate_output_directory(
                REPOSITORY_ROOT / "artifacts" / "corrected_research"
            )

    def test_legacy_robust_inputs_remain_non_comparable_by_design(self):
        self.assertNotEqual(
            self.config["optimization"]["gamma_classical"],
            self.config["optimization"]["gamma_robust"],
        )
        self.assertNotEqual(
            self.config["estimation"]["classical_covariance"]["method"],
            self.config["estimation"]["robust_covariance"]["method"],
        )
        robust_signature = inspect.signature(legacy_strategies.robust_markowitz_strategy)
        self.assertIn("cov_uncertainty", robust_signature.parameters)


if __name__ == "__main__":
    unittest.main()
