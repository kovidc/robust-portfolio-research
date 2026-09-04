"""Core experiment transaction-cost scenarios and reproducibility tests."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/robust_portfolio_test_mpl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.data import FrozenCsvReturnProvider  # noqa: E402
from robust_portfolio.research.core_experiment import run_core_experiment  # noqa: E402
from robust_portfolio.research.simulation import simulate_targets  # noqa: E402


class TestCostScenarios(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dates = pd.bdate_range("2020-01-01", periods=6)
        self.returns = pd.DataFrame(
            {
                "A": [0.0, 0.10, 0.00, -0.05, 0.02, 0.01],
                "B": [0.0, 0.00, 0.05, 0.02, -0.01, 0.03],
            },
            index=self.dates,
        )
        path = self.root / "returns.csv"
        self.returns.to_csv(path)
        self.provider = FrozenCsvReturnProvider(path)
        self.targets = {
            self.dates[1]: pd.Series({"A": 0.5, "B": 0.5}),
            self.dates[3]: pd.Series({"A": 0.2, "B": 0.8}),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_cost_net_and_gross_paths_are_identical(self):
        path = simulate_targets(
            self.provider,
            self.targets,
            strategy="synthetic",
            cost_bps=0,
            maximum_weight=1.0,
            market_returns=self.returns,
        )
        pd.testing.assert_series_equal(
            path.daily["gross_wealth"], path.daily["net_wealth"], check_names=False
        )
        self.assertEqual(float(path.daily["transaction_cost"].sum()), 0.0)

    def test_positive_cost_is_charged_to_actual_trades_and_reduces_wealth(self):
        path = simulate_targets(
            self.provider,
            self.targets,
            strategy="synthetic",
            cost_bps=100,
            maximum_weight=1.0,
            market_returns=self.returns,
        )
        self.assertGreater(float(path.daily["transaction_cost"].sum()), 0.0)
        self.assertLess(path.daily["net_wealth"].iloc[-1], path.daily["gross_wealth"].iloc[-1])
        self.assertAlmostEqual(path.net_executions[0].gross_traded_fraction, 1.0)
        self.assertAlmostEqual(path.net_executions[0].one_way_turnover, 0.5)

    def test_cost_scenario_does_not_change_gross_path(self):
        zero = simulate_targets(
            self.provider, self.targets, strategy="synthetic", cost_bps=0,
            maximum_weight=1.0, market_returns=self.returns,
        )
        costly = simulate_targets(
            self.provider, self.targets, strategy="synthetic", cost_bps=25,
            maximum_weight=1.0, market_returns=self.returns,
        )
        pd.testing.assert_series_equal(
            zero.daily["gross_wealth"], costly.daily["gross_wealth"], check_names=False
        )


class TestCoreExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="core_experiment_test_")
        cls.root = Path(cls.temporary.name)
        payload = json.loads(
            (REPOSITORY_ROOT / "configs" / "core_experiment.json").read_text(encoding="utf-8")
        )
        payload["uncertainty"]["bootstrap_replications"] = 8
        payload["uncertainty"]["block_length_observations"] = 5
        payload["risk_matching"]["target_annual_volatility"] = [0.10]
        payload["costs"]["basis_points_per_dollar_traded"] = [0, 5]
        cls.config_path = cls.root / "config.json"
        cls.config_path.write_text(json.dumps(payload), encoding="utf-8")
        cls.first = run_core_experiment(
            cls.config_path,
            repository_root=REPOSITORY_ROOT,
            output_dir=cls.root / "first",
            max_outer_dates=1,
        )
        cls.second = run_core_experiment(
            cls.config_path,
            repository_root=REPOSITORY_ROOT,
            output_dir=cls.root / "second",
            max_outer_dates=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_config_and_numeric_results_are_reproducible(self):
        first = self.first["metrics"].sort_values(["strategy", "cost_bps"]).reset_index(drop=True)
        second = self.second["metrics"].sort_values(["strategy", "cost_bps"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            self.first["manifest"]["configuration"],
            self.second["manifest"]["configuration"],
        )

    def test_manifest_records_required_reproducibility_metadata(self):
        manifest = self.first["manifest"]
        self.assertEqual(manifest["universe"]["label"], "SURVIVOR-CONDITIONED PUBLIC-DATA EXPERIMENT")
        self.assertFalse(manifest["universe"]["survivorship_bias_free"])
        self.assertEqual(manifest["outer_experiment"]["formal_start_date"], "2018-04-02T00:00:00")
        self.assertEqual(len(manifest["outer_experiment"]["first_outer_inner_folds"]), 4)
        for key in ("git", "configuration", "inputs", "environment", "bootstrap", "risk_targets", "cost_scenarios_bps"):
            self.assertIn(key, manifest)

    def test_all_required_tables_and_figures_are_created(self):
        paths = self.first["artifact_paths"]
        required = {
            "table_1_covariance_estimator_study",
            "table_2_core_strategy_comparison",
            "figure_1_gross_vs_net_wealth",
            "figure_2_realized_risk_return",
            "figure_3_predicted_vs_realized_volatility",
            "figure_4_covariance_forecast_loss",
            "figure_5_turnover_cost_drag",
            "experiment_manifest",
            "run_manifest",
            "universe_snapshots",
            "inner_folds",
        }
        self.assertTrue(required.issubset(paths))
        self.assertTrue(all(Path(paths[key]).exists() for key in required))

    def test_provisional_sharpe_is_explicitly_labeled(self):
        self.assertIn("provisional_zero_rf_sharpe", self.first["metrics"].columns)
        self.assertNotIn("sharpe_ratio", self.first["metrics"].columns)


if __name__ == "__main__":
    unittest.main()
