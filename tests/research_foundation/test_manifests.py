"""Manifest, labeling, isolation, and reproducibility tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.backtest import BacktestEngine
from robust_portfolio.config import ResearchConfig
from robust_portfolio.data import (
    FrozenCsvReturnProvider,
    SurvivorPanelUniverseBuilder,
    UniverseRules,
)
from robust_portfolio.reporting.artifacts import (
    validate_artifact_directory,
)


def _setup_fixture(root: Path):
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    returns_path = root / "returns.csv"
    pd.DataFrame({"A": [0.0, 0.1, -0.05, 0.02]}, index=dates).to_csv(returns_path)
    payload = {
        "schema_version": 1,
        "experiment": {
            "name": "reproducibility",
            "result_label": "ACCOUNTING DIAGNOSTIC — NOT FINAL RESEARCH RESULTS",
        },
        "data": {"universe_mode": "SURVIVOR_PANEL"},
        "universe": {
            "required_history_observations": 1,
            "require_complete_required_window": True,
        },
        "backtest": {
            "estimation_window_observations": 1,
            "execution_convention": "CLOSE_T_AFTER_RETURN",
            "initial_nav": 1.0,
            "cash_daily_return": 0.0,
            "maximum_weight": 1.0,
        },
        "turnover": {},
        "costs": {"model": "ZERO", "linear_rate_per_dollar_traded": 0.0},
        "artifacts": {"manifest_filename": "run_manifest.json"},
        "limitations": ["survivor-conditioned synthetic fixture"],
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = ResearchConfig.load(config_path)
    provider = FrozenCsvReturnProvider(returns_path)
    universe = SurvivorPanelUniverseBuilder(provider.assets, UniverseRules(1))
    return dates, returns_path, config, provider, universe


class TestManifests(unittest.TestCase):
    def test_every_engine_run_writes_required_manifest_and_universe_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates, returns_path, config, provider, universe = _setup_fixture(root)
            result = BacktestEngine(
                returns=provider,
                universe_builder=universe,
                config=config,
            ).run(
                strategy_name="all_a",
                target_policy=lambda context: pd.Series({"A": 1.0}),
                rebalance_dates=[dates[1]],
                artifact_dir=root / "run",
                input_paths={"config": config.path, "returns": returns_path},
                repository_root=REPOSITORY_ROOT,
            )
            manifest_path = Path(result.artifact_paths["run_manifest"])
            universe_path = Path(result.artifact_paths["universe_snapshots"])
            self.assertTrue(manifest_path.exists())
            self.assertTrue(universe_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in (
                "git",
                "configuration",
                "inputs",
                "environment",
                "execution_convention",
                "universe",
                "run_timestamp_utc",
                "artifact_locations",
            ):
                self.assertIn(field, manifest)
            self.assertEqual(manifest["configuration"]["canonical_sha256"], config.sha256)
            self.assertEqual(manifest["universe"]["mode"], "SURVIVOR_PANEL")
            self.assertTrue(manifest["universe"]["survivor_conditioned"])
            self.assertFalse(manifest["universe"]["survivorship_bias_free"])
            self.assertEqual(manifest["execution_convention"]["execution"],
                             "target executes at close t after row t is earned")

    def test_same_config_and_inputs_reproduce_identical_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates, returns_path, config, provider, universe = _setup_fixture(root)

            def run(name):
                return BacktestEngine(
                    returns=provider,
                    universe_builder=universe,
                    config=config,
                ).run(
                    strategy_name="all_a",
                    target_policy=lambda context: pd.Series({"A": 1.0}),
                    rebalance_dates=[dates[1]],
                    artifact_dir=root / name,
                    input_paths={"config": config.path, "returns": returns_path},
                    repository_root=REPOSITORY_ROOT,
                )

            first, second = run("first"), run("second")
            pd.testing.assert_frame_equal(first.daily_ledger, second.daily_ledger)
            pd.testing.assert_frame_equal(
                pd.read_csv(first.artifact_paths["daily_holdings"], index_col=0),
                pd.read_csv(second.artifact_paths["daily_holdings"], index_col=0),
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(first.artifact_paths["rebalance_details"]),
                pd.read_csv(second.artifact_paths["rebalance_details"]),
            )
            self.assertEqual(first.manifest["configuration"], second.manifest["configuration"])
            self.assertEqual(first.manifest["inputs"], second.manifest["inputs"])

    def test_research_artifacts_cannot_overwrite_source_or_results(self):
        for directory in ("data", "results", "src"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                validate_artifact_directory(REPOSITORY_ROOT / directory, REPOSITORY_ROOT)
        allowed = validate_artifact_directory(
            REPOSITORY_ROOT / "artifacts" / "accounting" / "run",
            REPOSITORY_ROOT,
        )
        self.assertEqual(
            allowed,
            (REPOSITORY_ROOT / "artifacts" / "accounting" / "run").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
