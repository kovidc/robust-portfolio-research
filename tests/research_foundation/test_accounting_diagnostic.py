"""End-to-end controlled replay of the exact frozen legacy targets."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from run_accounting_diagnostic import DEFAULT_CONFIG, run_diagnostic  # noqa: E402


class TestAccountingDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="accounting_diagnostic_test_")
        cls.output = Path(cls.temporary_directory.name) / "diagnostic"
        cls.summary, cls.diagnostic, cls.manifest = run_diagnostic(
            DEFAULT_CONFIG, cls.output
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_diagnostic_has_required_warning_and_all_strategies(self):
        self.assertEqual(
            self.diagnostic["result_label"],
            "ACCOUNTING DIAGNOSTIC — NOT FINAL RESEARCH RESULTS",
        )
        self.assertEqual(
            set(self.diagnostic["strategies"]),
            {"equal_weight", "classical_markowitz", "robust_markowitz"},
        )
        self.assertIn("not a corrected optimizer", self.diagnostic["interpretation_warning"])

    def test_legacy_values_are_preserved_and_corrected_path_is_distinct(self):
        expected_legacy = {
            "equal_weight": (1.1636623544546851, 0.024443153991022284),
            "classical_markowitz": (0.37954613271910076, 0.4611757664374712),
            "robust_markowitz": (0.14782040824274367, 0.5189614678929727),
        }
        expected_corrected = {
            "equal_weight": (1.1588457577088622, 0.0254045519339145),
            "classical_markowitz": (0.3630471219860818, 0.4615407512740873),
            "robust_markowitz": (0.0954243220624406, 0.5193344494508176),
        }
        for strategy, (expected_return, expected_turnover) in expected_legacy.items():
            row = self.summary.loc[strategy]
            self.assertAlmostEqual(
                row["legacy_daily_reset_cumulative_return"], expected_return, places=14
            )
            self.assertAlmostEqual(
                row["legacy_reported_recurring_one_way_turnover"],
                expected_turnover,
                places=14,
            )
            self.assertNotAlmostEqual(
                row["corrected_close_timing_cumulative_return"], expected_return, places=6
            )
            self.assertAlmostEqual(
                row["corrected_close_timing_cumulative_return"],
                expected_corrected[strategy][0],
                delta=1e-10,
            )
            self.assertAlmostEqual(
                row["corrected_recurring_one_way_turnover"],
                expected_corrected[strategy][1],
                delta=1e-10,
            )
            self.assertNotAlmostEqual(
                row["timing_convention_cumulative_return_difference"], 0.0, places=8
            )
            self.assertAlmostEqual(row["initial_formation_gross_traded_fraction"], 1.0)
            self.assertAlmostEqual(row["initial_formation_one_way_turnover"], 0.5)
            self.assertEqual(row["corrected_total_transaction_cost"], 0.0)

    def test_each_strategy_persists_35_universe_snapshots_and_run_manifest(self):
        for strategy in self.summary.index:
            strategy_dir = self.output / strategy
            snapshots = json.loads(
                (strategy_dir / "universe_snapshots.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (strategy_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(snapshots["snapshots"]), 35)
            self.assertTrue(all(item["survivor_conditioned"] for item in snapshots["snapshots"]))
            self.assertTrue(
                all(not item["survivorship_bias_free"] for item in snapshots["snapshots"])
            )
            self.assertEqual(manifest["universe"]["mode"], "SURVIVOR_PANEL")
            self.assertEqual(
                manifest["result_label"],
                "ACCOUNTING DIAGNOSTIC — NOT FINAL RESEARCH RESULTS",
            )

    def test_combined_manifest_contains_all_frozen_inputs_and_artifacts(self):
        self.assertIn("git", self.manifest)
        self.assertIn("configuration", self.manifest)
        self.assertEqual(len(self.manifest["inputs"]), 8)
        self.assertIn("accounting_diagnostic", self.manifest["artifact_locations"])
        self.assertTrue((self.output / "accounting_diagnostic_summary.csv").exists())
        self.assertTrue((self.output / "accounting_diagnostic.json").exists())
        self.assertTrue((self.output / "accounting_diagnostic_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
