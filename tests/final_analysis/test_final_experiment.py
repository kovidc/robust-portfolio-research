import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robust_portfolio.data.providers import sha256_file
from robust_portfolio.research.final_analysis import run_final_analysis
from robust_portfolio.research.final_analysis_configuration import FinalAnalysisConfig


class TestFinalExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "configs" / "final_analysis.json"
        cls.config = FinalAnalysisConfig.load(cls.config_path)
        cls.result = run_final_analysis(cls.config_path, repository_root=ROOT)
        cls.output = Path(cls.result["output_directory"])

    def test_final_artifact_completeness(self):
        for number in range(1, 14):
            matches = list(self.output.glob(f"figure_{number:02d}_*.png"))
            self.assertEqual(len(matches), 1, number)
            self.assertGreater(matches[0].stat().st_size, 1000)
        for number in range(1, 5):
            self.assertEqual(len(list(self.output.glob(f"table_{number}_*.csv"))), 1)

    def test_common_ceiling_is_not_mislabeled_as_risk_matched(self):
        frame = pd.read_csv(self.output / "common_risk_ceiling_diagnostics.csv")
        self.assertEqual(set(frame["comparison_type"]), {"COMMON EX-ANTE RISK CEILING"})
        self.assertTrue((~frame["binding_indicator"]).any())
        self.assertTrue((frame["slack"] > 0.001).any())
        self.assertEqual(len(frame), 31 * 4 * 5)
        self.assertEqual(int((frame["status"] == "TARGET_NOT_FEASIBLE").sum()), 4)

    def test_target_nonattainment_and_all_failures_are_preserved(self):
        attainment = pd.read_csv(self.output / "target_attainment_diagnostics.csv")
        self.assertIn("TARGET_NOT_ATTAINABLE", set(attainment["status"]))
        self.assertTrue((attainment[attainment["status"] == "ATTAINED"]["risk_aversion"] >= 0).all())
        failures = pd.read_csv(self.output / "all_recorded_failures.csv")
        self.assertEqual(set(failures["status"]), {"FAILED_EXPLICITLY"})
        self.assertGreaterEqual(len(failures), 4)

    def test_perturbation_protocol_and_regime_clock(self):
        direct = pd.read_csv(self.output / "direct_robustness_observations.csv")
        bootstrap = direct[direct["perturbation_kind"] == "training_block_bootstrap"]
        self.assertEqual(bootstrap.groupby(["decision_date", "model"]).size().nunique(), 1)
        self.assertEqual(int(bootstrap.groupby(["decision_date", "model"]).size().iloc[0]), 24)
        self.assertEqual(set(direct[direct["perturbation_kind"] == "mean_standard_error_shock"]["perturbation"]), {"mu_+1s", "mu_-1s"})
        regimes = pd.read_csv(
            self.output / "regime_definitions.csv",
            parse_dates=["decision_date", "information_end"],
        )
        self.assertTrue((regimes["information_end"] < regimes["decision_date"]).all())
        self.assertGreater(regimes["information_end"].nunique(), 20)
        summary = pd.read_csv(self.output / "regime_summary.csv")
        absent = summary[summary["regime"] == "weak/cooling"]
        self.assertEqual(set(absent["status"]), {"NO_OBSERVATIONS"})

    def test_manifest_metadata_and_hashes_are_consistent(self):
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["configuration"]["canonical_sha256"], self.config.sha256)
        self.assertIn("not survivorship-bias-free", manifest["protocol"]["limitations"][0])
        self.assertEqual(manifest["counts"]["inference_replications"], 2000)
        self.assertEqual(manifest["counts"]["dsr_candidates"], 27)
        clustering = pd.read_csv(self.output / "clustering_date_diagnostics.csv")
        self.assertEqual(manifest["counts"]["clustering_model_date_attempts"], len(clustering))
        self.assertEqual(int((clustering["status"] == "FAILED_EXPLICITLY").sum()), 14)
        self.assertEqual(len(manifest["core_source"]["artifact_sha256"]), 19)
        core_manifest = json.loads(
            (Path(manifest["core_source"]["artifact_directory"]) / "run_manifest.json").read_text()
        )
        self.assertEqual(manifest["core_source"]["commit"], core_manifest["git"]["commit"])
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(sha256_file(Path(manifest["artifact_locations"][name])), expected)

    def test_final_analysis_configuration_is_unchanged(self):
        payload = self.config.payload
        direct = payload["direct_robustness"]
        self.assertEqual(
            direct["selected_outer_dates"],
            ["2018-04-02", "2020-04-01", "2022-10-03", "2025-10-01"],
        )
        self.assertEqual(
            (direct["bootstrap_replications"], direct["block_length_observations"], direct["seed"]),
            (24, 21, 91001),
        )
        self.assertEqual(payload["clone_experiment"]["relative_noise_standard_deviations"], [0.0, 0.01, 0.05])
        self.assertEqual(payload["clustering"]["correlation_thresholds"], [0.80, 0.90, 0.95, 0.975])
        self.assertEqual(
            (payload["inference"]["replications"], payload["inference"]["expected_block_length"], payload["inference"]["seed"]),
            (2000, 10.0, 36129),
        )
        self.assertEqual(payload["sensitivity"]["rho_multipliers"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(payload["sensitivity"]["kappa_multipliers"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(payload["sensitivity"]["maximum_weights"], [0.05, 0.10, 0.20])

    def test_report_values_match_generated_artifacts(self):
        headline = pd.read_csv(self.output / "table_2_headline_strategies.csv").set_index("strategy")
        expected = {
            "etf_equal_weight": (0.091646, 0.143193, 0.684461),
            "nominal_risk_10pct": (0.013416, 0.159963, 0.163922),
            "box_risk_10pct": (0.002593, 0.076084, 0.072121),
            "box_diagonal_risk_10pct": (0.009948, 0.062096, 0.190498),
            "ellipsoid_risk_10pct": (0.028323, 0.035795, 0.798193),
        }
        for strategy, values in expected.items():
            actual = headline.loc[strategy]
            self.assertAlmostEqual(actual["net_annualized_return"], values[0], delta=5e-6)
            self.assertAlmostEqual(actual["realized_volatility"], values[1], delta=5e-6)
            self.assertAlmostEqual(actual["provisional_zero_rf_sharpe"], values[2], delta=5e-6)

        attainment = pd.read_csv(self.output / "target_attainment_diagnostics.csv")
        counts = attainment.groupby(["model", "status"]).size().to_dict()
        self.assertEqual(counts[("nominal", "ATTAINED")], 31)
        self.assertEqual(counts[("box", "TARGET_NOT_ATTAINABLE")], 25)
        self.assertEqual(counts[("box_diagonal", "TARGET_NOT_ATTAINABLE")], 17)
        self.assertEqual(counts[("ellipsoid", "TARGET_NOT_ATTAINABLE")], 31)

        overlap = pd.read_csv(self.output / "overlapping_risk_comparison.csv").iloc[0]
        self.assertGreater(overlap["predicted_overlap_low"], overlap["predicted_overlap_high"])
        self.assertGreater(
            overlap["common_base_predicted_overlap_low"],
            overlap["common_base_predicted_overlap_high"],
        )
        dsr = pd.read_csv(self.output / "deflated_sharpe_diagnostics.csv")
        self.assertEqual(set(dsr["candidate_count"]), {27})
        self.assertLess(dsr["deflated_sharpe_probability"].max(), 0.95)

        report = (ROOT / "docs" / "FINAL_RESEARCH_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("configs/final_analysis.json", report)
        self.assertIn("These are not same-risk portfolios.", report)
        self.assertIn("zero-RF/provisional", report)
