"""Automated as-of and universe-eligibility tests."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.data import (  # noqa: E402
    FrozenCsvReturnProvider,
    PointInTimeDataUnavailable,
    PointInTimeUniverseBuilder,
    SurvivorPanelUniverseBuilder,
    UniverseRules,
)


class TestDataAndUniverse(unittest.TestCase):
    def test_return_panel_excludes_as_of_and_future_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "returns.csv"
            dates = pd.date_range("2020-01-01", periods=5, freq="D")
            pd.DataFrame({"A": range(5)}, index=dates).to_csv(path)
            panel = FrozenCsvReturnProvider(path).panel(as_of=dates[3])
            self.assertEqual(panel.as_of, dates[3])
            self.assertEqual(panel.last_observation, dates[2])
            self.assertTrue(bool((panel.values.index < dates[3]).all()))
            defensive_copy = panel.values
            defensive_copy.iloc[0, 0] = 999
            self.assertNotEqual(panel.values.iloc[0, 0], 999)

    def test_estimator_visibility_contract_rejects_contemporaneous_panel(self):
        from robust_portfolio.data.schemas import ReturnPanel

        as_of = pd.Timestamp("2020-01-03")
        invalid = pd.DataFrame(
            {"A": [0.1, 99.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        with self.assertRaisesRegex(ValueError, "strictly before as_of"):
            ReturnPanel(as_of=as_of, _values=invalid, source_sha256="synthetic")

    def test_universe_as_of_and_required_history(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        metadata = pd.DataFrame(
            {
                "asset": ["A", "B"],
                "listing_date": [dates[0], dates[2]],
                "inactive_date": [pd.NaT, pd.NaT],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "returns.csv"
            pd.DataFrame(
                {
                    "A": [0.01] * 6,
                    "B": [None, None, 0.02, 0.02, 0.02, 0.02],
                },
                index=dates,
            ).to_csv(path)
            provider = FrozenCsvReturnProvider(path)
            builder = PointInTimeUniverseBuilder(metadata, UniverseRules(2))

            early = builder.snapshot(provider.panel(as_of=dates[2]))
            self.assertEqual(early.eligible_assets, ("A",))
            self.assertEqual(early.exclusion_reasons["B"], "NOT_LISTED_AS_OF")

            one_b_return = builder.snapshot(provider.panel(as_of=dates[3]))
            self.assertEqual(one_b_return.eligible_assets, ("A",))
            self.assertTrue(one_b_return.exclusion_reasons["B"].startswith("INSUFFICIENT_HISTORY"))

            eligible = builder.snapshot(provider.panel(as_of=dates[4]))
            self.assertEqual(eligible.eligible_assets, ("A", "B"))

    def test_future_data_cannot_change_earlier_universe_snapshot(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        metadata = pd.DataFrame(
            {"asset": ["A", "B"], "listing_date": [dates[0], dates[2]]}
        )
        base = pd.DataFrame(
            {"A": [0.01] * 5, "B": [None, None, 0.02, 0.02, 0.02]},
            index=dates[:5],
        )
        extended = pd.concat(
            [base, pd.DataFrame({"A": [9.0], "B": [9.0]}, index=[dates[5]])]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path, extended_path = root / "base.csv", root / "extended.csv"
            base.to_csv(base_path)
            extended.to_csv(extended_path)
            builder = PointInTimeUniverseBuilder(metadata, UniverseRules(2))
            base_snapshot = builder.snapshot(
                FrozenCsvReturnProvider(base_path).panel(as_of=dates[4])
            )
            extended_snapshot = builder.snapshot(
                FrozenCsvReturnProvider(extended_path).panel(as_of=dates[4])
            )
            self.assertEqual(base_snapshot.eligible_assets, extended_snapshot.eligible_assets)
            self.assertEqual(
                dict(base_snapshot.exclusion_reasons),
                dict(extended_snapshot.exclusion_reasons),
            )
            self.assertEqual(
                dict(base_snapshot.history_observations),
                dict(extended_snapshot.history_observations),
            )

    def test_survivor_panel_is_honestly_labeled(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "returns.csv"
            pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates).to_csv(path)
            provider = FrozenCsvReturnProvider(path)
            snapshot = SurvivorPanelUniverseBuilder(
                provider.assets, UniverseRules(1)
            ).snapshot(provider.panel(as_of=dates[2]))
            self.assertEqual(snapshot.mode, "SURVIVOR_PANEL")
            self.assertTrue(snapshot.survivor_conditioned)
            self.assertFalse(snapshot.survivorship_bias_free)
            self.assertIn("not survivorship-bias-free", snapshot.limitation)

    def test_point_in_time_mode_fails_without_required_metadata(self):
        with self.assertRaisesRegex(PointInTimeDataUnavailable, "not present"):
            PointInTimeUniverseBuilder(None, UniverseRules(2))


if __name__ == "__main__":
    unittest.main()
