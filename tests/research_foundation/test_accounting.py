"""Hand-calculated tests for drift and NAV accounting."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.backtest.accounting import (  # noqa: E402
    apply_close_to_close_return,
    turnover_from_weights,
)
from robust_portfolio.backtest.state import PortfolioState  # noqa: E402


class TestAccounting(unittest.TestCase):
    def test_two_asset_drift(self):
        state = PortfolioState(
            timestamp=pd.Timestamp("2020-01-01"),
            nav=1.0,
            holdings=pd.Series({"A": 0.5, "B": 0.5}),
            cash=0.0,
        )
        drifted = apply_close_to_close_return(
            state,
            pd.Series({"A": 0.10, "B": 0.0}),
            return_date="2020-01-02",
        )
        self.assertAlmostEqual(drifted.nav, 1.05, places=15)
        self.assertAlmostEqual(drifted.weights["A"], 0.55 / 1.05, places=15)
        self.assertAlmostEqual(drifted.weights["B"], 0.50 / 1.05, places=15)

    def test_no_daily_reset_between_rebalances(self):
        state = PortfolioState(
            timestamp=pd.Timestamp("2020-01-01"),
            nav=1.0,
            holdings=pd.Series({"A": 0.5, "B": 0.5}),
            cash=0.0,
        )
        for date in ("2020-01-02", "2020-01-03"):
            state = apply_close_to_close_return(
                state,
                pd.Series({"A": 0.10, "B": 0.0}),
                return_date=date,
            )
        buy_and_hold_nav = 0.5 * 1.1 * 1.1 + 0.5
        daily_reset_nav = 1.05 * 1.05
        self.assertAlmostEqual(state.nav, buy_and_hold_nav, places=15)
        self.assertNotAlmostEqual(state.nav, daily_reset_nav, places=12)

    def test_full_nav_identity_for_manual_path(self):
        state = PortfolioState(
            timestamp=pd.Timestamp("2020-01-01"),
            nav=2.0,
            holdings=pd.Series({"A": 1.2, "B": 0.6}),
            cash=0.2,
        )
        path = [
            ("2020-01-02", {"A": 0.10, "B": -0.20}),
            ("2020-01-03", {"A": -0.50, "B": 0.25}),
        ]
        state = apply_close_to_close_return(state, pd.Series(path[0][1]), return_date=path[0][0])
        self.assertAlmostEqual(state.nav, 1.32 + 0.48 + 0.2, places=15)
        state = apply_close_to_close_return(state, pd.Series(path[1][1]), return_date=path[1][0])
        expected = 1.32 * 0.5 + 0.48 * 1.25 + 0.2
        self.assertAlmostEqual(state.nav, expected, places=15)
        self.assertAlmostEqual(state.nav, state.holdings.sum() + state.cash, places=15)

    def test_turnover_definitions_are_exact(self):
        pre = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.2, "B": 0.8})
        gross, one_way, trades = turnover_from_weights(pre, target)
        self.assertAlmostEqual(gross, 0.8, places=15)
        self.assertAlmostEqual(one_way, 0.4, places=15)
        pd.testing.assert_series_equal(trades, pd.Series({"A": -0.4, "B": 0.4}))

    def test_missing_return_for_held_asset_fails(self):
        state = PortfolioState(
            timestamp=pd.Timestamp("2020-01-01"),
            nav=1.0,
            holdings=pd.Series({"A": 1.0, "B": 0.0}),
            cash=0.0,
        )
        with self.assertRaisesRegex(ValueError, "Missing return for held assets"):
            apply_close_to_close_return(
                state,
                pd.Series({"A": np.nan, "B": 0.0}),
                return_date="2020-01-02",
            )


if __name__ == "__main__":
    unittest.main()
