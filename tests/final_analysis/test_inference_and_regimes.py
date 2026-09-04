from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robust_portfolio.inference import (
    bootstrap_headline_statistics, deflated_sharpe_probability,
    stationary_bootstrap_indices,
)
from robust_portfolio.research.regimes import classify_regimes


class TestInferenceAndRegimes(unittest.TestCase):
    def test_joint_indices_are_reproducible_and_pairing_is_preserved(self):
        first = stationary_bootstrap_indices(30, 20, 5.0, 7)
        second = stationary_bootstrap_indices(30, 20, 5.0, 7)
        np.testing.assert_array_equal(first, second)
        base = np.linspace(-0.01, 0.02, 60)
        returns = pd.DataFrame({"a": base, "b": base + 0.001})
        _, differences, indices = bootstrap_headline_statistics(
            returns, replications=100, expected_block_length=5, seed=9,
            confidence_level=0.95, annualization_factor=252,
            certainty_equivalent_risk_aversion=3.0, comparators=("a",),
        )
        np.testing.assert_allclose((returns.to_numpy()[indices, 1] - returns.to_numpy()[indices, 0]), 0.001)
        delta = differences[differences.metric == "delta_provisional_zero_rf_sharpe"]
        self.assertEqual(len(delta), 1)

    def test_sharpe_difference_bootstrap_reproducibility(self):
        generator = np.random.default_rng(2)
        returns = pd.DataFrame(generator.normal(0.0003, 0.01, size=(120, 3)), columns=list("abc"))
        one = bootstrap_headline_statistics(
            returns, replications=100, expected_block_length=6, seed=11,
            confidence_level=0.95, annualization_factor=252,
            certainty_equivalent_risk_aversion=3.0, comparators=("a",),
        )[1]
        two = bootstrap_headline_statistics(
            returns, replications=100, expected_block_length=6, seed=11,
            confidence_level=0.95, annualization_factor=252,
            certainty_equivalent_risk_aversion=3.0, comparators=("a",),
        )[1]
        pd.testing.assert_frame_equal(one, two)

    def test_dsr_synthetic_example(self):
        generator = np.random.default_rng(4)
        result = deflated_sharpe_probability(
            generator.normal(0.001, 0.01, 500), [0.1, 0.2, 0.3, 0.4]
        )
        self.assertEqual(result["candidate_count"], 4)
        self.assertGreaterEqual(result["deflated_sharpe_probability"], 0.0)
        self.assertLessEqual(result["deflated_sharpe_probability"], 1.0)

    def test_regime_classification_is_asof(self):
        index = pd.bdate_range("2018-01-01", periods=400)
        returns = pd.Series(np.linspace(-0.002, 0.003, 400), index=index)
        date = index[320]
        first = classify_regimes(returns, [date], trend_lookback=252, volatility_lookback=63)
        changed = returns.copy()
        changed.loc[changed.index >= date] = -0.5
        second = classify_regimes(changed, [date], trend_lookback=252, volatility_lookback=63)
        pd.testing.assert_frame_equal(first, second)
        self.assertLess(pd.Timestamp(first.loc[0, "information_end"]), date)
