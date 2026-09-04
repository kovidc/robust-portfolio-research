"""Execution, timing, cost, and state-continuity tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_portfolio.backtest import (  # noqa: E402
    BacktestEngine,
    LinearCostModel,
    PortfolioState,
    ZeroCostModel,
    cost_model_from_config,
    execute_target,
)
from robust_portfolio.config import ResearchConfig  # noqa: E402
from robust_portfolio.data import (  # noqa: E402
    FrozenCsvReturnProvider,
    SurvivorPanelUniverseBuilder,
    UniverseRules,
)


def _write_config(directory: Path, *, cost_model="ZERO", cost_rate=0.0) -> ResearchConfig:
    payload = {
        "schema_version": 1,
        "experiment": {"name": "toy", "result_label": "RESEARCH FOUNDATION TEST"},
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
        "costs": {
            "model": cost_model,
            "linear_rate_per_dollar_traded": cost_rate,
        },
        "artifacts": {"manifest_filename": "run_manifest.json"},
        "limitations": ["synthetic fixture"],
    }
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ResearchConfig.load(path)


class TestExecution(unittest.TestCase):
    def test_zero_cost_rebalance_is_self_financing(self):
        pre = PortfolioState(
            timestamp=pd.Timestamp("2020-01-02"),
            nav=1.0,
            holdings=pd.Series({"A": 0.6, "B": 0.4}),
            cash=0.0,
        )
        result = execute_target(
            pre,
            pd.Series({"A": 0.25, "B": 0.75}),
            execution_date="2020-01-02",
            cost_model=ZeroCostModel(),
        )
        self.assertEqual(result.transaction_cost, 0.0)
        self.assertAlmostEqual(result.post_trade_state.nav, pre.nav, places=15)
        residual = (
            result.dollar_trades.sum()
            + result.post_trade_state.cash
            - result.pre_trade_state.cash
            + result.transaction_cost
        )
        self.assertAlmostEqual(residual, 0.0, places=15)

    def test_linear_cost_model_and_nav_reduction_are_exact(self):
        model = LinearCostModel(0.01)
        self.assertAlmostEqual(
            model.cost(pd.Series({"A": 100.0, "B": -40.0})), 1.4, places=15
        )
        pre = PortfolioState(
            timestamp=pd.Timestamp("2020-01-02"),
            nav=1.0,
            holdings=pd.Series({"A": 0.5, "B": 0.5}),
            cash=0.0,
        )
        result = execute_target(
            pre,
            pd.Series({"A": 1.0, "B": 0.0}),
            execution_date="2020-01-02",
            cost_model=model,
        )
        expected_post_nav = 1.0 / 1.01
        self.assertAlmostEqual(result.post_trade_state.nav, expected_post_nav, places=11)
        self.assertAlmostEqual(result.transaction_cost, 1.0 - expected_post_nav, places=11)
        self.assertAlmostEqual(
            result.transaction_cost,
            model.cost(result.dollar_trades),
            places=15,
        )

    def test_cost_model_is_selected_from_configuration(self):
        model = cost_model_from_config(
            {"model": "LINEAR", "linear_rate_per_dollar_traded": 0.0025}
        )
        self.assertIsInstance(model, LinearCostModel)
        self.assertAlmostEqual(model.cost(pd.Series({"A": 2.0})), 0.005)
        zero = cost_model_from_config(
            {"model": "ZERO", "linear_rate_per_dollar_traded": 0.0}
        )
        self.assertIsInstance(zero, ZeroCostModel)

    def test_initial_trade_is_separate_from_recurring_turnover(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_config(root)
            dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
            returns_path = root / "returns.csv"
            pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates).to_csv(
                returns_path
            )
            provider = FrozenCsvReturnProvider(returns_path)
            universe = SurvivorPanelUniverseBuilder(provider.assets, UniverseRules(1))

            def targets(context):
                return (
                    pd.Series({"A": 1.0, "B": 0.0})
                    if context.execution_date == dates[1]
                    else pd.Series({"A": 0.0, "B": 1.0})
                )

            result = BacktestEngine(
                returns=provider,
                universe_builder=universe,
                config=config,
            ).run(
                strategy_name="toy",
                target_policy=targets,
                rebalance_dates=dates[1:],
                artifact_dir=root / "artifacts",
                input_paths={"config": config.path, "returns": returns_path},
                repository_root=REPOSITORY_ROOT,
            )
            self.assertEqual(len(result.executions), 2)
            self.assertTrue(result.initial_execution.initial_formation)
            self.assertAlmostEqual(result.initial_execution.gross_traded_fraction, 1.0)
            self.assertAlmostEqual(result.initial_execution.one_way_turnover, 0.5)
            self.assertEqual(len(result.recurring_executions), 1)
            self.assertAlmostEqual(result.recurring_executions[0].gross_traded_fraction, 2.0)
            self.assertAlmostEqual(result.recurring_executions[0].one_way_turnover, 1.0)

    def test_execution_timing_old_portfolio_gets_date_t_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_config(root)
            dates = pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            )
            returns_path = root / "returns.csv"
            pd.DataFrame(
                {"A": [0.0, 0.50, 1.00, 0.0], "B": [0.0, 0.0, 0.0, 0.10]},
                index=dates,
            ).to_csv(returns_path)
            provider = FrozenCsvReturnProvider(returns_path)
            universe = SurvivorPanelUniverseBuilder(provider.assets, UniverseRules(1))
            observed_panels = []
            observed_decision_nav = []

            def targets(context):
                observed_panels.append(context.returns.values)
                observed_decision_nav.append(context.decision_state.nav)
                if context.execution_date == dates[1]:
                    return pd.Series({"A": 1.0, "B": 0.0})
                return pd.Series({"A": 0.0, "B": 1.0})

            result = BacktestEngine(
                returns=provider,
                universe_builder=universe,
                config=config,
            ).run(
                strategy_name="timing",
                target_policy=targets,
                rebalance_dates=[dates[1], dates[2]],
                artifact_dir=root / "artifacts",
                input_paths={"config": config.path, "returns": returns_path},
                repository_root=REPOSITORY_ROOT,
            )
            # First target executes after the +50% row, so initial cash does not earn it.
            self.assertAlmostEqual(result.daily_ledger.loc[dates[1], "end_nav"], 1.0)
            # Old A position earns +100% on the second execution date before switching to B.
            self.assertAlmostEqual(result.executions[1].pre_trade_state.nav, 2.0)
            self.assertAlmostEqual(result.daily_ledger.loc[dates[2], "end_nav"], 2.0)
            # New B position starts with the following +10% row.
            self.assertAlmostEqual(result.daily_ledger.loc[dates[3], "end_nav"], 2.2)
            self.assertNotIn(dates[1], observed_panels[0].index)
            self.assertNotIn(dates[2], observed_panels[1].index)
            self.assertEqual(observed_panels[1].index[-1], dates[1])
            self.assertAlmostEqual(observed_decision_nav[1], 1.0)

    def test_rebalance_state_continuity(self):
        pre = PortfolioState(
            timestamp=pd.Timestamp("2020-01-02"),
            nav=3.0,
            holdings=pd.Series({"A": 1.5, "B": 1.0}),
            cash=0.5,
        )
        execution = execute_target(
            pre,
            pd.Series({"A": 0.2, "B": 0.6}),
            execution_date="2020-01-02",
            cost_model=ZeroCostModel(),
        )
        self.assertAlmostEqual(execution.post_trade_state.nav, 3.0)
        self.assertAlmostEqual(execution.post_trade_state.holdings["A"], 0.6)
        self.assertAlmostEqual(execution.post_trade_state.holdings["B"], 1.8)
        self.assertAlmostEqual(execution.post_trade_state.cash, 0.6)
        self.assertAlmostEqual(
            execution.dollar_trades.sum()
            + execution.post_trade_state.cash
            - execution.pre_trade_state.cash,
            0.0,
            places=15,
        )


if __name__ == "__main__":
    unittest.main()
