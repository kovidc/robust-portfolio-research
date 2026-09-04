#!/usr/bin/env python3
"""Replay frozen legacy targets through the corrected research foundation accounting engine."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from robust_portfolio.backtest import (  # noqa: E402
    BacktestEngine,
    CloseAfterReturnExecutionConvention,
    cost_model_from_config,
)
from robust_portfolio.config import ResearchConfig  # noqa: E402
from robust_portfolio.data import (  # noqa: E402
    FrozenCsvReturnProvider,
    SurvivorPanelUniverseBuilder,
    UniverseRules,
)
from robust_portfolio.reporting.manifests import build_run_manifest, write_manifest  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "research_foundation.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "accounting_diagnostic" / "run"
WEIGHT_PATHS = {
    "equal_weight": REPOSITORY_ROOT / "outputs" / "weights_equal_weight.csv",
    "classical_markowitz": REPOSITORY_ROOT / "outputs" / "weights_classical_markowitz.csv",
    "robust_markowitz": REPOSITORY_ROOT / "outputs" / "weights_robust_markowitz.csv",
}


class FrozenTargetPolicy:
    """Return the exact stored legacy target for each execution date."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.weights = pd.read_csv(self.path, index_col=0, parse_dates=True).astype(float)

    def __call__(self, context):
        if context.execution_date not in self.weights.index:
            raise KeyError(f"No frozen target exists for {context.execution_date}.")
        return self.weights.loc[context.execution_date].copy()


def _quarterly_drift_with_target_effective_before_rebalance_return(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    targets: pd.DataFrame,
) -> float:
    """Timing counterfactual: buy each target before its date-t return is earned."""
    growth = 1.0
    for position, rebalance_date in enumerate(rebalance_dates):
        if position + 1 < len(rebalance_dates):
            next_date = rebalance_dates[position + 1]
            period = returns.loc[
                (returns.index >= rebalance_date) & (returns.index < next_date)
            ]
        else:
            period = returns.loc[returns.index >= rebalance_date]
        target = targets.loc[rebalance_date].reindex(returns.columns, fill_value=0.0)
        asset_growth = (1.0 + period).prod(axis=0)
        growth *= float((target * asset_growth).sum())
    return growth - 1.0


def run_diagnostic(config_path: Path | str, output_dir: Path | str):
    config = ResearchConfig.load(config_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    returns_path = config.resolve_repository_path(REPOSITORY_ROOT, "returns_path")
    rebalance_path = config.resolve_repository_path(REPOSITORY_ROOT, "rebalance_dates_path")
    provider = FrozenCsvReturnProvider(returns_path)
    rebalance_dates = pd.DatetimeIndex(
        pd.read_csv(rebalance_path, parse_dates=["rebalance_date"])["rebalance_date"]
    )
    legacy_returns_path = REPOSITORY_ROOT / "outputs" / "portfolio_returns.csv"
    legacy_turnover_path = REPOSITORY_ROOT / "outputs" / "turnover.csv"
    legacy_returns = pd.read_csv(legacy_returns_path, index_col=0, parse_dates=True)
    legacy_turnover = pd.read_csv(legacy_turnover_path, index_col=0, parse_dates=True)
    full_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).astype(float)

    rules = UniverseRules(
        required_history_observations=int(
            config.section("universe")["required_history_observations"]
        ),
        require_complete_required_window=bool(
            config.section("universe")["require_complete_required_window"]
        ),
    )
    universe_builder = SurvivorPanelUniverseBuilder(provider.assets, rules)
    cost_model = cost_model_from_config(config.section("costs"))
    results = {}
    summary_rows = []

    for strategy, weights_path in WEIGHT_PATHS.items():
        strategy_output = output / strategy
        target_policy = FrozenTargetPolicy(weights_path)
        engine = BacktestEngine(
            returns=provider,
            universe_builder=universe_builder,
            config=config,
            cost_model=cost_model,
            execution_convention=CloseAfterReturnExecutionConvention(),
        )
        input_paths = {
            "configuration": config.path,
            "frozen_returns": returns_path,
            "frozen_rebalance_dates": rebalance_path,
            "frozen_legacy_targets": weights_path,
            "legacy_reported_returns": legacy_returns_path,
            "legacy_reported_turnover": legacy_turnover_path,
        }
        corrected = engine.run(
            strategy_name=strategy,
            target_policy=target_policy,
            rebalance_dates=rebalance_dates,
            artifact_dir=strategy_output,
            input_paths=input_paths,
            repository_root=REPOSITORY_ROOT,
        )

        legacy_cumulative = float((1.0 + legacy_returns[strategy]).prod() - 1.0)
        corrected_cumulative = float(
            corrected.daily_ledger.iloc[-1]["end_nav"]
            / config.section("backtest")["initial_nav"]
            - 1.0
        )
        legacy_recurring_turnover = float(legacy_turnover[strategy].dropna().mean())
        corrected_recurring = corrected.recurring_executions
        corrected_recurring_one_way = float(
            sum(item.one_way_turnover for item in corrected_recurring)
            / len(corrected_recurring)
        )
        corrected_recurring_gross = float(
            sum(item.gross_traded_fraction for item in corrected_recurring)
            / len(corrected_recurring)
        )
        targets = target_policy.weights
        pre_return_counterfactual = (
            _quarterly_drift_with_target_effective_before_rebalance_return(
                full_returns, rebalance_dates, targets
            )
        )
        timing_difference = corrected_cumulative - pre_return_counterfactual
        total_difference = corrected_cumulative - legacy_cumulative
        daily_reset_only_difference = pre_return_counterfactual - legacy_cumulative
        initial = corrected.initial_execution

        row = {
            "strategy": strategy,
            "legacy_daily_reset_cumulative_return": legacy_cumulative,
            "corrected_close_timing_cumulative_return": corrected_cumulative,
            "corrected_minus_legacy_cumulative_return": total_difference,
            "quarterly_drift_pre_return_timing_counterfactual": pre_return_counterfactual,
            "timing_convention_cumulative_return_difference": timing_difference,
            "drift_vs_daily_reset_difference_at_legacy_timing": daily_reset_only_difference,
            "legacy_reported_recurring_one_way_turnover": legacy_recurring_turnover,
            "corrected_recurring_one_way_turnover": corrected_recurring_one_way,
            "corrected_recurring_gross_traded_fraction": corrected_recurring_gross,
            "initial_formation_one_way_turnover": initial.one_way_turnover,
            "initial_formation_gross_traded_fraction": initial.gross_traded_fraction,
            "corrected_final_nav": float(corrected.daily_ledger.iloc[-1]["end_nav"]),
            "corrected_total_transaction_cost": float(
                sum(item.transaction_cost for item in corrected.executions)
            ),
        }
        summary_rows.append(row)
        results[strategy] = {
            **{key: value for key, value in row.items() if key != "strategy"},
            "strategy_artifacts": corrected.artifact_paths,
            "strategy_manifest": corrected.manifest,
        }

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    summary_path = output / "accounting_diagnostic_summary.csv"
    summary.to_csv(summary_path)
    diagnostic_path = output / "accounting_diagnostic.json"
    diagnostic = {
        "artifact_type": "ACCOUNTING_DIAGNOSTIC",
        "result_label": config.section("experiment")["result_label"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_convention": CloseAfterReturnExecutionConvention().describe(),
        "turnover_definitions": config.section("turnover"),
        "cost_model": cost_model.describe(),
        "universe": {
            "mode": "SURVIVOR_PANEL",
            "survivor_conditioned": True,
            "survivorship_bias_free": False,
        },
        "strategies": results,
        "remaining_scientific_limitations": config.payload["limitations"],
        "interpretation_warning": (
            "This isolates accounting and execution timing while reusing legacy targets. "
            "It is not a corrected optimizer comparison or a final research result."
        ),
    }
    with diagnostic_path.open("w", encoding="utf-8") as file:
        json.dump(diagnostic, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")

    combined_inputs = {
        "configuration": config.path,
        "frozen_returns": returns_path,
        "frozen_rebalance_dates": rebalance_path,
        "legacy_reported_returns": legacy_returns_path,
        "legacy_reported_turnover": legacy_turnover_path,
        **{f"legacy_targets_{name}": path for name, path in WEIGHT_PATHS.items()},
    }
    combined_manifest_path = output / "accounting_diagnostic_manifest.json"
    combined_manifest = build_run_manifest(
        repository_root=REPOSITORY_ROOT,
        config=config,
        input_paths=combined_inputs,
        artifact_paths={
            "accounting_diagnostic": str(diagnostic_path),
            "accounting_diagnostic_summary": str(summary_path),
            "combined_manifest": str(combined_manifest_path),
            **{f"strategy_{name}": str(output / name) for name in WEIGHT_PATHS},
        },
        execution_convention=CloseAfterReturnExecutionConvention().describe(),
        universe_mode="SURVIVOR_PANEL",
        survivor_conditioned=True,
        survivorship_bias_free=False,
        strategy_name="combined_frozen_legacy_target_accounting_diagnostic",
        cost_model=cost_model.describe(),
        result_label=config.section("experiment")["result_label"],
    )
    write_manifest(combined_manifest_path, combined_manifest)
    return summary, diagnostic, combined_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the corrected-accounting diagnostic on frozen legacy targets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary, _, _ = run_diagnostic(args.config, args.output_dir)
    print("ACCOUNTING DIAGNOSTIC — NOT FINAL RESEARCH RESULTS")
    print(summary.to_string(float_format=lambda value: f"{value:.10f}"))
    print(f"\nArtifacts: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
