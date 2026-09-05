"""Persist daily holdings, weights, trades, costs, and run metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def validate_artifact_directory(artifact_dir: Path | str, repository_root: Path) -> Path:
    output = Path(artifact_dir).expanduser().resolve()
    repository = repository_root.resolve()
    if output == repository:
        raise ValueError("The repository root cannot be an artifact directory.")
    if repository in output.parents:
        artifacts_root = (repository / "artifacts").resolve()
        if artifacts_root not in output.parents:
            raise ValueError("Repository-local research outputs must stay under artifacts/.")
    return output


def _state_frames(result):
    dates = pd.DatetimeIndex([state.timestamp for state in result.daily_states], name="date")
    holdings = pd.DataFrame(
        [state.holdings for state in result.daily_states],
        index=dates,
    ).fillna(0.0)
    weights = pd.DataFrame(
        [state.weights for state in result.daily_states],
        index=dates,
    ).fillna(0.0)
    return holdings, weights


def _execution_summary(result) -> pd.DataFrame:
    records = []
    for execution in result.executions:
        records.append(
            {
                "execution_date": execution.execution_date,
                "initial_formation": execution.initial_formation,
                "pre_trade_nav": execution.pre_trade_state.nav,
                "transaction_cost": execution.transaction_cost,
                "post_trade_nav": execution.post_trade_state.nav,
                "gross_traded_fraction": execution.gross_traded_fraction,
                "one_way_turnover": execution.one_way_turnover,
                "pre_trade_cash": execution.pre_trade_state.cash,
                "post_trade_cash": execution.post_trade_state.cash,
            }
        )
    return pd.DataFrame(records).set_index("execution_date")


def _rebalance_details(result) -> pd.DataFrame:
    records = []
    for execution in result.executions:
        assets = execution.pre_trade_state.holdings.index.union(
            execution.target_weights.index, sort=False
        )
        pre_holdings = execution.pre_trade_state.holdings.reindex(assets, fill_value=0.0)
        post_holdings = execution.post_trade_state.holdings.reindex(assets, fill_value=0.0)
        for asset in assets:
            records.append(
                {
                    "execution_date": execution.execution_date,
                    "asset": asset,
                    "initial_formation": execution.initial_formation,
                    "pre_trade_holding": float(pre_holdings[asset]),
                    "pre_trade_weight": float(execution.pre_trade_state.weights[asset]),
                    "target_weight": float(execution.target_weights[asset]),
                    "weight_trade": float(execution.weight_trades[asset]),
                    "dollar_trade": float(execution.dollar_trades[asset]),
                    "post_trade_holding": float(post_holdings[asset]),
                    "post_trade_weight": float(execution.post_trade_state.weights[asset]),
                }
            )
    return pd.DataFrame(records)


def write_backtest_artifacts(result, *, artifact_dir, repository_root: Path) -> dict[str, str]:
    output = validate_artifact_directory(artifact_dir, repository_root)
    output.mkdir(parents=True, exist_ok=True)
    holdings, weights = _state_frames(result)
    execution_summary = _execution_summary(result)
    details = _rebalance_details(result)

    frames = {
        "daily_ledger": (result.daily_ledger, output / "daily_ledger.csv"),
        "daily_holdings": (holdings, output / "daily_holdings.csv"),
        "daily_weights": (weights, output / "daily_weights.csv"),
        "executions": (execution_summary, output / "executions.csv"),
        "rebalance_details": (details, output / "rebalance_details.csv"),
    }
    paths = {}
    for name, (frame, path) in frames.items():
        frame.to_csv(path)
        paths[name] = str(path)

    universe_path = output / "universe_snapshots.json"
    with universe_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "result_label": result.result_label,
                "strategy": result.strategy_name,
                "snapshots": [snapshot.to_dict() for snapshot in result.universe_snapshots],
            },
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")
    paths["universe_snapshots"] = str(universe_path)
    return paths
