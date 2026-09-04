"""Compact tables, figures, and JSON artifacts for the core experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import validate_artifact_directory


def write_json(path: Path, payload: dict) -> None:
    def compliant(value):
        if isinstance(value, dict):
            return {str(key): compliant(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [compliant(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return compliant(value.item())
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(compliant(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def create_core_figures(
    *,
    output: Path,
    metrics: pd.DataFrame,
    covariance_periods: pd.DataFrame,
    headline_wealth: pd.DataFrame,
    headline_cost_bps: float,
) -> dict[str, str]:
    paths = {}

    figure, axis = plt.subplots(figsize=(10, 5.5))
    subset = headline_wealth[headline_wealth["cost_bps"] == headline_cost_bps]
    for strategy, group in subset.groupby("strategy"):
        axis.plot(group["date"], group["gross_wealth"], alpha=0.45, linewidth=1.0)
        axis.plot(group["date"], group["net_wealth"], label=f"{strategy} net", linewidth=1.5)
    axis.set(title=f"Gross and net wealth ({headline_cost_bps:g} bp scenario)", ylabel="Wealth")
    axis.legend(fontsize=7, ncol=2)
    path = output / "figure_1_gross_vs_net_wealth.png"
    _save(figure, path)
    paths["figure_1_gross_vs_net_wealth"] = str(path)

    frontier = metrics[(metrics["cost_bps"] == headline_cost_bps) & metrics["target_risk"].notna()]
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for model, group in frontier.groupby("robust_set"):
        axis.plot(
            group["realized_volatility"], group["net_annualized_return"],
            marker="o", label=model,
        )
    axis.set(xlabel="Realized annualized volatility", ylabel="Net annualized return", title="Risk-matched realized frontier")
    axis.legend(fontsize=8)
    path = output / "figure_2_realized_risk_return.png"
    _save(figure, path)
    paths["figure_2_realized_risk_return"] = str(path)

    figure, axis = plt.subplots(figsize=(7, 5.5))
    for estimator, group in covariance_periods.groupby("estimator"):
        axis.scatter(
            group["predicted_equal_weight_volatility"],
            group["realized_equal_weight_volatility"],
            s=18, alpha=0.65, label=estimator,
        )
    values = covariance_periods[["predicted_equal_weight_volatility", "realized_equal_weight_volatility"]].to_numpy()
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Predicted annualized volatility", ylabel="Later realized volatility", title="Predicted versus realized volatility")
    axis.legend(fontsize=8)
    path = output / "figure_3_predicted_vs_realized_volatility.png"
    _save(figure, path)
    paths["figure_3_predicted_vs_realized_volatility"] = str(path)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for estimator, group in covariance_periods.groupby("estimator"):
        ordered = group.sort_values("forecast_date")
        axis.plot(
            ordered["forecast_date"], ordered["oos_gaussian_nll_per_asset"].cumsum(),
            label=estimator,
        )
    axis.set(title="Cumulative out-of-sample covariance loss", ylabel="Cumulative Gaussian NLL per asset")
    axis.legend(fontsize=8)
    path = output / "figure_4_covariance_forecast_loss.png"
    _save(figure, path)
    paths["figure_4_covariance_forecast_loss"] = str(path)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for strategy, group in metrics.groupby("strategy"):
        if strategy not in set(headline_wealth["strategy"]):
            continue
        ordered = group.sort_values("cost_bps")
        axis.plot(
            ordered["recurring_one_way_turnover"], ordered["cost_drag_final_wealth"],
            marker="o", alpha=0.7, label=strategy,
        )
    axis.set(xlabel="Average recurring one-way turnover", ylabel="Final wealth cost drag", title="Turnover and implementation drag")
    axis.legend(fontsize=7)
    path = output / "figure_5_turnover_cost_drag.png"
    _save(figure, path)
    paths["figure_5_turnover_cost_drag"] = str(path)
    return paths


def prepare_output_directory(path: Path | str, repository_root: Path) -> Path:
    output = validate_artifact_directory(path, repository_root)
    output.mkdir(parents=True, exist_ok=True)
    return output
