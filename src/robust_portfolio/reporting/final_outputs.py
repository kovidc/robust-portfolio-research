"""Figures generated from saved final-analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def create_final_figures(output: Path) -> dict[str, str]:
    """Load saved CSV artifacts and regenerate all final figures."""
    paths = {}
    returns = pd.read_csv(output / "headline_net_returns.csv", index_col=0, parse_dates=True)
    wealth = (1.0 + returns).cumprod()
    figure, axis = plt.subplots(figsize=(10, 5.5))
    wealth.plot(ax=axis, linewidth=1.35)
    axis.set(title="Headline net wealth — 5 bp scenario", ylabel="Wealth", xlabel="")
    axis.legend(fontsize=7, ncol=2)
    paths["figure_01_net_cumulative_wealth"] = _save(figure, output / "figure_01_net_cumulative_wealth.png")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    drawdown = wealth.divide(wealth.cummax()).subtract(1.0)
    drawdown.plot(ax=axis, linewidth=1.2)
    axis.set(title="Headline drawdowns — 5 bp scenario", ylabel="Drawdown", xlabel="")
    axis.legend(fontsize=7, ncol=2)
    paths["figure_02_drawdowns"] = _save(figure, output / "figure_02_drawdowns.png")

    metrics = pd.read_csv(output / "core_strategy_metrics.csv")
    ceiling = metrics[(metrics["cost_bps"] == 5.0) & metrics["target_risk"].notna()]
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for robust_set, group in ceiling.groupby("robust_set"):
        ordered = group.sort_values("target_risk")
        axis.plot(ordered["realized_volatility"], ordered["net_annualized_return"], marker="o", label=robust_set)
    axis.set(title="Common ex-ante risk ceiling: realized outcomes", xlabel="Realized annualized volatility", ylabel="Net annualized return")
    axis.legend(fontsize=8)
    paths["figure_03_common_ceiling_realized_frontier"] = _save(figure, output / "figure_03_common_ceiling_realized_frontier.png")

    risk = pd.read_csv(output / "common_risk_ceiling_diagnostics.csv")
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for model, group in risk.groupby("model"):
        axis.scatter(group["requested_volatility"], group["attained_predicted_decision_volatility"], s=13, alpha=0.5, label=model)
    low, high = float(risk["requested_volatility"].min()), float(risk["requested_volatility"].max())
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set(title="Requested ceiling versus attained predicted risk", xlabel="Requested annualized ceiling", ylabel="Attained predicted annualized risk")
    axis.legend(fontsize=8)
    paths["figure_04_requested_vs_attained_risk"] = _save(figure, output / "figure_04_requested_vs_attained_risk.png")

    regime_periods = pd.read_csv(output / "regime_period_diagnostics.csv")
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for strategy, group in regime_periods.groupby("strategy"):
        axis.scatter(group["predicted_common_base_volatility"], group["realized_volatility"], s=10, alpha=0.35, label=strategy)
    values = regime_periods[["predicted_common_base_volatility", "realized_volatility"]].to_numpy()
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set(title="Predicted versus subsequent realized volatility", xlabel="Predicted annualized volatility", ylabel="Realized annualized volatility")
    axis.legend(fontsize=6, ncol=2)
    paths["figure_05_predicted_vs_realized_volatility"] = _save(figure, output / "figure_05_predicted_vs_realized_volatility.png")

    covariance = pd.read_csv(output / "covariance_forecast_periods.csv", parse_dates=["forecast_date"])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for estimator, group in covariance.groupby("estimator"):
        ordered = group.sort_values("forecast_date")
        axis.plot(ordered["forecast_date"], ordered["oos_gaussian_nll_per_asset"].cumsum(), label=estimator)
    axis.set(title="Cumulative out-of-sample covariance loss", ylabel="Cumulative Gaussian NLL per asset", xlabel="")
    axis.legend(fontsize=8)
    paths["figure_06_covariance_forecast_loss"] = _save(figure, output / "figure_06_covariance_forecast_loss.png")

    headline = pd.read_csv(output / "table_2_headline_strategies.csv")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(headline["recurring_one_way_turnover"], headline["net_annualized_return"], s=40)
    for _, row in headline.iterrows():
        axis.annotate(row["strategy"], (row["recurring_one_way_turnover"], row["net_annualized_return"]), fontsize=6)
    axis.set(title="Turnover versus net performance — 5 bp scenario", xlabel="Average recurring one-way turnover", ylabel="Net annualized return")
    paths["figure_07_turnover_vs_net_performance"] = _save(figure, output / "figure_07_turnover_vs_net_performance.png")

    direct = pd.read_csv(output / "direct_robustness_observations.csv")
    bootstrap = direct[direct["perturbation_kind"] == "training_block_bootstrap"]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    groups = [group["l1_weight_change"].dropna().to_numpy() for _, group in bootstrap.groupby("model")]
    labels = [name for name, _ in bootstrap.groupby("model")]
    axis.boxplot(groups, tick_labels=labels, showfliers=False)
    axis.set(title="Dependence-aware bootstrap weight sensitivity", ylabel=r"$\|w_b-w_0\|_1$")
    paths["figure_08_bootstrap_weight_sensitivity"] = _save(figure, output / "figure_08_bootstrap_weight_sensitivity.png")

    clones = pd.read_csv(output / "clone_diagnostics.csv")
    grouped = clones.groupby(["model", "relative_noise_standard_deviation"])[["l1_weight_change", "economic_exposure_l1_change"]].mean().reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for model, group in grouped.groupby("model"):
        axes[0].plot(group["relative_noise_standard_deviation"], group["l1_weight_change"], marker="o", label=model)
        axes[1].plot(group["relative_noise_standard_deviation"], group["economic_exposure_l1_change"], marker="o", label=model)
    axes[0].set(title="Security-weight distortion", xlabel="Clone noise / source volatility", ylabel="Mean L1 change")
    axes[1].set(title="Economic-exposure distortion", xlabel="Clone noise / source volatility")
    axes[1].legend(fontsize=7)
    paths["figure_09_clone_distortion"] = _save(figure, output / "figure_09_clone_distortion.png")

    clustering = pd.read_csv(output / "clustering_date_diagnostics.csv")
    cluster_average = clustering.groupby("threshold")[["medoid_condition_number", "medoid_effective_rank"]].median().reset_index()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    twin = axis.twinx()
    axis.plot(cluster_average["threshold"], cluster_average["medoid_condition_number"], marker="o", color="tab:blue")
    twin.plot(cluster_average["threshold"], cluster_average["medoid_effective_rank"], marker="s", color="tab:orange")
    axis.set(title="Clustering threshold versus covariance geometry", xlabel="Correlation-equivalent threshold", ylabel="Median condition number", yscale="log")
    twin.set_ylabel("Median effective rank")
    paths["figure_10_clustering_conditioning"] = _save(figure, output / "figure_10_clustering_conditioning.png")

    redundancy = pd.read_csv(output / "table_4_redundancy_ablation.csv")
    medoid = redundancy[redundancy["universe"] == "CLUSTER_MEDOID"]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for model, group in medoid.groupby("model"):
        axis.plot(group["threshold"], group["recurring_one_way_turnover"], marker="o", label=model)
    axis.set(title="Clustering threshold versus turnover", xlabel="Correlation-equivalent threshold", ylabel="Recurring one-way turnover")
    axis.legend(fontsize=8)
    paths["figure_11_clustering_turnover"] = _save(figure, output / "figure_11_clustering_turnover.png")

    regimes = pd.read_csv(output / "regime_summary.csv")
    pivot = regimes.pivot(index="strategy", columns="regime", values="mean_period_return")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Mean holding-period return by assigned regime")
    figure.colorbar(image, ax=axis, label="Mean period return")
    paths["figure_12_regime_performance"] = _save(figure, output / "figure_12_regime_performance.png")

    sensitivity = pd.read_csv(output / "sensitivity_diagnostics.csv")
    grid = sensitivity[sensitivity["sensitivity_kind"] == "rho_kappa_grid"].groupby(
        ["rho_multiplier", "kappa_multiplier"]
    )["l1_weight_change"].mean().unstack()
    figure, axis = plt.subplots(figsize=(7, 5.5))
    image = axis.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
    axis.set_yticks(range(len(grid.index)), [f"{x:g}" for x in grid.index])
    axis.set(xlabel="Kappa multiplier", ylabel="Rho multiplier", title="Box+diagonal allocation sensitivity")
    figure.colorbar(image, ax=axis, label="Mean L1 change vs baseline")
    paths["figure_13_rho_kappa_sensitivity"] = _save(figure, output / "figure_13_rho_kappa_sensitivity.png")
    return paths
