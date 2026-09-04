from pathlib import Path
import os
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "cs361_matplotlib_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_DISPLAY_NAMES = {
    "equal_weight": "Equal Weight",
    "classical_markowitz": "Classical Markowitz",
    "robust_markowitz": "Robust Markowitz",
}

STRATEGY_STYLES = {
    "equal_weight": {
        "color": "#1f77b4",
        "linewidth": 2.6,
        "linestyle": "-",
        "alpha": 0.98,
        "zorder": 4,
    },
    "classical_markowitz": {
        "color": "#ff7f0e",
        "linewidth": 2.3,
        "linestyle": "--",
        "alpha": 0.98,
        "zorder": 3,
    },
    "robust_markowitz": {
        "color": "#2ca02c",
        "linewidth": 2.3,
        "linestyle": "-.",
        "alpha": 0.98,
        "zorder": 2,
    },
}


def _compute_drawdowns(portfolio_values):
    running_max = portfolio_values.cummax()
    return portfolio_values / running_max - 1.0


def _style_axes(axis):
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _display_name(strategy_name):
    return STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name.replace("_", " ").title())


def _plot_strategy_lines(axis, data):
    for column in data.columns:
        data[column].plot(
            ax=axis,
            label=_display_name(column),
            **STRATEGY_STYLES.get(column, {"linewidth": 2.0}),
        )


def _annotate_line_endpoints(axis, data):
    for column in data.columns:
        series = data[column].dropna()
        if series.empty:
            continue

        style = STRATEGY_STYLES.get(column, {})
        axis.annotate(
            _display_name(column),
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=style.get("color", "black"),
            fontsize=9,
            fontweight="bold",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
        )


def _plot_portfolio_values(portfolio_values):
    figure, axis = plt.subplots(figsize=(12, 6))
    _plot_strategy_lines(axis, portfolio_values)
    _annotate_line_endpoints(axis, portfolio_values)
    axis.set_title("Portfolio Value Through Time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio Value")
    axis.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, title="Strategy")
    _style_axes(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "portfolio_value_plot.png", dpi=200)
    plt.close(figure)


def _plot_drawdowns(portfolio_values):
    drawdowns = _compute_drawdowns(portfolio_values)
    figure, axis = plt.subplots(figsize=(12, 6))
    _plot_strategy_lines(axis, drawdowns)
    _annotate_line_endpoints(axis, drawdowns)
    axis.set_title("Portfolio Drawdowns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, title="Strategy")
    _style_axes(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "drawdown_plot.png", dpi=200)
    plt.close(figure)


def _plot_turnover(turnover):
    average_turnover = turnover.mean(skipna=True).sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 5))
    bar_colors = [STRATEGY_STYLES.get(name, {}).get("color", "#4c4c4c") for name in average_turnover.index]
    average_turnover.plot(kind="bar", ax=axis, color=bar_colors)
    axis.set_title("Average Quarterly Turnover")
    axis.set_xlabel("Strategy")
    axis.set_ylabel("Average Turnover")
    axis.set_xticklabels([_display_name(name) for name in average_turnover.index], rotation=20, ha="right")
    _style_axes(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "turnover_bar_chart.png", dpi=200)
    plt.close(figure)


def _plot_metrics_table(metrics):
    figure, axis = plt.subplots(figsize=(12, 3))
    axis.axis("off")

    formatted_metrics = metrics.copy()
    formatted_metrics.index = [_display_name(name) for name in formatted_metrics.index]
    for column in formatted_metrics.columns:
        formatted_metrics[column] = formatted_metrics[column].map(lambda value: f"{value:.4f}")

    table = axis.table(
        cellText=formatted_metrics.values,
        rowLabels=formatted_metrics.index.tolist(),
        colLabels=formatted_metrics.columns.tolist(),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    axis.set_title("Performance Metrics", pad=12)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "performance_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_weight_heatmap(weights_path, output_name, title, top_n=25):
    if not weights_path.exists():
        return

    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    weights = weights.dropna(axis=1, how="all").fillna(0.0)
    if weights.empty:
        return

    top_assets = weights.mean(axis=0).sort_values(ascending=False).head(top_n).index
    heatmap_data = weights[top_assets].T

    figure_height = max(6, len(top_assets) * 0.35)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    image = axis.imshow(heatmap_data, aspect="auto", cmap="YlGnBu", interpolation="nearest")

    axis.set_title(title)
    axis.set_xlabel("Rebalance Date")
    axis.set_ylabel("ETF")
    axis.set_yticks(np.arange(len(heatmap_data.index)))
    axis.set_yticklabels(heatmap_data.index.tolist())
    axis.set_xticks(np.arange(len(heatmap_data.columns)))
    axis.set_xticklabels(
        [timestamp.strftime("%Y-%m") for timestamp in heatmap_data.columns],
        rotation=45,
        ha="right",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Portfolio Weight")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close(figure)


def create_plots():
    """Generate plots from the saved backtest outputs."""
    portfolio_values_path = OUTPUT_DIR / "portfolio_values.csv"
    turnover_path = OUTPUT_DIR / "turnover.csv"
    metrics_path = OUTPUT_DIR / "performance_metrics.csv"

    if not portfolio_values_path.exists() or not turnover_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run the backtest and evaluation before generating plots.")

    print("Generating plots...")
    portfolio_values = pd.read_csv(portfolio_values_path, index_col=0, parse_dates=True)
    turnover = pd.read_csv(turnover_path, index_col=0, parse_dates=True)
    metrics = pd.read_csv(metrics_path, index_col=0)

    _plot_portfolio_values(portfolio_values)
    _plot_drawdowns(portfolio_values)
    _plot_turnover(turnover)
    _plot_metrics_table(metrics)
    _plot_weight_heatmap(
        OUTPUT_DIR / "weights_robust_markowitz.csv",
        "weights_heatmap_robust_markowitz.png",
        "Robust Markowitz Weight Heatmap",
    )

    print("Saved plots:")
    print(f"  {OUTPUT_DIR / 'portfolio_value_plot.png'}")
    print(f"  {OUTPUT_DIR / 'drawdown_plot.png'}")
    print(f"  {OUTPUT_DIR / 'turnover_bar_chart.png'}")
    print(f"  {OUTPUT_DIR / 'performance_metrics_table.png'}")
    print(f"  {OUTPUT_DIR / 'weights_heatmap_robust_markowitz.png'}")


if __name__ == "__main__":
    create_plots()
