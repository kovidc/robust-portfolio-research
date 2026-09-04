#!/usr/bin/env python3
"""Offline golden-master reproduction of the historical CS361 experiment.

This wrapper intentionally executes the unchanged legacy implementation. It
does not repair any methodology and does not invoke the data-download module.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE_DIR = REPOSITORY_ROOT / "src"
LEGACY_MANIFEST_PATH = REPOSITORY_ROOT / "legacy" / "baseline_manifest.json"
LEGACY_CONFIG_PATH = REPOSITORY_ROOT / "legacy" / "config.json"
HISTORICAL_OUTPUT_DIR = REPOSITORY_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "legacy_cs361" / "reproduced"

if str(LEGACY_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_SOURCE_DIR))

# These are the historical modules. Their algorithms are deliberately not
# copied or corrected here.
import backtest as legacy_backtest  # noqa: E402
import evaluate as legacy_evaluate  # noqa: E402
import plots as legacy_plots  # noqa: E402


STRATEGIES = ("equal_weight", "classical_markowitz", "robust_markowitz")
WEIGHT_FILES = {
    "equal_weight": "weights_equal_weight.csv",
    "classical_markowitz": "weights_classical_markowitz.csv",
    "robust_markowitz": "weights_robust_markowitz.csv",
}


class _Tee:
    """Write legacy progress to the terminal and retain it for provenance."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash_group(expected_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for relative_path, expected_hash in expected_hashes.items():
        path = REPOSITORY_ROOT / relative_path
        actual_hash = _sha256(path) if path.exists() else None
        checks[relative_path] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "matches": actual_hash == expected_hash,
        }
    return checks


def verify_frozen_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify immutable inputs, legacy source, config, and stored evidence."""
    groups = {
        "frozen_inputs": _verify_hash_group(manifest["frozen_input_sha256"]),
        "legacy_source": _verify_hash_group(manifest["legacy_source_sha256"]),
        "historical_evidence": _verify_hash_group(manifest["historical_evidence_sha256"]),
        "configuration": _verify_hash_group(
            {manifest["configuration"]["path"]: manifest["configuration"]["sha256"]}
        ),
    }
    mismatches = [
        path
        for checks in groups.values()
        for path, check in checks.items()
        if not check["matches"]
    ]
    return {"groups": groups, "all_match": not mismatches, "mismatches": mismatches}


def _validate_output_directory(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    historical = HISTORICAL_OUTPUT_DIR.resolve()
    if resolved == historical or historical in resolved.parents:
        raise ValueError("Legacy reproduction may not overwrite the historical outputs/ evidence.")

    repository = REPOSITORY_ROOT.resolve()
    allowed_repository_namespace = (REPOSITORY_ROOT / "artifacts" / "legacy_cs361").resolve()
    if resolved == repository:
        raise ValueError("The repository root cannot be used as a reproduction output directory.")
    if repository in resolved.parents:
        if resolved != allowed_repository_namespace and allowed_repository_namespace not in resolved.parents:
            raise ValueError(
                "Repository-local legacy artifacts must stay under artifacts/legacy_cs361/."
            )
    return resolved


def _native_metrics(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(strategy): {str(metric): float(value) for metric, value in row.items()}
        for strategy, row in metrics.iterrows()
    }


def validate_metrics(
    metrics: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(manifest["numerical_tolerances"]["metric_absolute"])
    expected = manifest["expected_legacy_metrics"]
    comparisons: dict[str, dict[str, Any]] = {}
    passed = True

    for strategy, expected_metrics in expected.items():
        comparisons[strategy] = {}
        for metric_name, expected_value in expected_metrics.items():
            actual_value = float(metrics.loc[strategy, metric_name])
            absolute_error = abs(actual_value - float(expected_value))
            metric_passed = absolute_error <= tolerance
            passed = passed and metric_passed
            comparisons[strategy][metric_name] = {
                "expected": float(expected_value),
                "actual": actual_value,
                "absolute_error": absolute_error,
                "tolerance": tolerance,
                "passed": metric_passed,
            }

    return {"passed": passed, "comparisons": comparisons}


def build_accounting_diagnostic(
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Prove both incompatible accounting rules used by the legacy engine."""
    returns = pd.read_csv(
        REPOSITORY_ROOT / "data" / "returns_clean.csv", index_col=0, parse_dates=True
    )
    rebalance_dates = pd.DatetimeIndex(
        pd.read_csv(
            REPOSITORY_ROOT / "data" / "quarterly_rebalance_dates.csv",
            parse_dates=["rebalance_date"],
        )["rebalance_date"]
    )
    reported_returns = pd.read_csv(
        output_dir / "portfolio_returns.csv", index_col=0, parse_dates=True
    )
    reported_turnover = pd.read_csv(output_dir / "turnover.csv", index_col=0, parse_dates=True)

    accounting_tolerance = float(
        manifest["numerical_tolerances"]["accounting_identity_absolute"]
    )
    required_path_difference = float(
        manifest["numerical_tolerances"]["buy_and_hold_cumulative_difference_minimum"]
    )
    results: dict[str, Any] = {}

    for strategy in STRATEGIES:
        target_weights = pd.read_csv(
            output_dir / WEIGHT_FILES[strategy], index_col=0, parse_dates=True
        )
        reconstructed_fixed = pd.Series(0.0, index=reported_returns.index, dtype=float)
        reconstructed_turnover = pd.Series(np.nan, index=rebalance_dates, dtype=float)
        previous_drifted_weights = None
        quarterly_buy_and_hold_growth = 1.0

        for position, rebalance_date in enumerate(rebalance_dates):
            if position + 1 < len(rebalance_dates):
                next_rebalance_date = rebalance_dates[position + 1]
                holding_returns = returns.loc[
                    (returns.index >= rebalance_date) & (returns.index < next_rebalance_date)
                ]
            else:
                holding_returns = returns.loc[returns.index >= rebalance_date]

            weights = target_weights.loc[rebalance_date].reindex(returns.columns).fillna(0.0)
            fixed_daily_returns = holding_returns.mul(weights, axis=1).sum(axis=1)
            reconstructed_fixed.loc[fixed_daily_returns.index] = fixed_daily_returns

            if previous_drifted_weights is not None:
                reconstructed_turnover.loc[rebalance_date] = 0.5 * np.abs(
                    weights - previous_drifted_weights
                ).sum()

            asset_growth = (1.0 + holding_returns).prod(axis=0)
            quarterly_buy_and_hold_growth *= float((weights * asset_growth).sum())
            previous_drifted_weights = legacy_backtest._compute_drifted_weights(
                weights, holding_returns
            )

        fixed_target_max_error = float(
            (reconstructed_fixed - reported_returns[strategy]).abs().max()
        )
        comparable_turnover = reported_turnover[strategy].notna()
        drift_turnover_max_error = float(
            (
                reconstructed_turnover.loc[comparable_turnover]
                - reported_turnover.loc[comparable_turnover, strategy]
            )
            .abs()
            .max()
        )
        reported_cumulative_return = float((1.0 + reported_returns[strategy]).prod() - 1.0)
        quarterly_buy_and_hold_cumulative_return = quarterly_buy_and_hold_growth - 1.0
        path_difference = abs(
            reported_cumulative_return - quarterly_buy_and_hold_cumulative_return
        )

        results[strategy] = {
            "reported_returns_match_daily_fixed_target_weights": (
                fixed_target_max_error <= accounting_tolerance
            ),
            "fixed_target_daily_return_max_absolute_error": fixed_target_max_error,
            "reported_turnover_matches_quarterly_drifted_pretrade_weights": (
                drift_turnover_max_error <= accounting_tolerance
            ),
            "quarterly_drift_turnover_max_absolute_error": drift_turnover_max_error,
            "reported_constant_target_cumulative_return": reported_cumulative_return,
            "quarterly_buy_and_hold_cumulative_return": (
                quarterly_buy_and_hold_cumulative_return
            ),
            "absolute_cumulative_return_path_difference": path_difference,
            "return_and_turnover_accounting_are_contradictory": (
                fixed_target_max_error <= accounting_tolerance
                and drift_turnover_max_error <= accounting_tolerance
                and path_difference >= required_path_difference
            ),
        }

    return {
        "artifact_type": "LEGACY_ACCOUNTING_CONTRADICTION",
        "result_label": "LEGACY",
        "warning": "This diagnostic preserves a known defect; it is not corrected accounting.",
        "reported_return_rule": "daily constant target weights",
        "reported_turnover_rule": "quarterly drifted pre-trade weights",
        "accounting_identity_tolerance": accounting_tolerance,
        "required_cumulative_path_difference": required_path_difference,
        "all_strategies_exhibit_contradiction": all(
            result["return_and_turnover_accounting_are_contradictory"]
            for result in results.values()
        ),
        "strategies": results,
    }


def _runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "cvxpy", "osqp", "clarabel", "scs", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"

    try:
        import cvxpy as cp

        installed_solvers = cp.installed_solvers()
    except ImportError:
        installed_solvers = []

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cvxpy_installed_solvers": installed_solvers,
    }


def reproduce_legacy(output_dir: Path | str = DEFAULT_OUTPUT_DIR, create_plots: bool = True):
    """Run the unchanged legacy experiment from frozen cached CSV inputs."""
    output_dir = _validate_output_directory(Path(output_dir))
    manifest = _load_json(LEGACY_MANIFEST_PATH)
    config = _load_json(LEGACY_CONFIG_PATH)
    hash_verification = verify_frozen_evidence(manifest)
    if not hash_verification["all_match"]:
        mismatches = ", ".join(hash_verification["mismatches"])
        raise RuntimeError(f"Frozen legacy evidence hash mismatch: {mismatches}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_buffer = io.StringIO()
    original_directories = (
        legacy_backtest.DATA_DIR,
        legacy_backtest.OUTPUT_DIR,
        legacy_evaluate.OUTPUT_DIR,
        legacy_plots.OUTPUT_DIR,
    )

    try:
        legacy_backtest.DATA_DIR = REPOSITORY_ROOT / "data"
        legacy_backtest.OUTPUT_DIR = output_dir
        legacy_evaluate.OUTPUT_DIR = output_dir
        legacy_plots.OUTPUT_DIR = output_dir

        with redirect_stdout(_Tee(sys.stdout, log_buffer)):
            legacy_backtest.run_backtest(
                classical_gamma=config["optimization"]["gamma_classical"],
                robust_gamma=config["optimization"]["gamma_robust"],
                rho=config["optimization"]["rho"],
                cov_uncertainty=config["optimization"]["kappa"],
                max_weight=config["optimization"]["maximum_weight"],
                initial_value=config["walk_forward"]["initial_portfolio_value"],
            )
            metrics = legacy_evaluate.evaluate_performance()
            if create_plots:
                legacy_plots.create_plots()
    finally:
        (
            legacy_backtest.DATA_DIR,
            legacy_backtest.OUTPUT_DIR,
            legacy_evaluate.OUTPUT_DIR,
            legacy_plots.OUTPUT_DIR,
        ) = original_directories

    log_text = log_buffer.getvalue()
    (output_dir / "legacy_console.log").write_text(log_text, encoding="utf-8")
    fallback_detected = (
        "falling back to equal-weight portfolios" in log_text
        or "failed, using equal weight fallback" in log_text
    )
    metric_validation = validate_metrics(metrics, manifest)
    accounting_diagnostic = build_accounting_diagnostic(output_dir, manifest)

    metrics_artifact = {
        "artifact_type": "LEGACY_METRICS",
        "result_label": "LEGACY",
        "warning": "Historical reproduction only; these are not corrected research metrics.",
        "metric_semantics": config["metrics"],
        "metrics": _native_metrics(metrics),
        "golden_master_validation": metric_validation,
    }
    _write_json(output_dir / "legacy_metrics.json", metrics_artifact)
    _write_json(output_dir / "accounting_contradiction.json", accounting_diagnostic)

    run_artifact = {
        "artifact_type": "LEGACY_REPRODUCTION_RUN",
        "result_label": "LEGACY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction_command": manifest["reproduction_command"],
        "output_directory": str(output_dir),
        "network_access_used": False,
        "hash_verification": hash_verification,
        "runtime_environment": _runtime_environment(),
        "optimizer_fallback_detected": fallback_detected,
        "golden_metrics_passed": metric_validation["passed"],
        "accounting_contradiction_preserved": accounting_diagnostic[
            "all_strategies_exhibit_contradiction"
        ],
    }
    _write_json(output_dir / "legacy_run.json", run_artifact)

    if fallback_detected:
        raise RuntimeError("Optimizer fallback detected during the LEGACY reproduction.")
    if not metric_validation["passed"]:
        raise RuntimeError("Reproduced LEGACY metrics do not match the golden master.")
    if not accounting_diagnostic["all_strategies_exhibit_contradiction"]:
        raise RuntimeError("The frozen LEGACY accounting contradiction was not reproduced.")

    return {
        "output_dir": output_dir,
        "metrics": metrics,
        "metrics_artifact": metrics_artifact,
        "accounting_diagnostic": accounting_diagnostic,
        "run_artifact": run_artifact,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the historical CS361 LEGACY baseline from frozen cached data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (repository-local paths must remain under artifacts/legacy_cs361).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation; numeric LEGACY reproduction and diagnostics still run.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = reproduce_legacy(args.output_dir, create_plots=not args.skip_plots)
    print()
    print("LEGACY CS361 reproduction passed.")
    print(f"Artifacts: {result['output_dir']}")
    print("These outputs preserve historical defects and are not corrected research results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
