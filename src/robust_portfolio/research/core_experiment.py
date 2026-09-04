"""End-to-end orchestration for the core quantitative experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import cvxpy as cp
import numpy as np
import pandas as pd

from robust_portfolio.calibration import derive_outer_schedule
from robust_portfolio.data import (
    FrozenCsvReturnProvider,
    SurvivorPanelUniverseBuilder,
    UniverseRules,
)
from robust_portfolio.data.providers import sha256_file
from robust_portfolio.estimators import (
    calibrate_uncertainty,
    estimate_covariance,
    estimate_mean,
)
from robust_portfolio.optimizers import (
    OptimizationFailure,
    asset_class_equal_weight,
    diagonal_robust_covariance,
    equal_weight,
    global_minimum_variance,
    inverse_volatility,
    risk_parity,
    solve_target_risk,
)
from robust_portfolio.reporting.manifests import dependency_versions, git_state
from robust_portfolio.reporting.metrics import scenario_metrics
from robust_portfolio.reporting.core_outputs import (
    create_core_figures,
    prepare_output_directory,
    write_json,
)

from .configuration import CoreExperimentConfig
from .covariance_study import run_covariance_study
from .simulation import simulate_targets


def _covariance_kwargs(config: CoreExperimentConfig) -> dict:
    section = config.section("covariances")
    return {
        "annualization_factor": int(config.section("annualization")["trading_days"]),
        "ewma_half_life": float(section["ewma_half_life"]),
        "iewma_volatility_half_life": float(section["iewma_volatility_half_life"]),
        "iewma_correlation_half_life": float(section["iewma_correlation_half_life"]),
        "iewma_winsorize_clip": float(section["iewma_winsorize_clip"]),
        "iewma_variance_floor": float(section["iewma_variance_floor"]),
        "absolute_eigenvalue_floor": float(section["absolute_eigenvalue_floor"]),
        "relative_eigenvalue_floor": float(section["relative_eigenvalue_floor"]),
    }


def _mean_kwargs(config: CoreExperimentConfig) -> dict:
    section = config.section("means")
    return {
        "annualization_factor": int(config.section("annualization")["trading_days"]),
        "ewma_half_life": float(section["ewma_half_life"]),
        "shrinkage_intensity": float(section["shrinkage_intensity"]),
    }


def _asset_classes(path: Path, assets) -> pd.Series:
    metadata = pd.read_csv(path)
    kept = metadata[metadata["kept_after_cleaning"].astype(str).str.lower() == "true"]
    classes = kept.set_index("ticker")["asset_class"].astype(str)
    missing = pd.Index(assets).difference(classes.index)
    if len(missing):
        raise ValueError(f"Static asset-class metadata are missing: {missing.tolist()}")
    return classes.reindex(assets)


def _strategy_metadata(ablation, mean, covariance, robust_set, target_risk):
    return {
        "ablation": ablation,
        "mean_estimator": mean,
        "covariance_estimator": covariance,
        "robust_set": robust_set,
        "target_risk": target_risk,
    }


def _risk_name(prefix: str, target: float) -> str:
    return f"{prefix}_risk_{int(round(target * 100)):02d}pct"


def run_core_experiment(
    config_path: Path | str,
    *,
    repository_root: Path | str,
    output_dir: Path | str | None = None,
    max_outer_dates: int | None = None,
) -> dict:
    repository = Path(repository_root).resolve()
    config = CoreExperimentConfig.load(config_path)
    returns_path = config.repository_path(repository, "returns_path")
    schedule_path = config.repository_path(repository, "rebalance_dates_path")
    metadata_path = config.repository_path(repository, "universe_metadata_path")
    provider = FrozenCsvReturnProvider(returns_path)
    all_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).astype(float)
    schedule = pd.DatetimeIndex(
        pd.read_csv(schedule_path, parse_dates=["rebalance_date"])["rebalance_date"]
    )
    walk = config.section("walkforward")
    outer_dates, folds_by_outer = derive_outer_schedule(
        provider.dates,
        schedule,
        estimation_window=int(walk["estimation_window_observations"]),
        minimum_prior_inner_folds=int(walk["minimum_prior_inner_folds"]),
    )
    configured_start = pd.Timestamp(walk["formal_outer_start_date"])
    if outer_dates[0] != configured_start:
        raise ValueError(
            f"Mechanical outer start {outer_dates[0].date()} does not match configured start "
            f"{configured_start.date()}."
        )
    if max_outer_dates is not None:
        if max_outer_dates < 1:
            raise ValueError("max_outer_dates must be positive when supplied.")
        outer_dates = outer_dates[:max_outer_dates]

    output = prepare_output_directory(
        output_dir
        or repository
        / config.section("outputs")["default_root"]
        / config.sha256[:12],
        repository,
    )
    existing_manifest = output / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("configuration", {}).get("canonical_sha256") != config.sha256:
            raise ValueError("Refusing to overwrite results from a different configuration.")

    universe_rules = UniverseRules(
        required_history_observations=int(
            config.section("universe")["required_history_observations"]
        ),
        require_complete_required_window=bool(
            config.section("universe")["require_complete_required_window"]
        ),
    )
    universe_builder = SurvivorPanelUniverseBuilder(provider.assets, universe_rules)
    classes = _asset_classes(metadata_path, provider.assets)
    maximum_weight = float(config.section("constraints")["maximum_weight"])
    annualization = int(config.section("annualization")["trading_days"])
    optimization = config.section("optimization")
    quadratic_solvers = list(optimization["solver_order_quadratic"])
    conic_solvers = list(optimization["solver_order_conic"])
    feasibility_tolerance = float(optimization["feasibility_tolerance"])
    covariance_methods = list(config.section("covariances")["implemented"])
    headline_covariance = config.section("covariances")["headline"]
    mean_methods = list(config.section("means")["implemented"])
    headline_mean = config.section("means")["headline"]
    risk_targets = [
        float(value)
        for value in config.section("risk_matching")["target_annual_volatility"]
    ]

    targets: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    metadata: dict[str, dict] = {}
    failures: dict[str, list[str]] = {}
    solver_records = []
    calibration_records = []
    mean_records = []
    covariance_records = []
    universe_records = []
    forecasts_by_date = {}

    def register(name: str, strategy_metadata: dict, date, weights: pd.Series) -> None:
        metadata.setdefault(name, strategy_metadata)
        aligned = weights.reindex(provider.assets, fill_value=0.0).astype(float)
        if not np.isfinite(aligned.to_numpy()).all():
            raise ValueError(f"{name} produced non-finite weights at {date}.")
        if abs(float(aligned.sum()) - 1.0) > feasibility_tolerance:
            raise ValueError(f"{name} is not fully invested at {date}.")
        if float(aligned.min()) < -feasibility_tolerance:
            raise ValueError(f"{name} violates long-only constraints at {date}.")
        if float(aligned.max()) > maximum_weight + feasibility_tolerance:
            raise ValueError(f"{name} violates the maximum weight at {date}.")
        targets.setdefault(name, {})[pd.Timestamp(date)] = aligned

    def record_solver(name: str, date, result) -> None:
        solver_records.append(
            {
                "strategy": name,
                "decision_date": date,
                "status": result.status,
                "solver": result.solver,
                "objective_value": result.objective_value,
                "predicted_decision_volatility": result.predicted_volatility,
                "predicted_common_base_volatility": result.common_base_volatility,
                "target_volatility": result.target_volatility,
                "target_binding": result.target_binding,
                "sum_residual": result.sum_residual,
                "lower_violation": result.lower_violation,
                "cap_violation": result.cap_violation,
            }
        )

    for date in outer_dates:
        full_panel = provider.panel(as_of=date)
        snapshot = universe_builder.snapshot(full_panel)
        universe_records.append(snapshot.to_dict())
        assets = list(snapshot.eligible_assets)
        panel = provider.panel(
            as_of=date,
            assets=assets,
            trailing_observations=int(walk["estimation_window_observations"]),
        )
        means = {
            method: estimate_mean(panel, method, **_mean_kwargs(config))
            for method in mean_methods
        }
        covariances = {
            method: estimate_covariance(panel, method, **_covariance_kwargs(config))
            for method in covariance_methods
        }
        forecasts_by_date[pd.Timestamp(date)] = covariances
        for method, forecast in means.items():
            mean_records.append(
                {
                    "decision_date": date,
                    "estimator": method,
                    "cross_sectional_mean": float(forecast.annualized_mean.mean()),
                    "cross_sectional_std": float(forecast.annualized_mean.std(ddof=1)),
                    "minimum": float(forecast.annualized_mean.min()),
                    "maximum": float(forecast.annualized_mean.max()),
                    "observations": forecast.observations,
                }
            )
        for method, forecast in covariances.items():
            covariance_records.append(
                {
                    "decision_date": date,
                    "estimator": method,
                    "ridge_added": forecast.ridge_added,
                    "minimum_eigenvalue_before": forecast.minimum_eigenvalue_before,
                    "shrinkage_intensity": forecast.shrinkage_intensity,
                    "observations": forecast.observations,
                }
            )

        uncertainty_config = config.section("uncertainty")
        date_seed = int(uncertainty_config["bootstrap_seed"]) + int(date.strftime("%Y%m%d"))
        uncertainty = calibrate_uncertainty(
            panel,
            bootstrap_seed=date_seed,
            bootstrap_replications=int(uncertainty_config["bootstrap_replications"]),
            block_length=int(uncertainty_config["block_length_observations"]),
            coverage_probability=float(uncertainty_config["coverage_probability"]),
            annualization_factor=annualization,
            standard_error_floor=float(uncertainty_config["standard_error_floor"]),
            relative_eigenvalue_floor=float(
                uncertainty_config["mean_uncertainty_relative_eigenvalue_floor"]
            ),
            absolute_eigenvalue_floor=float(
                uncertainty_config["mean_uncertainty_absolute_eigenvalue_floor"]
            ),
            iewma_volatility_half_life=float(
                config.section("covariances")["iewma_volatility_half_life"]
            ),
            iewma_variance_floor=float(
                config.section("covariances")["iewma_variance_floor"]
            ),
        )
        calibration_records.append(
            {
                "decision_date": date,
                "bootstrap_seed": date_seed,
                "box_rho": uncertainty.box_rho,
                "ellipsoid_rho": uncertainty.ellipsoid_rho,
                "diagonal_kappa": uncertainty.diagonal_kappa,
                "minimum_standard_error": float(uncertainty.standard_errors.min()),
                "maximum_standard_error": float(uncertainty.standard_errors.max()),
                "mean_covariance_ridge": uncertainty.mean_covariance_ridge,
                "latest_inner_validation_end": folds_by_outer[pd.Timestamp(date)][-1].validation_end,
            }
        )

        common_covariance = covariances[headline_covariance].annualized_covariance
        common_mean = means[headline_mean].annualized_mean
        register(
            "etf_equal_weight",
            _strategy_metadata("A0", "none", "none", "none", None),
            date,
            equal_weight(assets, maximum_weight),
        )
        register(
            "asset_class_equal_weight",
            _strategy_metadata("A0", "none", "none", "none", None),
            date,
            asset_class_equal_weight(assets, classes, maximum_weight),
        )
        register(
            "inverse_volatility_iewma",
            _strategy_metadata("A0", "none", headline_covariance, "none", None),
            date,
            inverse_volatility(common_covariance, maximum_weight),
        )
        register(
            "risk_parity_iewma",
            _strategy_metadata("benchmark", "none", headline_covariance, "none", None),
            date,
            risk_parity(
                common_covariance,
                maximum_weight,
                tolerance=float(optimization["risk_parity_objective_tolerance"]),
            ),
        )

        for covariance_method, forecast in covariances.items():
            name = f"gmv_{covariance_method}"
            try:
                result = global_minimum_variance(
                    forecast.annualized_covariance,
                    maximum_weight=maximum_weight,
                    solver_order=quadratic_solvers,
                    feasibility_tolerance=feasibility_tolerance,
                )
                register(
                    name,
                    _strategy_metadata(
                        "A1" if covariance_method == "sample" else "A2",
                        "ignored",
                        covariance_method,
                        "none",
                        None,
                    ),
                    date,
                    result.weights,
                )
                record_solver(name, date, result)
            except OptimizationFailure as error:
                failures.setdefault(name, []).append(f"{date.date()}:{error}")

        robust_covariance = pd.DataFrame(
            diagonal_robust_covariance(common_covariance, uncertainty.diagonal_kappa),
            index=assets,
            columns=assets,
        )
        for target in risk_targets:
            specifications = [
                (
                    _risk_name("nominal", target),
                    "A3",
                    "none",
                    common_covariance,
                    {},
                ),
                (
                    _risk_name("box", target),
                    "A4",
                    "box_mean",
                    common_covariance,
                    {
                        "standard_errors": uncertainty.standard_errors,
                        "box_rho": uncertainty.box_rho,
                    },
                ),
                (
                    _risk_name("box_diagonal", target),
                    "A5",
                    "box_mean_plus_diagonal_covariance",
                    robust_covariance,
                    {
                        "standard_errors": uncertainty.standard_errors,
                        "box_rho": uncertainty.box_rho,
                    },
                ),
                (
                    _risk_name("ellipsoid", target),
                    "A6",
                    "ellipsoidal_mean",
                    common_covariance,
                    {
                        "mean_error_covariance": uncertainty.mean_error_covariance,
                        "ellipsoid_rho": uncertainty.ellipsoid_rho,
                    },
                ),
            ]
            for name, ablation, robust_set, decision_covariance, robust_arguments in specifications:
                try:
                    result = solve_target_risk(
                        common_mean,
                        decision_covariance,
                        target_volatility=target,
                        maximum_weight=maximum_weight,
                        solver_order=conic_solvers,
                        common_base_covariance=common_covariance,
                        feasibility_tolerance=feasibility_tolerance,
                        target_binding_tolerance=float(
                            config.section("risk_matching")["target_binding_tolerance"]
                        ),
                        **robust_arguments,
                    )
                    register(
                        name,
                        _strategy_metadata(
                            ablation, headline_mean, headline_covariance, robust_set, target
                        ),
                        date,
                        result.weights,
                    )
                    record_solver(name, date, result)
                except OptimizationFailure as error:
                    failures.setdefault(name, []).append(f"{date.date()}:{error}")

    complete_targets = {
        name: values
        for name, values in targets.items()
        if len(values) == len(outer_dates) and name not in failures
    }
    covariance_summary, covariance_periods = run_covariance_study(
        forecasts_by_date,
        all_returns,
        outer_dates,
        annualization_factor=annualization,
        maximum_weight=maximum_weight,
        solver_order=quadratic_solvers,
    )

    metrics_records = []
    wealth_records = []
    experiment_variants = []
    headline_target = float(
        config.section("risk_matching")["headline_target_annual_volatility"]
    )
    headline_names = {
        "etf_equal_weight",
        "asset_class_equal_weight",
        "inverse_volatility_iewma",
        "gmv_iewma",
        _risk_name("nominal", headline_target),
        _risk_name("box", headline_target),
        _risk_name("box_diagonal", headline_target),
        _risk_name("ellipsoid", headline_target),
    }
    cost_scenarios = [
        float(value) for value in config.section("costs")["basis_points_per_dollar_traded"]
    ]
    solver_diagnostics = pd.DataFrame(solver_records)
    solver_summary = {}
    if not solver_diagnostics.empty:
        for name, group in solver_diagnostics.groupby("strategy"):
            binding = group["target_binding"].dropna()
            solver_summary[name] = {
                "average_predicted_decision_volatility": float(
                    group["predicted_decision_volatility"].mean()
                ),
                "target_binding_fraction": (
                    float(binding.astype(float).mean()) if len(binding) else np.nan
                ),
            }
    common_predicted_risk = {}
    for name, strategy_targets in complete_targets.items():
        values = []
        for date, weights in strategy_targets.items():
            matrix = forecasts_by_date[date][headline_covariance].annualized_covariance
            aligned = weights.reindex(matrix.index)
            values.append(float(np.sqrt(max(float(aligned @ matrix @ aligned), 0.0))))
        common_predicted_risk[name] = float(np.mean(values))

    for name, strategy_targets in sorted(complete_targets.items()):
        for cost_bps in cost_scenarios:
            path = simulate_targets(
                provider,
                strategy_targets,
                strategy=name,
                cost_bps=cost_bps,
                maximum_weight=maximum_weight,
                cash_daily_return=float(config.section("annualization")["cash_daily_return"]),
                market_returns=all_returns,
            )
            solver_values = solver_summary.get(name, {})
            record = {
                "strategy": name,
                "cost_bps": cost_bps,
                **metadata[name],
                "average_predicted_decision_volatility": solver_values.get(
                    "average_predicted_decision_volatility",
                    common_predicted_risk[name],
                ),
                "average_predicted_common_base_volatility": common_predicted_risk[name],
                "target_binding_fraction": solver_values.get(
                    "target_binding_fraction", np.nan
                ),
            }
            record.update(scenario_metrics(path, strategy_targets, annualization_factor=annualization))
            metrics_records.append(record)
            experiment_variants.append({**record, "status": "COMPLETED"})
            if name in headline_names:
                frame = path.daily.reset_index()
                frame.insert(1, "strategy", name)
                frame.insert(2, "cost_bps", cost_bps)
                wealth_records.extend(frame.to_dict(orient="records"))
    for name, messages in sorted(failures.items()):
        experiment_variants.append(
            {
                "strategy": name,
                "status": "FAILED_EXPLICITLY",
                "failures": messages,
                **metadata.get(name, {}),
            }
        )

    metrics = pd.DataFrame(metrics_records)
    wealth = pd.DataFrame(wealth_records)
    calibrations = pd.DataFrame(calibration_records)
    mean_forecasts = pd.DataFrame(mean_records)
    covariance_diagnostics = pd.DataFrame(covariance_records)
    weight_records = []
    for name, values in sorted(complete_targets.items()):
        for date, weights in values.items():
            for asset, weight in weights.items():
                weight_records.append(
                    {"strategy": name, "decision_date": date, "asset": asset, "weight": weight}
                )
    target_weights = pd.DataFrame(weight_records)

    table1_path = output / "table_1_covariance_estimator_study.csv"
    table2_path = output / "table_2_core_strategy_comparison.csv"
    covariance_period_path = output / "covariance_forecast_periods.csv"
    solver_path = output / "solver_diagnostics.csv"
    calibration_path = output / "uncertainty_calibrations.csv"
    mean_path = output / "mean_forecast_diagnostics.csv"
    covariance_diagnostic_path = output / "covariance_estimator_diagnostics.csv"
    weights_path = output / "target_weights.csv"
    wealth_path = output / "headline_daily_wealth.csv"
    infeasibility_path = output / "infeasible_variants.csv"
    universe_path = output / "universe_snapshots.json"
    inner_folds_path = output / "inner_folds.json"
    covariance_summary.to_csv(table1_path)
    metrics.to_csv(table2_path, index=False)
    covariance_periods.to_csv(covariance_period_path, index=False)
    solver_diagnostics.to_csv(solver_path, index=False)
    calibrations.to_csv(calibration_path, index=False)
    mean_forecasts.to_csv(mean_path, index=False)
    covariance_diagnostics.to_csv(covariance_diagnostic_path, index=False)
    target_weights.to_csv(weights_path, index=False)
    wealth.to_csv(wealth_path, index=False)
    infeasible_records = [
        {"strategy": name, "failure": failure}
        for name, messages in sorted(failures.items())
        for failure in messages
    ]
    pd.DataFrame(infeasible_records, columns=["strategy", "failure"]).to_csv(
        infeasibility_path, index=False
    )
    write_json(
        universe_path,
        {
            "result_label": config.section("experiment")["result_label"],
            "snapshots": universe_records,
        },
    )
    write_json(
        inner_folds_path,
        {
            date.isoformat(): [fold.to_dict() for fold in folds_by_outer[date]]
            for date in outer_dates
        },
    )

    figure_paths = create_core_figures(
        output=output,
        metrics=metrics,
        covariance_periods=covariance_periods,
        headline_wealth=wealth,
        headline_cost_bps=float(config.section("costs")["headline_basis_points"]),
    )
    experiment_manifest_path = output / "experiment_manifest.json"
    write_json(
        experiment_manifest_path,
        {
            "result_label": config.section("experiment")["result_label"],
            "research_status": config.section("experiment")["research_status"],
            "variants": experiment_variants,
        },
    )
    artifact_paths = {
        "table_1_covariance_estimator_study": str(table1_path),
        "table_2_core_strategy_comparison": str(table2_path),
        "covariance_forecast_periods": str(covariance_period_path),
        "solver_diagnostics": str(solver_path),
        "uncertainty_calibrations": str(calibration_path),
        "mean_forecast_diagnostics": str(mean_path),
        "covariance_estimator_diagnostics": str(covariance_diagnostic_path),
        "target_weights": str(weights_path),
        "headline_daily_wealth": str(wealth_path),
        "infeasible_variants": str(infeasibility_path),
        "universe_snapshots": str(universe_path),
        "inner_folds": str(inner_folds_path),
        "experiment_manifest": str(experiment_manifest_path),
        **figure_paths,
    }
    run_manifest_path = output / "run_manifest.json"
    artifact_paths["run_manifest"] = str(run_manifest_path)
    run_manifest = {
        "schema_version": 2,
        "result_label": config.section("experiment")["result_label"],
        "research_status": config.section("experiment")["research_status"],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repository),
        "configuration": {"path": str(config.path), "canonical_sha256": config.sha256},
        "inputs": {
            "returns": {"path": str(returns_path), "sha256": sha256_file(returns_path)},
            "rebalance_dates": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
            "universe_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": dependency_versions(),
            "cvxpy_installed_solvers": cp.installed_solvers(),
        },
        "universe": {
            "mode": "SURVIVOR_PANEL",
            "label": config.section("experiment")["result_label"],
            "survivor_conditioned": True,
            "survivorship_bias_free": False,
        },
        "outer_experiment": {
            "formal_start_date": outer_dates[0].isoformat(),
            "final_decision_date": outer_dates[-1].isoformat(),
            "decision_count": len(outer_dates),
            "reason": walk["outer_start_rule"],
            "minimum_prior_inner_folds": int(walk["minimum_prior_inner_folds"]),
            "first_outer_inner_folds": [
                fold.to_dict() for fold in folds_by_outer[outer_dates[0]]
            ],
        },
        "estimation": {
            "mean_estimators": mean_methods,
            "headline_mean": headline_mean,
            "covariance_estimators": covariance_methods,
            "headline_covariance": headline_covariance,
        },
        "bootstrap": {
            "method": config.section("uncertainty")["bootstrap_method"],
            "base_seed": config.section("uncertainty")["bootstrap_seed"],
            "replications": config.section("uncertainty")["bootstrap_replications"],
            "block_length": config.section("uncertainty")["block_length_observations"],
            "coverage_probability": config.section("uncertainty")["coverage_probability"],
        },
        "risk_targets": risk_targets,
        "cost_scenarios_bps": cost_scenarios,
        "execution_convention": config.section("clock"),
        "artifact_locations": artifact_paths,
        "variant_counts": {
            "completed_strategy_cost_scenarios": len(metrics),
            "explicit_failures": len(failures),
        },
        "limitations": list(config.payload["limitations"]),
    }
    write_json(run_manifest_path, run_manifest)
    return {
        "output_directory": str(output),
        "outer_dates": outer_dates,
        "covariance_summary": covariance_summary,
        "metrics": metrics,
        "failures": failures,
        "manifest": run_manifest,
        "artifact_paths": artifact_paths,
    }
