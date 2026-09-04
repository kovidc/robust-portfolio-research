"""Final robustness, redundancy, inference, and reporting analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import cvxpy as cp
import numpy as np
import pandas as pd

from robust_portfolio.calibration import calibrate_risk_aversion
from robust_portfolio.data import FrozenCsvReturnProvider
from robust_portfolio.data.providers import sha256_file
from robust_portfolio.data.schemas import ReturnPanel
from robust_portfolio.estimators import calibrate_uncertainty, estimate_covariance, estimate_mean
from robust_portfolio.estimators.covariance import iewma_covariance, nearest_psd
from robust_portfolio.estimators.uncertainty import circular_block_bootstrap_indices
from robust_portfolio.inference import bootstrap_headline_statistics, deflated_sharpe_probability
from robust_portfolio.optimizers import (
    OptimizationFailure,
    diagonal_robust_covariance,
    global_minimum_variance,
    solve_target_risk,
)
from robust_portfolio.reporting.manifests import dependency_versions, git_state
from robust_portfolio.reporting.metrics import maximum_drawdown, scenario_metrics
from robust_portfolio.reporting.core_outputs import prepare_output_directory, write_json
from robust_portfolio.reporting.final_outputs import create_final_figures

from .clustering import (
    cluster_medoids,
    correlation_distance,
    covariance_spectrum,
    hierarchical_clusters,
)
from .final_analysis_configuration import FinalAnalysisConfig
from .regimes import classify_regimes
from .robustness import (
    allocation_diagnostics,
    asset_class_l1_change,
    clone_distortions,
    psd_covariance_perturbations,
)
from .simulation import simulate_targets


@dataclass(frozen=True)
class DateInputs:
    date: pd.Timestamp
    panel: ReturnPanel
    mean: pd.Series
    covariance: pd.DataFrame
    standard_errors: pd.Series
    mean_error_covariance: pd.DataFrame
    box_rho: float
    ellipsoid_rho: float
    kappa: float


MODEL_TO_STRATEGY = {
    "nominal": "nominal_risk_10pct",
    "box": "box_risk_10pct",
    "box_diagonal": "box_diagonal_risk_10pct",
    "ellipsoid": "ellipsoid_risk_10pct",
}


def _load_core_targets(path: Path) -> dict[str, dict[pd.Timestamp, pd.Series]]:
    frame = pd.read_csv(path, parse_dates=["decision_date"])
    output = {}
    for (strategy, date), group in frame.groupby(["strategy", "decision_date"], sort=False):
        output.setdefault(strategy, {})[pd.Timestamp(date)] = group.set_index("asset")["weight"].astype(float)
    return output


def _asset_classes(path: Path, assets) -> pd.Series:
    metadata = pd.read_csv(path)
    kept = metadata[metadata["kept_after_cleaning"].astype(str).str.lower() == "true"]
    result = kept.set_index("ticker")["asset_class"].astype(str).reindex(assets)
    if result.isna().any():
        raise ValueError("Static asset-class metadata are incomplete.")
    return result


def _validate_core(config: FinalAnalysisConfig, root: Path) -> tuple[Path, dict]:
    core = config.repository_path(root, "core_artifact_directory")
    manifest_path = core / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config.section("experiment")["core_config_sha256"]
    if manifest["configuration"]["canonical_sha256"] != expected:
        raise ValueError("Core experiment artifact configuration hash does not match final analysis.")
    expected_commit = config.section("experiment")["core_commit"]
    if manifest.get("git", {}).get("commit") != expected_commit:
        raise ValueError("Core experiment manifest does not identify the configured core experiment commit.")
    for record in manifest["inputs"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Core experiment input hash changed: {path}")
    required = set(manifest["artifact_locations"].values())
    missing = sorted(path for path in required if not Path(path).exists())
    if missing:
        raise FileNotFoundError(f"Core experiment artifacts are incomplete: {missing}")
    return core, manifest


def _core_kwargs(core_config: dict) -> tuple[dict, dict, dict]:
    covariance = core_config["covariances"]
    annualization = int(core_config["annualization"]["trading_days"])
    covariance_kwargs = {
        "annualization_factor": annualization,
        "ewma_half_life": float(covariance["ewma_half_life"]),
        "iewma_volatility_half_life": float(covariance["iewma_volatility_half_life"]),
        "iewma_correlation_half_life": float(covariance["iewma_correlation_half_life"]),
        "iewma_winsorize_clip": float(covariance["iewma_winsorize_clip"]),
        "iewma_variance_floor": float(covariance["iewma_variance_floor"]),
        "absolute_eigenvalue_floor": float(covariance["absolute_eigenvalue_floor"]),
        "relative_eigenvalue_floor": float(covariance["relative_eigenvalue_floor"]),
    }
    mean = core_config["means"]
    mean_kwargs = {
        "annualization_factor": annualization,
        "ewma_half_life": float(mean["ewma_half_life"]),
        "shrinkage_intensity": float(mean["shrinkage_intensity"]),
    }
    return covariance_kwargs, mean_kwargs, core_config["optimization"]


def _build_date_inputs(
    provider: FrozenCsvReturnProvider,
    dates: pd.DatetimeIndex,
    core_config: dict,
) -> dict[pd.Timestamp, DateInputs]:
    covariance_kwargs, mean_kwargs, _ = _core_kwargs(core_config)
    uncertainty = core_config["uncertainty"]
    window = int(core_config["walkforward"]["estimation_window_observations"])
    output = {}
    for date in dates:
        panel = provider.panel(as_of=date, trailing_observations=window)
        mean = estimate_mean(panel, core_config["means"]["headline"], **mean_kwargs).annualized_mean
        covariance = estimate_covariance(
            panel, core_config["covariances"]["headline"], **covariance_kwargs
        ).annualized_covariance
        seed = int(uncertainty["bootstrap_seed"]) + int(date.strftime("%Y%m%d"))
        calibrated = calibrate_uncertainty(
            panel,
            bootstrap_seed=seed,
            bootstrap_replications=int(uncertainty["bootstrap_replications"]),
            block_length=int(uncertainty["block_length_observations"]),
            coverage_probability=float(uncertainty["coverage_probability"]),
            annualization_factor=int(core_config["annualization"]["trading_days"]),
            standard_error_floor=float(uncertainty["standard_error_floor"]),
            relative_eigenvalue_floor=float(uncertainty["mean_uncertainty_relative_eigenvalue_floor"]),
            absolute_eigenvalue_floor=float(uncertainty["mean_uncertainty_absolute_eigenvalue_floor"]),
            iewma_volatility_half_life=float(core_config["covariances"]["iewma_volatility_half_life"]),
            iewma_variance_floor=float(core_config["covariances"]["iewma_variance_floor"]),
        )
        output[pd.Timestamp(date)] = DateInputs(
            pd.Timestamp(date), panel, mean, covariance, calibrated.standard_errors,
            calibrated.mean_error_covariance, calibrated.box_rho,
            calibrated.ellipsoid_rho, calibrated.diagonal_kappa,
        )
    return output


def _model_problem(inputs: DateInputs, model: str) -> tuple[pd.DataFrame, dict]:
    decision = inputs.covariance
    arguments = {}
    if model in {"box", "box_diagonal"}:
        arguments = {"standard_errors": inputs.standard_errors, "box_rho": inputs.box_rho}
    elif model == "ellipsoid":
        arguments = {
            "mean_error_covariance": inputs.mean_error_covariance,
            "ellipsoid_rho": inputs.ellipsoid_rho,
        }
    elif model != "nominal":
        raise ValueError(f"Unknown model: {model}")
    if model == "box_diagonal":
        decision = pd.DataFrame(
            diagonal_robust_covariance(inputs.covariance, inputs.kappa),
            index=inputs.covariance.index,
            columns=inputs.covariance.columns,
        )
    return decision, arguments


def _solve_ceiling(
    inputs: DateInputs,
    model: str,
    *,
    target: float,
    maximum_weight: float,
    solver_order: list[str],
    mean: pd.Series | None = None,
    covariance: pd.DataFrame | None = None,
    arguments_override: dict | None = None,
):
    active = inputs
    if mean is not None or covariance is not None:
        active = DateInputs(
            inputs.date, inputs.panel, inputs.mean if mean is None else mean,
            inputs.covariance if covariance is None else covariance,
            inputs.standard_errors, inputs.mean_error_covariance,
            inputs.box_rho, inputs.ellipsoid_rho, inputs.kappa,
        )
    decision, arguments = _model_problem(active, model)
    if arguments_override:
        arguments.update(arguments_override)
    return solve_target_risk(
        active.mean, decision, target_volatility=target,
        maximum_weight=maximum_weight, solver_order=solver_order,
        common_base_covariance=active.covariance, **arguments,
    )


def _robust_return(inputs: DateInputs, model: str, weights: pd.Series, *, mean=None) -> float:
    mu = inputs.mean if mean is None else mean
    value = float(mu.reindex(weights.index) @ weights)
    if model in {"box", "box_diagonal"}:
        value -= inputs.box_rho * float(inputs.standard_errors.reindex(weights.index) @ weights.abs())
    elif model == "ellipsoid":
        covariance = inputs.mean_error_covariance.reindex(index=weights.index, columns=weights.index)
        value -= inputs.ellipsoid_rho * np.sqrt(max(float(weights @ covariance @ weights), 0.0))
    return float(value)


def _risk_diagnostics(
    date_inputs: dict[pd.Timestamp, DateInputs],
    core: Path,
    config: FinalAnalysisConfig,
    core_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[pd.Timestamp, pd.Series]]]:
    solver = pd.read_csv(core / "solver_diagnostics.csv", parse_dates=["decision_date"])
    solver = solver[solver["strategy"].str.contains("_risk_")].copy()
    _, _, optimization = _core_kwargs(core_config)
    solvers = list(optimization["solver_order_conic"])
    maximum_weight = float(core_config["constraints"]["maximum_weight"])
    attainment_config = config.section("risk_attainment")
    bounds = {}
    attainment_records = []
    attainment_targets: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    for date, inputs in date_inputs.items():
        for model in config.section("comparison")["principal_optimized_models"]:
            decision, arguments = _model_problem(inputs, model)
            minimum = global_minimum_variance(
                decision, maximum_weight=maximum_weight, solver_order=solvers
            )
            # Calibration computes and records the deterministic zero-gamma endpoint.
            for target in [float(x) for x in attainment_config["targets"]]:
                result = calibrate_risk_aversion(
                    inputs.mean, decision, target_volatility=target,
                    maximum_weight=maximum_weight, solver_order=solvers,
                    common_base_covariance=inputs.covariance, **arguments,
                    volatility_tolerance=float(attainment_config["volatility_tolerance"]),
                    initial_upper_risk_aversion=float(attainment_config["initial_upper_risk_aversion"]),
                    maximum_risk_aversion=float(attainment_config["maximum_risk_aversion"]),
                    maximum_bisection_iterations=int(attainment_config["maximum_bisection_iterations"]),
                )
                key = (date, model)
                bounds[key] = {
                    "minimum_feasible_volatility": minimum.predicted_volatility,
                    "zero_risk_aversion_volatility": result.zero_risk_aversion_volatility,
                }
                solution = result.solution
                attainment_records.append(
                    {
                        "comparison_type": "TARGET_ATTAINMENT",
                        "decision_date": date,
                        "model": model,
                        "requested_volatility": target,
                        "status": result.status,
                        "attained_predicted_decision_volatility": None if solution is None else solution.predicted_volatility,
                        "attained_predicted_common_base_volatility": None if solution is None else solution.common_base_volatility,
                        "risk_aversion": None if solution is None else solution.risk_aversion,
                        "minimum_feasible_volatility": result.minimum_feasible_volatility,
                        "zero_risk_aversion_volatility": result.zero_risk_aversion_volatility,
                        "absolute_target_error": None if solution is None else abs(solution.predicted_volatility - target),
                        "iterations": result.iterations,
                        "reason": result.reason,
                    }
                )
                if solution is not None:
                    name = f"{model}_target_attainment_{int(round(target*100)):02d}pct"
                    attainment_targets.setdefault(name, {})[date] = solution.weights

    def parse_model(name: str) -> str:
        if name.startswith("box_diagonal"):
            return "box_diagonal"
        return name.split("_risk_")[0]

    solver["model"] = solver["strategy"].map(parse_model)
    solver["comparison_type"] = "COMMON EX-ANTE RISK CEILING"
    solver["requested_volatility"] = solver["target_volatility"]
    solver["attained_predicted_decision_volatility"] = solver["predicted_decision_volatility"]
    solver["attained_predicted_common_base_volatility"] = solver["predicted_common_base_volatility"]
    solver["slack"] = solver["requested_volatility"] - solver["attained_predicted_decision_volatility"]
    solver["binding_indicator"] = solver["target_binding"].astype(bool)
    solver["minimum_feasible_volatility"] = [
        bounds[(pd.Timestamp(date), model)]["minimum_feasible_volatility"]
        for date, model in zip(solver["decision_date"], solver["model"])
    ]
    solver["zero_risk_aversion_volatility"] = [
        bounds[(pd.Timestamp(date), model)]["zero_risk_aversion_volatility"]
        for date, model in zip(solver["decision_date"], solver["model"])
    ]
    ceiling_columns = [
        "comparison_type", "decision_date", "strategy", "model",
        "requested_volatility", "attained_predicted_decision_volatility",
        "attained_predicted_common_base_volatility", "slack", "binding_indicator",
        "minimum_feasible_volatility", "zero_risk_aversion_volatility",
        "status", "solver",
    ]
    present = set(
        zip(solver["decision_date"], solver["model"], solver["requested_volatility"])
    )
    missing_records = []
    risk_targets = [
        float(value)
        for value in core_config["risk_matching"]["target_annual_volatility"]
    ]
    for date in date_inputs:
        for model in config.section("comparison")["principal_optimized_models"]:
            for target in risk_targets:
                if (date, model, target) in present:
                    continue
                missing_records.append(
                    {
                        "comparison_type": "COMMON EX-ANTE RISK CEILING",
                        "decision_date": date,
                        "strategy": f"{model}_risk_{int(round(target * 100)):02d}pct",
                        "model": model,
                        "requested_volatility": target,
                        "attained_predicted_decision_volatility": np.nan,
                        "attained_predicted_common_base_volatility": np.nan,
                        "slack": np.nan,
                        "binding_indicator": False,
                        "minimum_feasible_volatility": bounds[(date, model)][
                            "minimum_feasible_volatility"
                        ],
                        "zero_risk_aversion_volatility": bounds[(date, model)][
                            "zero_risk_aversion_volatility"
                        ],
                        "status": "TARGET_NOT_FEASIBLE",
                        "solver": None,
                    }
                )
    complete = pd.concat(
        [solver[ceiling_columns], pd.DataFrame(missing_records, columns=ceiling_columns)],
        ignore_index=True,
    ).sort_values(["decision_date", "model", "requested_volatility"])
    return complete, pd.DataFrame(attainment_records), attainment_targets


def _direct_robustness(
    date_inputs: dict[pd.Timestamp, DateInputs],
    baseline_targets: dict[str, dict[pd.Timestamp, pd.Series]],
    classes: pd.Series,
    config: FinalAnalysisConfig,
    core_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = config.section("direct_robustness")
    target = float(config.section("comparison")["headline_target_annual_volatility"])
    maximum_weight = float(core_config["constraints"]["maximum_weight"])
    solvers = list(core_config["optimization"]["solver_order_conic"])
    annualization = int(core_config["annualization"]["trading_days"])
    covariance_config = core_config["covariances"]
    records = []
    failures = []
    for date_position, date_text in enumerate(direct["selected_outer_dates"]):
        date = pd.Timestamp(date_text)
        inputs = date_inputs[date]
        x = inputs.panel.values.to_numpy(dtype=float)
        indices = circular_block_bootstrap_indices(
            len(x), int(direct["bootstrap_replications"]),
            int(direct["block_length_observations"]),
            int(direct["seed"]) + date_position,
        )
        for replication, sample_indices in enumerate(indices):
            sample = x[sample_indices]
            sample_mean = pd.Series(sample.mean(axis=0) * annualization, index=inputs.mean.index)
            daily_covariance = iewma_covariance(
                sample,
                volatility_half_life=float(covariance_config["iewma_volatility_half_life"]),
                correlation_half_life=float(covariance_config["iewma_correlation_half_life"]),
                winsorize_clip=float(covariance_config["iewma_winsorize_clip"]),
                variance_floor=float(covariance_config["iewma_variance_floor"]),
            )
            psd, _, _ = nearest_psd(
                daily_covariance * annualization,
                absolute_floor=float(covariance_config["absolute_eigenvalue_floor"]),
                relative_floor=float(covariance_config["relative_eigenvalue_floor"]),
            )
            sample_covariance = pd.DataFrame(psd, index=inputs.mean.index, columns=inputs.mean.index)
            perturbed = DateInputs(
                date, inputs.panel, sample_mean, sample_covariance,
                inputs.standard_errors, inputs.mean_error_covariance,
                inputs.box_rho, inputs.ellipsoid_rho, inputs.kappa,
            )
            for model, strategy in MODEL_TO_STRATEGY.items():
                baseline = baseline_targets[strategy][date]
                try:
                    result = _solve_ceiling(
                        perturbed, model, target=target, maximum_weight=maximum_weight,
                        solver_order=solvers,
                    )
                    diagnostic = allocation_diagnostics(result.weights, baseline)
                    baseline_risk = float(np.sqrt(max(float(
                        baseline @ sample_covariance @ baseline
                    ), 0.0)))
                    records.append(
                        {
                            "perturbation_kind": "training_block_bootstrap",
                            "perturbation": f"replication_{replication:03d}",
                            "decision_date": date,
                            "model": model,
                            "replication": replication,
                            "status": "COMPLETED",
                            **diagnostic,
                            "asset_class_exposure_l1_change": asset_class_l1_change(
                                result.weights, baseline, classes
                            ),
                            "predicted_volatility_change": result.common_base_volatility - baseline_risk,
                            "robust_return_change": _robust_return(perturbed, model, result.weights)
                            - _robust_return(perturbed, model, baseline),
                        }
                    )
                except Exception as error:
                    failures.append(
                        {
                            "experiment": "training_block_bootstrap",
                            "decision_date": date, "model": model,
                            "perturbation": f"replication_{replication:03d}",
                            "status": "FAILED_EXPLICITLY", "error": str(error),
                        }
                    )

        for multiplier in [float(x) for x in direct["mean_standard_error_shocks"]]:
            shocked_mean = inputs.mean + multiplier * inputs.standard_errors
            shocked_inputs = DateInputs(
                date, inputs.panel, shocked_mean, inputs.covariance,
                inputs.standard_errors, inputs.mean_error_covariance,
                inputs.box_rho, inputs.ellipsoid_rho, inputs.kappa,
            )
            for model, strategy in MODEL_TO_STRATEGY.items():
                baseline = baseline_targets[strategy][date]
                label = f"mu_{multiplier:+g}s"
                try:
                    result = _solve_ceiling(
                        shocked_inputs, model, target=target,
                        maximum_weight=maximum_weight, solver_order=solvers,
                    )
                    records.append(
                        {
                            "perturbation_kind": "mean_standard_error_shock",
                            "perturbation": label, "decision_date": date, "model": model,
                            "replication": None, "status": "COMPLETED",
                            **allocation_diagnostics(result.weights, baseline),
                            "asset_class_exposure_l1_change": asset_class_l1_change(
                                result.weights, baseline, classes
                            ),
                            "predicted_volatility_change": result.common_base_volatility
                            - float(np.sqrt(max(float(baseline @ inputs.covariance @ baseline), 0.0))),
                            "robust_return_change": _robust_return(
                                shocked_inputs, model, result.weights
                            ) - _robust_return(shocked_inputs, model, baseline),
                        }
                    )
                except Exception as error:
                    failures.append(
                        {"experiment": "mean_standard_error_shock", "decision_date": date,
                         "model": model, "perturbation": label,
                         "status": "FAILED_EXPLICITLY", "error": str(error)}
                    )

        shocks = psd_covariance_perturbations(
            inputs.covariance,
            variance_scale=float(direct["covariance_shocks"]["variance_scale"]),
            correlation_to_identity_weight=float(
                direct["covariance_shocks"]["correlation_to_identity_weight"]
            ),
            leading_eigenvalue_scale=float(
                direct["covariance_shocks"]["leading_eigenvalue_scale"]
            ),
        )
        for label, shocked_covariance in shocks.items():
            shocked_inputs = DateInputs(
                date, inputs.panel, inputs.mean, shocked_covariance,
                inputs.standard_errors, inputs.mean_error_covariance,
                inputs.box_rho, inputs.ellipsoid_rho, inputs.kappa,
            )
            for model, strategy in MODEL_TO_STRATEGY.items():
                baseline = baseline_targets[strategy][date]
                try:
                    result = _solve_ceiling(
                        shocked_inputs, model, target=target,
                        maximum_weight=maximum_weight, solver_order=solvers,
                    )
                    baseline_risk = float(np.sqrt(max(float(
                        baseline @ shocked_covariance @ baseline
                    ), 0.0)))
                    records.append(
                        {
                            "perturbation_kind": "psd_covariance_shock",
                            "perturbation": label, "decision_date": date, "model": model,
                            "replication": None, "status": "COMPLETED",
                            **allocation_diagnostics(result.weights, baseline),
                            "asset_class_exposure_l1_change": asset_class_l1_change(
                                result.weights, baseline, classes
                            ),
                            "predicted_volatility_change": result.common_base_volatility - baseline_risk,
                            "robust_return_change": _robust_return(inputs, model, result.weights)
                            - _robust_return(inputs, model, baseline),
                        }
                    )
                except Exception as error:
                    failures.append(
                        {"experiment": "psd_covariance_shock", "decision_date": date,
                         "model": model, "perturbation": label,
                         "status": "FAILED_EXPLICITLY", "error": str(error)}
                    )
    return pd.DataFrame(records), pd.DataFrame(failures)


def _clone_experiment(
    date_inputs: dict[pd.Timestamp, DateInputs],
    baseline_targets: dict[str, dict[pd.Timestamp, pd.Series]],
    classes: pd.Series,
    config: FinalAnalysisConfig,
    core_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clone_config = config.section("clone_experiment")
    target = float(config.section("comparison")["headline_target_annual_volatility"])
    maximum_weight = float(core_config["constraints"]["maximum_weight"])
    solvers = list(core_config["optimization"]["solver_order_conic"])
    covariance_kwargs, mean_kwargs, _ = _core_kwargs(core_config)
    records, failures = [], []
    for date_position, date_text in enumerate(clone_config["selected_outer_dates"]):
        date = pd.Timestamp(date_text)
        base = date_inputs[date]
        base_values = base.panel.values
        for source_position, source in enumerate(clone_config["source_assets"]):
            source_scale = float(base_values[source].std(ddof=1))
            for noise_position, relative_noise in enumerate(
                clone_config["relative_noise_standard_deviations"]
            ):
                relative_noise = float(relative_noise)
                clone = f"{source}__SYNTHETIC_CLONE"
                seed = int(clone_config["seed"]) + 100 * date_position + 10 * source_position + noise_position
                generator = np.random.default_rng(seed)
                augmented_values = base_values.copy()
                augmented_values[clone] = base_values[source] + generator.normal(
                    0.0, relative_noise * source_scale, len(base_values)
                )
                augmented_panel = ReturnPanel(date, augmented_values, base.panel.source_sha256)
                mean = estimate_mean(
                    augmented_panel, core_config["means"]["headline"], **mean_kwargs
                ).annualized_mean
                covariance = estimate_covariance(
                    augmented_panel, core_config["covariances"]["headline"], **covariance_kwargs
                ).annualized_covariance
                augmented_assets = covariance.index
                standard_errors = base.standard_errors.reindex(augmented_assets).copy()
                standard_errors.loc[clone] = float(base.standard_errors[source]) * np.sqrt(1.0 + relative_noise**2)
                error = np.zeros((len(augmented_assets), len(augmented_assets)))
                original_assets = list(base.mean.index)
                error[:-1, :-1] = base.mean_error_covariance.reindex(
                    index=original_assets, columns=original_assets
                ).to_numpy()
                source_index = original_assets.index(source)
                error[-1, :-1] = error[source_index, :-1]
                error[:-1, -1] = error[:-1, source_index]
                error[-1, -1] = error[source_index, source_index] * (1.0 + relative_noise**2)
                error, _, _ = nearest_psd(error, absolute_floor=1e-10, relative_floor=1e-6)
                augmented_inputs = DateInputs(
                    date, augmented_panel, mean, covariance, standard_errors,
                    pd.DataFrame(error, index=augmented_assets, columns=augmented_assets),
                    base.box_rho, base.ellipsoid_rho, base.kappa,
                )
                augmented_classes = classes.copy()
                augmented_classes.loc[clone] = classes.loc[source]
                for model, strategy in MODEL_TO_STRATEGY.items():
                    baseline = baseline_targets[strategy][date]
                    try:
                        result = _solve_ceiling(
                            augmented_inputs, model, target=target,
                            maximum_weight=maximum_weight, solver_order=solvers,
                        )
                        diagnostic = clone_distortions(
                            result.weights, baseline, source_asset=source, clone_asset=clone,
                            asset_classes=augmented_classes,
                        )
                        baseline_augmented = baseline.reindex(augmented_assets, fill_value=0.0)
                        candidate_risk = result.common_base_volatility
                        baseline_risk = float(np.sqrt(max(float(
                            baseline_augmented @ covariance @ baseline_augmented
                        ), 0.0)))
                        records.append(
                            {"decision_date": date, "model": model, "source_asset": source,
                             "clone_asset": clone, "relative_noise_standard_deviation": relative_noise,
                             "seed": seed, "status": "COMPLETED", **diagnostic,
                             "security_one_way_turnover_implication": 0.5 * diagnostic["l1_weight_change"],
                             "economic_one_way_turnover_implication": 0.5 * diagnostic["economic_exposure_l1_change"],
                             "predicted_common_base_volatility_change": candidate_risk - baseline_risk}
                        )
                    except Exception as error_value:
                        failures.append(
                            {"experiment": "synthetic_clone", "decision_date": date,
                             "model": model, "source_asset": source,
                             "relative_noise_standard_deviation": relative_noise,
                             "status": "FAILED_EXPLICITLY", "error": str(error_value)}
                        )
    return pd.DataFrame(records), pd.DataFrame(failures)


def _cluster_experiment(
    date_inputs: dict[pd.Timestamp, DateInputs],
    baseline_targets: dict[str, dict[pd.Timestamp, pd.Series]],
    provider: FrozenCsvReturnProvider,
    classes: pd.Series,
    config: FinalAnalysisConfig,
    core_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_config = config.section("clustering")
    target = float(config.section("comparison")["headline_target_annual_volatility"])
    maximum_weight = float(core_config["constraints"]["maximum_weight"])
    quadratic_solvers = list(core_config["optimization"]["solver_order_quadratic"])
    conic_solvers = list(core_config["optimization"]["solver_order_conic"])
    covariance_kwargs, mean_kwargs, _ = _core_kwargs(core_config)
    full_names = {"gmv": "gmv_iewma", **MODEL_TO_STRATEGY}
    date_records, failures = [], []
    target_maps: dict[tuple[float, str], dict[pd.Timestamp, pd.Series]] = {}
    for date, base in date_inputs.items():
        distance = correlation_distance(base.panel.values)
        for threshold in [float(x) for x in cluster_config["correlation_thresholds"]]:
            clusters = hierarchical_clusters(
                distance, correlation_threshold=threshold, method=cluster_config["linkage"]
            )
            medoids = cluster_medoids(distance, clusters)
            medoid_panel = ReturnPanel(
                date, base.panel.values.loc[:, list(medoids)], base.panel.source_sha256
            )
            mean = estimate_mean(
                medoid_panel, core_config["means"]["headline"], **mean_kwargs
            ).annualized_mean
            covariance = estimate_covariance(
                medoid_panel, core_config["covariances"]["headline"], **covariance_kwargs
            ).annualized_covariance
            medoid_inputs = DateInputs(
                date, medoid_panel, mean, covariance,
                base.standard_errors.reindex(medoids),
                base.mean_error_covariance.reindex(index=medoids, columns=medoids),
                base.box_rho, base.ellipsoid_rho, base.kappa,
            )
            full_spectrum = covariance_spectrum(base.covariance)
            medoid_spectrum = covariance_spectrum(covariance)
            for model in cluster_config["models"]:
                baseline = baseline_targets[full_names[model]][date]
                try:
                    if model == "gmv":
                        result = global_minimum_variance(
                            covariance, maximum_weight=maximum_weight,
                            solver_order=quadratic_solvers,
                        )
                    else:
                        result = _solve_ceiling(
                            medoid_inputs, model, target=target,
                            maximum_weight=maximum_weight, solver_order=conic_solvers,
                        )
                    aligned = result.weights.reindex(provider.assets, fill_value=0.0)
                    target_maps.setdefault((threshold, model), {})[date] = aligned
                    cluster_exposure_change = 0.0
                    for cluster_id, members in clusters.groupby(clusters):
                        member_assets = list(members.index)
                        cluster_exposure_change += abs(
                            float(aligned.reindex(member_assets, fill_value=0.0).sum())
                            - float(baseline.reindex(member_assets, fill_value=0.0).sum())
                        )
                    date_records.append(
                        {
                            "decision_date": date, "threshold": threshold, "model": model,
                            "status": "COMPLETED", "full_asset_count": len(base.mean),
                            "cluster_count": len(medoids), "medoid_asset_count": len(medoids),
                            "full_condition_number": full_spectrum["condition_number"],
                            "medoid_condition_number": medoid_spectrum["condition_number"],
                            "full_effective_rank": full_spectrum["effective_rank"],
                            "medoid_effective_rank": medoid_spectrum["effective_rank"],
                            "full_minimum_eigenvalue": full_spectrum["minimum_eigenvalue"],
                            "medoid_minimum_eigenvalue": medoid_spectrum["minimum_eigenvalue"],
                            "security_level_l1_change": allocation_diagnostics(aligned, baseline)["l1_weight_change"],
                            "cluster_exposure_l1_change": cluster_exposure_change,
                            "asset_class_exposure_l1_change": asset_class_l1_change(
                                aligned, baseline, classes
                            ),
                            "predicted_decision_volatility": result.predicted_volatility,
                            "predicted_common_base_volatility": float(np.sqrt(max(float(
                                aligned @ base.covariance @ aligned
                            ), 0.0))),
                        }
                    )
                except Exception as error:
                    date_records.append(
                        {
                            "decision_date": date, "threshold": threshold, "model": model,
                            "status": "FAILED_EXPLICITLY", "full_asset_count": len(base.mean),
                            "cluster_count": len(medoids), "medoid_asset_count": len(medoids),
                            "full_condition_number": full_spectrum["condition_number"],
                            "medoid_condition_number": medoid_spectrum["condition_number"],
                            "full_effective_rank": full_spectrum["effective_rank"],
                            "medoid_effective_rank": medoid_spectrum["effective_rank"],
                            "full_minimum_eigenvalue": full_spectrum["minimum_eigenvalue"],
                            "medoid_minimum_eigenvalue": medoid_spectrum["minimum_eigenvalue"],
                            "security_level_l1_change": np.nan,
                            "cluster_exposure_l1_change": np.nan,
                            "asset_class_exposure_l1_change": np.nan,
                            "predicted_decision_volatility": np.nan,
                            "predicted_common_base_volatility": np.nan,
                        }
                    )
                    failures.append(
                        {"experiment": "cluster_medoid", "decision_date": date,
                         "threshold": threshold, "model": model,
                         "status": "FAILED_EXPLICITLY", "error": str(error)}
                    )

    summary_records = []
    for (threshold, model), targets in sorted(target_maps.items()):
        completed = len(targets) == len(date_inputs)
        if not completed:
            diagnostics = pd.DataFrame(date_records)
            selected = diagnostics[
                (diagnostics["threshold"] == threshold) & (diagnostics["model"] == model)
            ]
            summary_records.append(
                {"universe": "CLUSTER_MEDOID", "threshold": threshold, "model": model,
                 "status": "INCOMPLETE_FAILED_SOLVES", "completed_dates": len(targets),
                 "average_asset_count": float(selected["medoid_asset_count"].mean()),
                 "average_cluster_count": float(selected["cluster_count"].mean()),
                 "median_condition_number": float(selected["medoid_condition_number"].median()),
                 "median_effective_rank": float(selected["medoid_effective_rank"].median()),
                 "average_security_level_l1_change": float(selected["security_level_l1_change"].mean()),
                 "average_cluster_exposure_l1_change": float(selected["cluster_exposure_l1_change"].mean())}
            )
            continue
        path = simulate_targets(
            provider, targets, strategy=f"cluster_{threshold}_{model}",
            cost_bps=float(config.section("comparison")["headline_cost_bps"]),
            maximum_weight=maximum_weight,
            cash_daily_return=float(core_config["annualization"]["cash_daily_return"]),
        )
        metrics = scenario_metrics(
            path, targets, annualization_factor=int(core_config["annualization"]["trading_days"])
        )
        diagnostics = pd.DataFrame(date_records)
        selected = diagnostics[(diagnostics["threshold"] == threshold) & (diagnostics["model"] == model)]
        summary_records.append(
            {"universe": "CLUSTER_MEDOID", "threshold": threshold, "model": model,
             "status": "COMPLETED", "completed_dates": len(targets),
             "average_asset_count": float(selected["medoid_asset_count"].mean()),
             "average_cluster_count": float(selected["cluster_count"].mean()),
             "median_condition_number": float(selected["medoid_condition_number"].median()),
             "median_effective_rank": float(selected["medoid_effective_rank"].median()),
             "average_security_level_l1_change": float(selected["security_level_l1_change"].mean()),
             "average_cluster_exposure_l1_change": float(selected["cluster_exposure_l1_change"].mean()),
             **metrics}
        )
    return pd.DataFrame(date_records), pd.DataFrame(summary_records), pd.DataFrame(failures)


def _overlap_table(ceiling: pd.DataFrame, metrics: pd.DataFrame, headline_cost: float) -> pd.DataFrame:
    risk = ceiling.groupby(["model", "strategy"], as_index=False).agg(
        average_attained_model_risk=("attained_predicted_decision_volatility", "mean"),
        average_attained_common_base_risk=("attained_predicted_common_base_volatility", "mean"),
    )
    performance = metrics[metrics["cost_bps"] == headline_cost][
        ["strategy", "realized_volatility", "net_annualized_return", "provisional_zero_rf_sharpe"]
    ]
    joined = risk.merge(performance, on="strategy", how="left")
    ranges = joined.groupby("model").agg(
        minimum_predicted_risk=("average_attained_model_risk", "min"),
        maximum_predicted_risk=("average_attained_model_risk", "max"),
        minimum_realized_risk=("realized_volatility", "min"),
        maximum_realized_risk=("realized_volatility", "max"),
    )
    predicted_low = float(ranges["minimum_predicted_risk"].max())
    predicted_high = float(ranges["maximum_predicted_risk"].min())
    common_ranges = joined.groupby("model")["average_attained_common_base_risk"].agg(["min", "max"])
    common_low = float(common_ranges["min"].max())
    common_high = float(common_ranges["max"].min())
    realized_low = float(ranges["minimum_realized_risk"].max())
    realized_high = float(ranges["maximum_realized_risk"].min())
    joined["predicted_overlap_low"] = predicted_low
    joined["predicted_overlap_high"] = predicted_high
    joined["inside_predicted_overlap"] = (
        (joined["average_attained_model_risk"] >= predicted_low)
        & (joined["average_attained_model_risk"] <= predicted_high)
        & (predicted_low <= predicted_high)
    )
    joined["common_base_predicted_overlap_low"] = common_low
    joined["common_base_predicted_overlap_high"] = common_high
    joined["inside_common_base_predicted_overlap"] = (
        (joined["average_attained_common_base_risk"] >= common_low)
        & (joined["average_attained_common_base_risk"] <= common_high)
        & (common_low <= common_high)
    )
    joined["realized_overlap_low_descriptive"] = realized_low
    joined["realized_overlap_high_descriptive"] = realized_high
    joined["realized_overlap_is_post_hoc"] = True
    return joined


def _simulate_core_paths(
    provider: FrozenCsvReturnProvider,
    targets: dict[str, dict[pd.Timestamp, pd.Series]],
    *,
    strategies: list[str] | tuple[str, ...],
    cost_bps: float,
    core_config: dict,
):
    maximum_weight = float(core_config["constraints"]["maximum_weight"])
    return {
        strategy: simulate_targets(
            provider, targets[strategy], strategy=strategy, cost_bps=cost_bps,
            maximum_weight=maximum_weight,
            cash_daily_return=float(core_config["annualization"]["cash_daily_return"]),
        )
        for strategy in strategies
    }


def _inference(
    paths: dict,
    all_paths: dict,
    core_metrics: pd.DataFrame,
    config: FinalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inference = config.section("inference")
    joint = pd.concat(
        {name: path.daily["net_return"] for name, path in paths.items()}, axis=1
    ).dropna()
    intervals, differences, indices = bootstrap_headline_statistics(
        joint,
        replications=int(inference["replications"]),
        expected_block_length=float(inference["expected_block_length"]),
        seed=int(inference["seed"]),
        confidence_level=float(inference["confidence_level"]),
        annualization_factor=int(inference["annualization_factor"]),
        certainty_equivalent_risk_aversion=float(
            inference["certainty_equivalent_risk_aversion"]
        ),
        comparators=("nominal_risk_10pct", "etf_equal_weight"),
    )
    index_frame = pd.DataFrame(indices)
    headline_cost = float(config.section("comparison")["headline_cost_bps"])
    trial_sharpes = core_metrics[
        core_metrics["cost_bps"] == headline_cost
    ]["provisional_zero_rf_sharpe"].to_numpy(dtype=float)
    dsr_records = []
    for strategy, path in all_paths.items():
        result = deflated_sharpe_probability(
            path.daily["net_return"].to_numpy(), trial_sharpes,
            annualization_factor=int(inference["annualization_factor"]),
        )
        dsr_records.append(
            {"strategy": strategy, **result,
             "role": "SECONDARY_MULTIPLE_TESTING_DIAGNOSTIC",
             "candidate_set": inference["dsr_candidate_set"],
             "assumption_warning": "Candidate returns and specifications are correlated; the independent-trial approximation is imperfect."}
        )
    return intervals, differences, pd.DataFrame(dsr_records), index_frame


def _regime_analysis(
    paths: dict,
    targets: dict[str, dict[pd.Timestamp, pd.Series]],
    date_inputs: dict[pd.Timestamp, DateInputs],
    market_returns: pd.Series,
    classes: pd.Series,
    config: FinalAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_config = config.section("regimes")
    dates = pd.DatetimeIndex(sorted(date_inputs))
    regimes = classify_regimes(
        market_returns, dates,
        trend_lookback=int(regime_config["trend_lookback_observations"]),
        volatility_lookback=int(regime_config["volatility_lookback_observations"]),
        annualization_factor=int(regime_config["annualization_factor"]),
    )
    regime_by_date = regimes.set_index("decision_date")["regime"]
    period_records = []
    allocation_records = []
    for strategy, path in paths.items():
        strategy_targets = targets[strategy]
        executions = {x.execution_date: x for x in path.net_executions}
        for position, date in enumerate(dates):
            next_date = dates[position + 1] if position + 1 < len(dates) else None
            mask = path.daily.index > date
            if next_date is not None:
                mask &= path.daily.index <= next_date
            holding = path.daily.loc[mask, "net_return"]
            if holding.empty:
                continue
            weights = strategy_targets[date]
            covariance = date_inputs[date].covariance
            aligned = weights.reindex(covariance.index, fill_value=0.0)
            predicted = float(np.sqrt(max(float(aligned @ covariance @ aligned), 0.0)))
            realized = float(holding.std(ddof=1) * np.sqrt(252)) if len(holding) > 1 else np.nan
            wealth = (1.0 + holding).cumprod()
            execution = executions.get(date)
            period_records.append(
                {"strategy": strategy, "decision_date": date, "regime": regime_by_date.loc[date],
                 "observations": len(holding), "net_period_return": float(wealth.iloc[-1] - 1.0),
                 "realized_volatility": realized, "predicted_common_base_volatility": predicted,
                 "prediction_error": predicted - realized,
                 "period_max_drawdown": maximum_drawdown(wealth),
                 "recurring_one_way_turnover": None if execution is None or execution.initial_formation else execution.one_way_turnover}
            )
            exposure = weights.groupby(classes.reindex(weights.index)).sum()
            for asset_class, value in exposure.items():
                allocation_records.append(
                    {"strategy": strategy, "decision_date": date,
                     "regime": regime_by_date.loc[date], "asset_class": asset_class,
                     "weight": float(value)}
                )
    periods = pd.DataFrame(period_records)
    summary_records = []
    regime_labels = [
        "calm risk-on", "volatile risk-on", "weak/cooling", "stress/risk-off"
    ]
    for strategy in paths:
        for regime in regime_labels:
            group = periods[
                (periods["strategy"] == strategy) & (periods["regime"] == regime)
            ]
            if group.empty:
                summary_records.append(
                    {"strategy": strategy, "regime": regime, "status": "NO_OBSERVATIONS",
                     "holding_periods": 0, "observations": 0, "sample_size_warning": True,
                     "compounded_net_return": np.nan, "mean_period_return": np.nan,
                     "mean_realized_volatility": np.nan, "worst_period_drawdown": np.nan,
                     "mean_recurring_one_way_turnover": np.nan,
                     "mean_predicted_common_base_volatility": np.nan,
                     "mean_prediction_error": np.nan}
                )
                continue
            total_observations = int(group["observations"].sum())
            weighted_return = float(np.prod(1.0 + group["net_period_return"]) - 1.0)
            summary_records.append(
                {"strategy": strategy, "regime": regime, "status": "OBSERVED",
                 "holding_periods": len(group), "observations": total_observations,
                 "sample_size_warning": total_observations < 126,
                 "compounded_net_return": weighted_return,
                 "mean_period_return": float(group["net_period_return"].mean()),
                 "mean_realized_volatility": float(group["realized_volatility"].mean()),
                 "worst_period_drawdown": float(group["period_max_drawdown"].min()),
                 "mean_recurring_one_way_turnover": float(group["recurring_one_way_turnover"].mean()),
                 "mean_predicted_common_base_volatility": float(group["predicted_common_base_volatility"].mean()),
                 "mean_prediction_error": float(group["prediction_error"].mean())}
            )
    allocations = pd.DataFrame(allocation_records)
    summary = pd.DataFrame(summary_records).merge(
        allocations.groupby(
            ["strategy", "regime", "asset_class"], as_index=False
        )["weight"].mean().groupby(["strategy", "regime"])["weight"].max().rename(
            "maximum_average_asset_class_exposure"
        ).reset_index(), on=["strategy", "regime"], how="left"
    )
    return regimes, periods, summary, allocations


def _sensitivity(
    date_inputs: dict[pd.Timestamp, DateInputs],
    baseline_targets: dict[str, dict[pd.Timestamp, pd.Series]],
    config: FinalAnalysisConfig,
    core_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = config.section("sensitivity")
    direct_dates = [pd.Timestamp(x) for x in config.section("direct_robustness")["selected_outer_dates"]]
    target = float(config.section("comparison")["headline_target_annual_volatility"])
    solvers = list(core_config["optimization"]["solver_order_conic"])
    records, failures = [], []
    for date in direct_dates:
        base = date_inputs[date]
        baseline = baseline_targets["box_diagonal_risk_10pct"][date]
        for rho_multiplier in [float(x) for x in sensitivity["rho_multipliers"]]:
            for kappa_multiplier in [float(x) for x in sensitivity["kappa_multipliers"]]:
                altered = DateInputs(
                    date, base.panel, base.mean, base.covariance, base.standard_errors,
                    base.mean_error_covariance, base.box_rho * rho_multiplier,
                    base.ellipsoid_rho * rho_multiplier, base.kappa * kappa_multiplier,
                )
                try:
                    result = _solve_ceiling(
                        altered, "box_diagonal", target=target,
                        maximum_weight=float(core_config["constraints"]["maximum_weight"]),
                        solver_order=solvers,
                    )
                    records.append(
                        {"sensitivity_kind": "rho_kappa_grid", "decision_date": date,
                         "model": "box_diagonal", "rho_multiplier": rho_multiplier,
                         "kappa_multiplier": kappa_multiplier, "maximum_weight": None,
                         "status": "COMPLETED", **allocation_diagnostics(result.weights, baseline),
                         "predicted_decision_volatility": result.predicted_volatility,
                         "predicted_common_base_volatility": result.common_base_volatility,
                         "robust_return": _robust_return(altered, "box_diagonal", result.weights)}
                    )
                except Exception as error:
                    failures.append(
                        {"experiment": "rho_kappa_sensitivity", "decision_date": date,
                         "model": "box_diagonal", "rho_multiplier": rho_multiplier,
                         "kappa_multiplier": kappa_multiplier,
                         "status": "FAILED_EXPLICITLY", "error": str(error)}
                    )
        for maximum_weight in [float(x) for x in sensitivity["maximum_weights"]]:
            for model, strategy in MODEL_TO_STRATEGY.items():
                try:
                    result = _solve_ceiling(
                        base, model, target=target, maximum_weight=maximum_weight,
                        solver_order=solvers,
                    )
                    records.append(
                        {"sensitivity_kind": "maximum_weight", "decision_date": date,
                         "model": model, "rho_multiplier": None, "kappa_multiplier": None,
                         "maximum_weight": maximum_weight, "status": "COMPLETED",
                         **allocation_diagnostics(result.weights, baseline_targets[strategy][date]),
                         "predicted_decision_volatility": result.predicted_volatility,
                         "predicted_common_base_volatility": result.common_base_volatility,
                         "robust_return": _robust_return(base, model, result.weights)}
                    )
                except Exception as error:
                    failures.append(
                        {"experiment": "maximum_weight_sensitivity", "decision_date": date,
                         "model": model, "maximum_weight": maximum_weight,
                         "status": "FAILED_EXPLICITLY", "error": str(error)}
                    )
    return pd.DataFrame(records), pd.DataFrame(failures)


def _headline_table(
    core_metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    differences: pd.DataFrame,
    direct: pd.DataFrame,
    config: FinalAnalysisConfig,
) -> pd.DataFrame:
    headline_cost = float(config.section("comparison")["headline_cost_bps"])
    names = config.section("comparison")["headline_strategies"]
    table = core_metrics[
        (core_metrics["cost_bps"] == headline_cost)
        & core_metrics["strategy"].isin(names)
    ].copy()
    table["risk_comparison"] = np.where(
        table["target_risk"].notna(), "COMMON EX-ANTE RISK CEILING", "NOT RISK-TARGETED"
    )
    sharpe = intervals[intervals["metric"] == "provisional_zero_rf_sharpe"]
    sharpe = sharpe[["strategy", "ci_lower", "ci_upper"]].rename(
        columns={"ci_lower": "provisional_zero_rf_sharpe_ci_lower", "ci_upper": "provisional_zero_rf_sharpe_ci_upper"}
    )
    table = table.merge(sharpe, on="strategy", how="left")
    for comparator, label in (
        ("nominal_risk_10pct", "nominal"), ("etf_equal_weight", "etf_equal_weight")
    ):
        delta = differences[
            (differences["comparator"] == comparator)
            & (differences["metric"] == "delta_provisional_zero_rf_sharpe")
        ][["strategy", "estimate", "ci_lower", "ci_upper"]].rename(
            columns={
                "estimate": f"delta_sharpe_vs_{label}",
                "ci_lower": f"delta_sharpe_vs_{label}_ci_lower",
                "ci_upper": f"delta_sharpe_vs_{label}_ci_upper",
            }
        )
        table = table.merge(delta, on="strategy", how="left")
        own = table["strategy"] == comparator
        table.loc[own, [
            f"delta_sharpe_vs_{label}", f"delta_sharpe_vs_{label}_ci_lower",
            f"delta_sharpe_vs_{label}_ci_upper",
        ]] = 0.0
    bootstrap = direct[direct["perturbation_kind"] == "training_block_bootstrap"].groupby("model").agg(
        bootstrap_weight_sensitivity=("l1_weight_change", "mean"),
        bootstrap_cosine_similarity=("cosine_similarity", "mean"),
    )
    strategy_to_model = {value: key for key, value in MODEL_TO_STRATEGY.items()}
    table["model"] = table["strategy"].map(strategy_to_model)
    table = table.merge(bootstrap, left_on="model", right_index=True, how="left")
    columns = [
        "strategy", "mean_estimator", "covariance_estimator", "robust_set",
        "risk_comparison", "target_risk", "average_predicted_decision_volatility",
        "average_predicted_common_base_volatility", "target_binding_fraction",
        "realized_volatility", "gross_annualized_return", "net_annualized_return",
        "provisional_zero_rf_sharpe", "provisional_zero_rf_sharpe_ci_lower",
        "provisional_zero_rf_sharpe_ci_upper", "delta_sharpe_vs_nominal",
        "delta_sharpe_vs_nominal_ci_lower", "delta_sharpe_vs_nominal_ci_upper",
        "delta_sharpe_vs_etf_equal_weight", "delta_sharpe_vs_etf_equal_weight_ci_lower",
        "delta_sharpe_vs_etf_equal_weight_ci_upper", "max_drawdown",
        "recurring_one_way_turnover", "initial_one_way_turnover",
        "cost_drag_final_wealth", "average_effective_holdings",
        "bootstrap_weight_sensitivity", "bootstrap_cosine_similarity",
    ]
    return table[columns].sort_values(
        "strategy", key=lambda series: series.map({name: i for i, name in enumerate(names)})
    )


def _robustness_table(
    direct: pd.DataFrame,
    direct_failures: pd.DataFrame,
    clones: pd.DataFrame,
    clone_failures: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for model in MODEL_TO_STRATEGY:
        selected = direct[direct["model"] == model]
        bootstrap = selected[selected["perturbation_kind"] == "training_block_bootstrap"]
        mean = selected[selected["perturbation_kind"] == "mean_standard_error_shock"]
        covariance = selected[selected["perturbation_kind"] == "psd_covariance_shock"]
        clone = clones[clones["model"] == model]
        completed = len(selected) + len(clone)
        failed = 0
        if not direct_failures.empty:
            failed += int((direct_failures["model"] == model).sum())
        if not clone_failures.empty:
            failed += int((clone_failures["model"] == model).sum())
        records.append(
            {"model": model,
             "bootstrap_l1_weight_sensitivity": float(bootstrap["l1_weight_change"].mean()),
             "bootstrap_cosine_similarity": float(bootstrap["cosine_similarity"].mean()),
             "bootstrap_hhi": float(bootstrap["hhi"].mean()),
             "bootstrap_effective_holdings": float(bootstrap["effective_holdings"].mean()),
             "mean_shock_l1_sensitivity": float(mean["l1_weight_change"].mean()),
             "mean_shock_asset_class_l1_sensitivity": float(mean["asset_class_exposure_l1_change"].mean()),
             "covariance_shock_l1_sensitivity": float(covariance["l1_weight_change"].mean()),
             "covariance_shock_risk_change": float(covariance["predicted_volatility_change"].mean()),
             "clone_security_level_distortion": float(clone["l1_weight_change"].mean()),
             "clone_economic_exposure_distortion": float(clone["economic_exposure_l1_change"].mean()),
             "clone_asset_class_exposure_distortion": float(clone["asset_class_exposure_l1_change"].mean()),
             "perturbed_solver_completed": completed,
             "perturbed_solver_failed": failed,
             "perturbed_solver_failure_rate": failed / (completed + failed) if completed + failed else np.nan}
        )
    return pd.DataFrame(records)


def _redundancy_table(
    cluster_dates: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    core_metrics: pd.DataFrame,
    direct: pd.DataFrame,
    config: FinalAnalysisConfig,
) -> pd.DataFrame:
    headline_cost = float(config.section("comparison")["headline_cost_bps"])
    full_names = {"gmv": "gmv_iewma", **MODEL_TO_STRATEGY}
    full_metrics = core_metrics[core_metrics["cost_bps"] == headline_cost].set_index("strategy")
    bootstrap = direct[direct["perturbation_kind"] == "training_block_bootstrap"].groupby("model")["l1_weight_change"].mean()
    records = []
    for threshold in [float(x) for x in config.section("clustering")["correlation_thresholds"]]:
        for model, strategy in full_names.items():
            dates = cluster_dates[(cluster_dates["threshold"] == threshold) & (cluster_dates["model"] == model)]
            metric = full_metrics.loc[strategy]
            records.append(
                {"universe": "FULL", "threshold": threshold, "model": model,
                 "status": "COMPLETED", "completed_dates": len(dates),
                 "average_asset_count": float(dates["full_asset_count"].mean()),
                 "average_cluster_count": float(dates["cluster_count"].mean()),
                 "median_condition_number": float(dates["full_condition_number"].median()),
                 "median_effective_rank": float(dates["full_effective_rank"].median()),
                 "average_security_level_l1_change": 0.0,
                 "average_cluster_exposure_l1_change": 0.0,
                 "bootstrap_weight_sensitivity": float(bootstrap.get(model, np.nan)),
                 "net_annualized_return": float(metric["net_annualized_return"]),
                 "realized_volatility": float(metric["realized_volatility"]),
                 "provisional_zero_rf_sharpe": float(metric["provisional_zero_rf_sharpe"]),
                 "recurring_one_way_turnover": float(metric["recurring_one_way_turnover"]),
                 "cost_drag_final_wealth": float(metric["cost_drag_final_wealth"])}
            )
    medoid = cluster_summary.copy()
    medoid["bootstrap_weight_sensitivity"] = np.nan
    medoid["bootstrap_weight_sensitivity_status"] = "NOT_ESTIMATED_FOR_MEDOID_UNIVERSE"
    medoid["universe_ablation_weight_change"] = medoid[
        "average_security_level_l1_change"
    ]
    full = pd.DataFrame(records)
    full["bootstrap_weight_sensitivity_status"] = np.where(
        full["bootstrap_weight_sensitivity"].notna(),
        "ESTIMATED_ON_SELECTED_DATES",
        "NOT_ESTIMATED",
    )
    full["universe_ablation_weight_change"] = 0.0
    combined = pd.concat([full, medoid], ignore_index=True, sort=False)
    return combined.sort_values(["threshold", "model", "universe"])


def run_final_analysis(
    config_path: Path | str,
    *,
    repository_root: Path | str,
    output_dir: Path | str | None = None,
) -> dict:
    root = Path(repository_root).resolve()
    config = FinalAnalysisConfig.load(config_path)
    core_dir, core_manifest = _validate_core(config, root)
    core_config_path = config.repository_path(root, "core_config")
    core_config = json.loads(core_config_path.read_text(encoding="utf-8"))
    provider = FrozenCsvReturnProvider(config.repository_path(root, "returns_path"))
    all_returns = pd.read_csv(provider.path, index_col=0, parse_dates=True).astype(float)
    metadata_path = config.repository_path(root, "universe_metadata_path")
    classes = _asset_classes(metadata_path, provider.assets)
    core_metrics = pd.read_csv(core_dir / "table_2_core_strategy_comparison.csv")
    core_targets = _load_core_targets(core_dir / "target_weights.csv")
    solver_dates = pd.DatetimeIndex(
        sorted(pd.read_csv(core_dir / "solver_diagnostics.csv", parse_dates=["decision_date"])["decision_date"].unique())
    )
    output = prepare_output_directory(
        output_dir or root / config.section("outputs")["default_root"] / config.sha256[:12], root
    )
    date_inputs = _build_date_inputs(provider, solver_dates, core_config)
    ceiling, attainment, attainment_targets = _risk_diagnostics(
        date_inputs, core_dir, config, core_config
    )
    direct, direct_failures = _direct_robustness(
        date_inputs, core_targets, classes, config, core_config
    )
    clones, clone_failures = _clone_experiment(
        date_inputs, core_targets, classes, config, core_config
    )
    cluster_dates, cluster_summary, cluster_failures = _cluster_experiment(
        date_inputs, core_targets, provider, classes, config, core_config
    )
    headline_names = tuple(config.section("comparison")["headline_strategies"])
    headline_cost = float(config.section("comparison")["headline_cost_bps"])
    headline_paths = _simulate_core_paths(
        provider, core_targets, strategies=headline_names, cost_bps=headline_cost,
        core_config=core_config,
    )
    all_names = sorted(core_targets)
    all_paths = _simulate_core_paths(
        provider, core_targets, strategies=all_names, cost_bps=headline_cost,
        core_config=core_config,
    )
    intervals, differences, dsr, bootstrap_indices = _inference(
        headline_paths, all_paths, core_metrics, config
    )
    regimes, regime_periods, regime_summary, regime_allocations = _regime_analysis(
        headline_paths, core_targets, date_inputs,
        all_returns[config.section("regimes")["market_proxy"]], classes, config
    )
    sensitivity, sensitivity_failures = _sensitivity(
        date_inputs, core_targets, config, core_config
    )
    overlap = _overlap_table(ceiling, core_metrics, headline_cost)
    table1 = pd.read_csv(core_dir / "table_1_covariance_estimator_study.csv")
    table2 = _headline_table(core_metrics, intervals, differences, direct, config)
    table3 = _robustness_table(direct, direct_failures, clones, clone_failures)
    table4 = _redundancy_table(cluster_dates, cluster_summary, core_metrics, direct, config)

    attainment_performance = []
    for name, targets in attainment_targets.items():
        if len(targets) != len(solver_dates):
            continue
        path = simulate_targets(
            provider, targets, strategy=name, cost_bps=headline_cost,
            maximum_weight=float(core_config["constraints"]["maximum_weight"]),
            cash_daily_return=float(core_config["annualization"]["cash_daily_return"]),
        )
        attainment_performance.append(
            {"strategy": name, **scenario_metrics(
                path, targets,
                annualization_factor=int(core_config["annualization"]["trading_days"]),
            )}
        )
    attainment_performance = pd.DataFrame(attainment_performance)

    outputs = {
        "table_1_forecasting": table1,
        "table_2_headline_strategies": table2,
        "table_3_robustness_diagnostics": table3,
        "table_4_redundancy_ablation": table4,
        "core_strategy_metrics": core_metrics,
        "covariance_forecast_periods": pd.read_csv(
            core_dir / "covariance_forecast_periods.csv"
        ),
        "common_risk_ceiling_diagnostics": ceiling,
        "target_attainment_diagnostics": attainment,
        "target_attainment_performance": attainment_performance,
        "overlapping_risk_comparison": overlap,
        "direct_robustness_observations": direct,
        "clone_diagnostics": clones,
        "clustering_date_diagnostics": cluster_dates,
        "regime_definitions": regimes,
        "regime_period_diagnostics": regime_periods,
        "regime_summary": regime_summary,
        "regime_asset_class_exposures": regime_allocations,
        "inference_intervals": intervals,
        "paired_inference_differences": differences,
        "deflated_sharpe_diagnostics": dsr,
        "sensitivity_diagnostics": sensitivity,
    }
    artifact_paths = {}
    for name, frame in outputs.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifact_paths[name] = str(path)
    headline_returns = pd.concat(
        {name: path.daily["net_return"] for name, path in headline_paths.items()}, axis=1
    )
    headline_path = output / "headline_net_returns.csv"
    headline_returns.to_csv(headline_path)
    artifact_paths["headline_net_returns"] = str(headline_path)
    bootstrap_path = output / "joint_stationary_bootstrap_indices.npz"
    np.savez_compressed(bootstrap_path, indices=bootstrap_indices)
    artifact_paths["joint_stationary_bootstrap_indices"] = str(bootstrap_path)

    failure_frames = [frame for frame in (
        direct_failures, clone_failures, cluster_failures, sensitivity_failures
    ) if not frame.empty]
    failures = pd.concat(failure_frames, ignore_index=True, sort=False) if failure_frames else pd.DataFrame(
        columns=["experiment", "status", "error"]
    )
    core_failures = pd.read_csv(core_dir / "infeasible_variants.csv")
    if not core_failures.empty:
        core_failures = core_failures.rename(columns={"failure": "error"})
        core_failures["experiment"] = "core_common_risk_ceiling"
        core_failures["status"] = "FAILED_EXPLICITLY"
        failures = pd.concat([failures, core_failures], ignore_index=True, sort=False)
    failure_path = output / "all_recorded_failures.csv"
    failures.to_csv(failure_path, index=False)
    artifact_paths["all_recorded_failures"] = str(failure_path)

    figure_paths = create_final_figures(output)
    artifact_paths.update(figure_paths)
    experiment_manifest_path = output / "experiment_manifest.json"
    target_status = attainment.groupby(["model", "status"]).size().rename("count").reset_index().to_dict(orient="records")
    write_json(
        experiment_manifest_path,
        {"result_label": config.section("experiment")["result_label"],
         "research_status": config.section("experiment")["research_status"],
         "configuration_sha256": config.sha256,
         "common_risk_ceiling_label": "COMMON EX-ANTE RISK CEILING",
         "target_attainment_status_counts": target_status,
         "failed_solve_count": len(failures),
         "dsr_candidate_count": int(dsr["candidate_count"].iloc[0]),
         "risk_free_treatment": config.section("inference")["risk_free_treatment"],
         "limitations": config.payload["limitations"]},
    )
    artifact_paths["experiment_manifest"] = str(experiment_manifest_path)
    artifact_hashes = {
        name: sha256_file(Path(path)) for name, path in sorted(artifact_paths.items())
    }
    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema_version": 3,
        "result_label": config.section("experiment")["result_label"],
        "research_status": config.section("experiment")["research_status"],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(root),
        "configuration": {"path": str(config.path), "canonical_sha256": config.sha256},
        "core_source": {
            "commit": config.section("experiment")["core_commit"],
            "configuration_sha256": core_manifest["configuration"]["canonical_sha256"],
            "artifact_directory": str(core_dir),
            "manifest_sha256": sha256_file(core_dir / "run_manifest.json"),
            "artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted(core_dir.iterdir())
                if path.is_file()
            },
        },
        "inputs": {
            "returns": {"path": str(provider.path), "sha256": provider.sha256},
            "rebalance_dates": {"path": str(config.repository_path(root, "rebalance_dates_path")),
                                "sha256": sha256_file(config.repository_path(root, "rebalance_dates_path"))},
            "universe_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
            "core_config": {"path": str(core_config_path), "sha256": sha256_file(core_config_path)},
        },
        "environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(), "platform": platform.platform(),
            "dependency_versions": dependency_versions(),
            "cvxpy_installed_solvers": cp.installed_solvers(),
        },
        "protocol": config.payload,
        "artifact_locations": artifact_paths,
        "artifact_sha256": artifact_hashes,
        "counts": {
            "outer_decisions": len(solver_dates),
            "direct_robustness_completed": len(direct),
            "clone_completed": len(clones),
            "clustering_completed_date_models": len(cluster_dates),
            "explicit_failed_solves": len(failures),
            "inference_replications": int(config.section("inference")["replications"]),
            "dsr_candidates": int(dsr["candidate_count"].iloc[0]),
        },
        "limitations": config.payload["limitations"],
    }
    write_json(manifest_path, manifest)
    return {
        "output_directory": str(output), "manifest": manifest,
        "tables": {"forecasting": table1, "headline": table2,
                   "robustness": table3, "redundancy": table4},
        "failures": failures, "target_attainment": attainment,
    }
